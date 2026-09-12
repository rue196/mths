#!/usr/bin/env python3
"""
latex_studio.py

Split‑pane LaTeX studio with the Möbius cross‑compiler.

Layout
------
    +--------------------------+---------------------------+
    |  Source editor           |  PDF / Aux / Artifact     |
    |                          |  viewer (tabbed)          |
    +--------------------------+                           |
    |  Terminal output         |                           |
    +--------------------------+---------------------------+

Features
--------
    * Open / edit / save .tex files
    * Build with pdflatex (optional; detected via PATH)
    * Möbius cross‑compile to .mob artifact (via latex_mobius_compiler)
    * Parse and display .aux files (labels, bibcite, bibdata, citations)
    * Render PDF preview (needs PyMuPDF + Pillow; text fallback otherwise)
    * Live terminal log with stdout / stderr / info / warn / ok tags

Run
---
    python latex_studio.py [file.tex]
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pymupdf

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# ============================================================
#  Optional: Möbius compiler
# ============================================================
try:
    from laTeX_compiler_mobius import LatexMobiusCompiler, CompiledArtifact
    HAS_MOBIUS = True
except Exception as _e:                                       # noqa: F841
    HAS_MOBIUS = False
    LatexMobiusCompiler = None                               # type: ignore
    CompiledArtifact = None                                  # type: ignore

# ============================================================
#  Optional: PDF rendering
# ============================================================
try:
    import fitz                                              # PyMuPDF
    from PIL import Image, ImageTk
    HAS_PDF = True
except Exception:
    HAS_PDF = False
    fitz = None                                              # type: ignore
    Image = ImageTk = None                                   # type: ignore


# ============================================================
#  .aux file parser
# ============================================================
@dataclass
class AuxEntry:
    kind: str
    name: str
    value: str = ""


class AuxFileParser:
    """Parse the common commands found in a LaTeX .aux file."""

    _NEWLABEL = re.compile(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}')
    _BIBCITE  = re.compile(r'\\bibcite\{([^}]*)\}\{([^}]*)\}')
    _BIBDATA  = re.compile(r'\\bibdata\{([^}]*)\}')
    _BIBSTYLE = re.compile(r'\\bibstyle\{([^}]*)\}')
    _CITATION = re.compile(r'\\citation\{([^}]*)\}')
    _BIBUNIT  = re.compile(r'\\bibcite|\\@input')

    @classmethod
    def parse(cls, path: str) -> List[AuxEntry]:
        entries: List[AuxEntry] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue

                m = cls._NEWLABEL.search(line)
                if m:
                    entries.append(AuxEntry(
                        "label", m.group(1),
                        f"value={m.group(2)} page={m.group(3)}"))
                    continue

                m = cls._BIBCITE.search(line)
                if m:
                    entries.append(AuxEntry("bibcite", m.group(1), m.group(2)))
                    continue

                m = cls._BIBDATA.search(line)
                if m:
                    entries.append(AuxEntry("bibdata", m.group(1)))
                    continue

                m = cls._BIBSTYLE.search(line)
                if m:
                    entries.append(AuxEntry("bibstyle", m.group(1)))
                    continue

                m = cls._CITATION.search(line)
                if m:
                    entries.append(AuxEntry("citation", m.group(1)))
                    continue

                if line.startswith("\\"):
                    entries.append(AuxEntry("other", line))
        return entries


# ============================================================
#  Sample LaTeX content
# ============================================================
SAMPLE_TEX = r"""\documentclass{article}
\usepackage{amsmath, amssymb}
\title{A Möbius Studio Sample}
\author{LaTeX Studio}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
The Möbius sieve $\mu(n)$ classifies each index as
square-free, prime-squared, or higher power.

\section{Math}
\begin{equation}
  \sum_{n=1}^{\infty} \frac{\mu(n)}{n^2}
  = \frac{6}{\pi^2}.
\end{equation}

\section{Repetition}
\begin{itemize}
  \item First item.
  \item Second item.
  \item Third item.
\end{itemize}

\end{document}
"""


# ============================================================
#  Main application
# ============================================================
class LatexStudio(tk.Tk):

    # ---------- lifecycle ----------
    def __init__(self, initial_file: Optional[str] = None) -> None:
        super().__init__()
        self.title("LaTeX Studio · Möbius Compiler")
        self.geometry("1400x900")
        self.minsize(900, 600)

        # State
        self.current_file: Optional[str] = None
        self.pdf_path: Optional[str] = None
        self.aux_path: Optional[str] = None
        self.mob_path: Optional[str] = None
        self.current_mob = None                              # CompiledArtifact
        self.pdf_doc = None
        self.pdf_page: int = 0
        self.pdf_zoom: float = 1.5
        self.pdf_photo = None                                # keep ref!

        self._build_style()
        self._build_menu()
        self._build_toolbar()
        self._build_layout()
        self._bind_shortcuts()

        # Banner
        self.log("LaTeX Studio ready.", "info")
        self.log(f"  Möbius compiler : "
                 f"{'available' if HAS_MOBIUS else 'NOT FOUND'}", 
                 "info" if HAS_MOBIUS else "warn")
        self.log(f"  PDF viewer      : "
                 f"{'PyMuPDF available' if HAS_PDF else 'unavailable'}", 
                 "info" if HAS_PDF else "warn")
        self.log(f"  pdflatex        : "
                 f"{'available' if shutil.which('pdflatex') else 'NOT FOUND'}", 
                 "info" if shutil.which('pdflatex') else "warn")

        # Load initial file
        if initial_file and Path(initial_file).exists():
            self._load_file(initial_file)
        else:
            self.editor.insert("1.0", SAMPLE_TEX)

    # ---------- styles ----------
    def _build_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))

    # ---------- menu ----------
    def _build_menu(self) -> None:
        m = tk.Menu(self)

        file_m = tk.Menu(m, tearoff=0)
        file_m.add_command(label="New",        accelerator="Ctrl+N", command=self.new_file)
        file_m.add_command(label="Open…",     accelerator="Ctrl+O", command=self.open_file)
        file_m.add_command(label="Save",       accelerator="Ctrl+S", command=self.save_file)
        file_m.add_command(label="Save As…",   accelerator="Ctrl+Shift+S", command=self.save_file_as)
        file_m.add_separator()
        file_m.add_command(label="Exit",       command=self._on_close)
        m.add_cascade(label="File", menu=file_m)

        build_m = tk.Menu(m, tearoff=0)
        build_m.add_command(label="Compile (pdflatex)", accelerator="F5",
                            command=self.build_latex)
        build_m.add_command(label="Möbius compile",     accelerator="F6",
                            command=self.compile_mobius)
        build_m.add_command(label="Reload PDF",         accelerator="F7",
                            command=self.pdf_reload)
        build_m.add_command(label="Reload .aux",        command=self.aux_reload)
        m.add_cascade(label="Build", menu=build_m)

        view_m = tk.Menu(m, tearoff=0)
        view_m.add_command(label="PDF tab",      command=lambda: self._select_tab("PDF"))
        view_m.add_command(label="Aux tab",      command=lambda: self._select_tab("Aux"))
        view_m.add_command(label="Artifact tab", command=lambda: self._select_tab("Artifact"))
        m.add_cascade(label="View", menu=view_m)

        help_m = tk.Menu(m, tearoff=0)
        help_m.add_command(label="About", command=self._about)
        m.add_cascade(label="Help", menu=help_m)

        self.config(menu=m)

    # ---------- toolbar ----------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x", padx=4, pady=(4, 0))

        ttk.Button(bar, text="📄 New",     command=self.new_file).pack(side="left", padx=2)
        ttk.Button(bar, text="📂 Open",    command=self.open_file).pack(side="left", padx=2)
        ttk.Button(bar, text="💾 Save",    command=self.save_file).pack(side="left", padx=2)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(bar, text="▶ pdflatex", command=self.build_latex).pack(side="left", padx=2)
        ttk.Button(bar, text="♾ Möbius",   command=self.compile_mobius).pack(side="left", padx=2)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(bar, text="🔄 Reload PDF", command=self.pdf_reload).pack(side="left", padx=2)
        ttk.Button(bar, text="🗂 Reload .aux", command=self.aux_reload).pack(side="left", padx=2)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)

        ttk.Button(bar, text="📦 Load .mob",  command=self.load_mob).pack(side="left", padx=2)
        ttk.Button(bar, text="🔓 Decompile",  command=self.decompile_mob).pack(side="left", padx=2)

        self.file_label = ttk.Label(bar, text="(no file)", style="Status.TLabel")
        self.file_label.pack(side="right", padx=8)

    # ---------- layout ----------
    def _build_layout(self) -> None:
        # Outer horizontal split
        outer = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ---- Left column: editor + terminal ----
        left = ttk.PanedWindow(outer, orient=tk.VERTICAL)
        outer.add(left, weight=1)

        # Editor
        ed_frame = ttk.Frame(left)
        ed_frame.rowconfigure(1, weight=1)
        ed_frame.columnconfigure(0, weight=1)
        ttk.Label(ed_frame, text="Source", style="Header.TLabel") \
            .grid(row=0, column=0, sticky="w", padx=4, pady=(2, 0))
        self.editor = ScrolledText(ed_frame, wrap="none", undo=True,
                                   font=("Consolas", 11),
                                   bg="#fdfaf3", fg="#222222",
                                   insertbackground="#222222")
        self.editor.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        left.add(ed_frame, weight=3)

        # Terminal
        term_frame = ttk.Frame(left)
        term_frame.rowconfigure(1, weight=1)
        term_frame.columnconfigure(0, weight=1)
        hdr = ttk.Frame(term_frame)
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Terminal", style="Header.TLabel").pack(side="left", padx=4)
        ttk.Button(hdr, text="Clear", command=self.clear_terminal).pack(side="right", padx=4)
        self.terminal = ScrolledText(term_frame, wrap="word",
                                     font=("Consolas", 10),
                                     bg="#1e1e1e", fg="#d4d4d4",
                                     insertbackground="white",
                                     state="disabled")
        self.terminal.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        self.terminal.tag_config("out",  foreground="#d4d4d4")
        self.terminal.tag_config("info", foreground="#569cd6")
        self.terminal.tag_config("warn", foreground="#dcdcaa")
        self.terminal.tag_config("ok",   foreground="#4ec9b0")
        self.terminal.tag_config("err",  foreground="#f48771")
        left.add(term_frame, weight=1)

        # ---- Right column: notebook ----
        self.right_nb = ttk.Notebook(outer)
        outer.add(self.right_nb, weight=2)

        self._build_pdf_tab()
        self._build_aux_tab()
        self._build_art_tab()

    def _build_pdf_tab(self) -> None:
        frame = ttk.Frame(self.right_nb)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tb = ttk.Frame(frame)
        tb.grid(row=0, column=0, sticky="ew", pady=2, padx=4)
        ttk.Button(tb, text="◀", width=3, command=self.pdf_prev).pack(side="left", padx=1)
        ttk.Button(tb, text="▶", width=3, command=self.pdf_next).pack(side="left", padx=1)
        self.pdf_page_label = ttk.Label(tb, text="— / —")
        self.pdf_page_label.pack(side="left", padx=8)

        ttk.Button(tb, text="−", width=3,
                   command=lambda: self.pdf_zoom_by(1 / 1.2)).pack(side="left", padx=1)
        ttk.Button(tb, text="+", width=3,
                   command=lambda: self.pdf_zoom_by(1.2)).pack(side="left", padx=1)
        self.pdf_zoom_label = ttk.Label(tb, text=f"{self.pdf_zoom:.2f}×")
        self.pdf_zoom_label.pack(side="left", padx=8)

        ttk.Button(tb, text="Fit", command=self.pdf_fit).pack(side="left", padx=4)
        ttk.Button(tb, text="Reload", command=self.pdf_reload).pack(side="right", padx=2)

        self.pdf_canvas = tk.Canvas(frame, bg="#2d2d2d", highlightthickness=0)
        self.pdf_canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 0), pady=(0, 4))

        sb_v = ttk.Scrollbar(frame, orient="vertical", command=self.pdf_canvas.yview)
        sb_v.grid(row=1, column=1, sticky="ns", pady=(0, 4))
        sb_h = ttk.Scrollbar(frame, orient="horizontal", command=self.pdf_canvas.xview)
        sb_h.grid(row=2, column=0, sticky="ew", padx=(4, 0))
        self.pdf_canvas.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)

        self.right_nb.add(frame, text="PDF")

    def _build_aux_tab(self) -> None:
        frame = ttk.Frame(self.right_nb)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tb = ttk.Frame(frame)
        tb.grid(row=0, column=0, sticky="ew", pady=2, padx=4)
        ttk.Label(tb, text=".aux entries").pack(side="left")
        ttk.Button(tb, text="Reload", command=self.aux_reload).pack(side="right", padx=2)

        cols = ("kind", "name", "value")
        self.aux_tree = ttk.Treeview(frame, columns=cols, show="headings")
        self.aux_tree.heading("kind",  text="Kind")
        self.aux_tree.heading("name",  text="Name")
        self.aux_tree.heading("value", text="Value")
        self.aux_tree.column("kind",  width=90,  stretch=False, anchor="w")
        self.aux_tree.column("name",  width=220, stretch=False, anchor="w")
        self.aux_tree.column("value", width=500, stretch=True,  anchor="w")
        self.aux_tree.grid(row=1, column=0, sticky="nsew", padx=(4, 0), pady=(0, 4))

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.aux_tree.yview)
        sb.grid(row=1, column=1, sticky="ns", pady=(0, 4))
        self.aux_tree.configure(yscrollcommand=sb.set)

        self.right_nb.add(frame, text="Aux")

    def _build_art_tab(self) -> None:
        frame = ttk.Frame(self.right_nb)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        tb = ttk.Frame(frame)
        tb.grid(row=0, column=0, sticky="ew", pady=2, padx=4)
        ttk.Label(tb, text="Möbius artifact").pack(side="left")
        ttk.Button(tb, text="Load .mob", command=self.load_mob).pack(side="right", padx=2)
        ttk.Button(tb, text="Decompile (lossy)",
                   command=self.decompile_mob).pack(side="right", padx=2)

        self.art_text = ScrolledText(frame, wrap="word", font=("Consolas", 10),
                                     bg="#fdfaf3", fg="#222222")
        self.art_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))

        self.right_nb.add(frame, text="Artifact")

    # ---------- shortcuts ----------
    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-n>", lambda e: self.new_file())
        self.bind_all("<Control-o>", lambda e: self.open_file())
        self.bind_all("<Control-s>", lambda e: self.save_file())
        self.bind_all("<Control-S>", lambda e: self.save_file_as())
        self.bind_all("<F5>",        lambda e: self.build_latex())
        self.bind_all("<F6>",        lambda e: self.compile_mobius())
        self.bind_all("<F7>",        lambda e: self.pdf_reload())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ============================================================
    #  Terminal logging
    # ============================================================
    def log(self, text: str, tag: str = "out") -> None:
        self.terminal.configure(state="normal")
        self.terminal.insert(tk.END, text + "\n", tag)
        self.terminal.see(tk.END)
        self.terminal.configure(state="disabled")

    def clear_terminal(self) -> None:
        self.terminal.configure(state="normal")
        self.terminal.delete("1.0", tk.END)
        self.terminal.configure(state="disabled")

    # ============================================================
    #  File handling
    # ============================================================
    def new_file(self) -> None:
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", SAMPLE_TEX)
        self.current_file = None
        self.pdf_path = None
        self.aux_path = None
        self.mob_path = None
        self.current_mob = None
        self._refresh_file_label()
        self.title("LaTeX Studio · Möbius Compiler")
        self.log("New file.", "info")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open LaTeX file",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")])
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", Path(path).read_text(encoding="utf-8"))
            self.current_file = path
            self.title(f"LaTeX Studio · {Path(path).name}")
            self._refresh_file_label()
            self.log(f"Opened: {path}", "ok")

            # auto‑detect siblings
            p = Path(path)
            aux = p.with_suffix(".aux")
            if aux.exists():
                self.aux_path = str(aux)
                self.aux_reload(silent=True)
            pdf = p.with_suffix(".pdf")
            if pdf.exists():
                self.pdf_path = str(pdf)
                self.pdf_reload(silent=True)
            mob = p.with_suffix(".tex.mob")
            if not mob.exists():
                mob = p.with_suffix(".mob")
            if mob.exists():
                self.mob_path = str(mob)
                self._load_mob_path(str(mob), silent=True)
        except Exception as e:
            self.log(f"Open error: {e}", "err")
            messagebox.showerror("Open error", str(e))

    def save_file(self) -> bool:
        if not self.current_file:
            return self.save_file_as()
        try:
            Path(self.current_file).write_text(
                self.editor.get("1.0", "end-1c"), encoding="utf-8")
            self.log(f"Saved: {self.current_file}", "ok")
            self._refresh_file_label()
            return True
        except Exception as e:
            self.log(f"Save error: {e}", "err")
            messagebox.showerror("Save error", str(e))
            return False

    def save_file_as(self) -> bool:
        path = filedialog.asksaveasfilename(
            title="Save LaTeX file",
            defaultextension=".tex",
            filetypes=[("LaTeX files", "*.tex"), ("All files", "*.*")])
        if not path:
            return False
        self.current_file = path
        if self.save_file():
            self.title(f"LaTeX Studio · {Path(path).name}")
            return True
        return False

    def _refresh_file_label(self) -> None:
        if self.current_file:
            self.file_label.config(text=Path(self.current_file).name)
        else:
            self.file_label.config(text="(no file)")

    # ============================================================
    #  pdflatex build
    # ============================================================
    def build_latex(self) -> None:
        if not self.current_file:
            if not self.save_file_as():
                return
        if not shutil.which("pdflatex"):
            self.log("pdflatex not found — install TeX Live / MiKTeX.", "err")
            messagebox.showerror("pdflatex not found",
                                 "Please install a LaTeX distribution "
                                 "and make sure pdflatex is on PATH.")
            return
        if not self.save_file():
            return

        tex = Path(self.current_file)
        self.log(f"$ pdflatex -interaction=nonstopmode {tex.name}",
                 "info")

        def worker():
            try:
                proc = subprocess.Popen(
                    ["pdflatex", "-interaction=nonstopmode",
                     "-halt-on-error", tex.name],
                    cwd=str(tex.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for line in proc.stdout:                      # type: ignore
                    self.after(0, self.log, line.rstrip("\n"), "out")
                proc.wait()
                rc = proc.returncode
            except Exception as e:
                self.after(0, self.log, f"pdflatex exception: {e}", "err")
                rc = -1
            self.after(0, self._after_pdflatex, rc)

        threading.Thread(target=worker, daemon=True).start()

    def _after_pdflatex(self, rc: int) -> None:
        if rc == 0:
            self.log("pdflatex: OK", "ok")
            pdf = Path(self.current_file).with_suffix(".pdf")
            if pdf.exists():
                self.pdf_path = str(pdf)
                self.pdf_reload()
            aux = Path(self.current_file).with_suffix(".aux")
            if aux.exists():
                self.aux_path = str(aux)
                self.aux_reload()
        else:
            self.log(f"pdflatex: FAILED (rc={rc})", "err")

    # ============================================================
    #  Möbius compile
    # ============================================================
    def compile_mobius(self) -> None:
        if not HAS_MOBIUS:
            self.log("Möbius compiler module not available.", "err")
            messagebox.showerror(
                "Möbius compiler unavailable",
                "Import of latex_mobius_compiler failed.\n"
                "Place latex_mobius_compiler.py next to this file.")
            return
        if not self.current_file:
            if not self.save_file_as():
                return
        if not self.save_file():
            return

        tex = Path(self.current_file)
        try:
            src = tex.read_text(encoding="utf-8")
        except Exception as e:
            self.log(f"read error: {e}", "err")
            return

        self.log(f"# Möbius compile: {tex.name}", "info")

        def worker():
            try:
                comp = LatexMobiusCompiler(max_K=8192,
                                           gate="log",
                                           lossless=True)
                art = comp.compile(src)
                mob = tex.with_suffix(".tex.mob")
                comp.write_artifact(art, str(mob))
                self.after(0, self._after_mobius, art, str(mob))
            except Exception as e:
                self.after(0, self.log, f"Möbius error: {e}", "err")

        threading.Thread(target=worker, daemon=True).start()

    def _after_mobius(self, art, mob_path: str) -> None:
        self.current_mob = art
        self.mob_path = mob_path
        self.log(f"Möbius artifact → {mob_path}", "ok")
        self.log(f"  S = {art.S:+.6f}   H = {art.H:.6f}   m = {art.m:.6f}",
                 "info")
        self.log(f"  K = {art.K} tokens · vocabulary {len(art.vocab)} · "
                 f"kept {len(art.kept)}", "info")
        self.log(f"  fingerprint = {art.fingerprint[:32]}…", "info")
        self._show_artifact(art, mob_path)

    # ============================================================
    #  Aux tab
    # ============================================================
    def aux_reload(self, silent: bool = False) -> None:
        if not self.aux_path or not Path(self.aux_path).exists():
            if not silent:
                self.log("No .aux file found.", "warn")
            return
        try:
            entries = AuxFileParser.parse(self.aux_path)
        except Exception as e:
            self.log(f".aux parse error: {e}", "err")
            return

        for item in self.aux_tree.get_children():
            self.aux_tree.delete(item)
        for e in entries:
            self.aux_tree.insert("", "end", values=(e.kind, e.name, e.value))

        self.log(f"Loaded {len(entries)} .aux entries from "
                 f"{Path(self.aux_path).name}", "ok")
        if not silent:
            self._select_tab("Aux")

    # ============================================================
    #  PDF tab
    # ============================================================
    def pdf_reload(self, silent: bool = False) -> None:
        if not self.pdf_path or not Path(self.pdf_path).exists():
            if not silent:
                self.log("No PDF to show.", "warn")
            self.pdf_canvas.delete("all")
            self.pdf_page_label.config(text="— / —")
            return

        if not HAS_PDF:
            self.pdf_canvas.delete("all")
            self.pdf_canvas.create_text(
                20, 20, anchor="nw", fill="#d4d4d4",
                font=("Consolas", 12),
                text=(f"PDF file: {self.pdf_path}\n\n"
                      "PyMuPDF (fitz) and Pillow are required for rendering.\n"
                      "Install with:\n"
                      "    pip install pymupdf pillow"))
            if not silent:
                self.log("PyMuPDF not installed — cannot render PDF.", "warn")
            return

        try:
            if self.pdf_doc:
                try:
                    self.pdf_doc.close()
                except Exception:
                    pass
            self.pdf_doc = fitz.open(self.pdf_path)
            self.pdf_page = min(self.pdf_page,
                                max(0, len(self.pdf_doc) - 1))
            self._render_pdf_page()
            self.log(f"PDF loaded: {len(self.pdf_doc)} pages "
                     f"({Path(self.pdf_path).name})", "ok")
            if not silent:
                self._select_tab("PDF")
        except Exception as e:
            self.log(f"PDF load error: {e}", "err")

    def _render_pdf_page(self) -> None:
        if not self.pdf_doc:
            return
        page = self.pdf_doc[self.pdf_page]
        mat = fitz.Matrix(self.pdf_zoom, self.pdf_zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        self.pdf_photo = ImageTk.PhotoImage(img)

        self.pdf_canvas.delete("all")
        self.pdf_canvas.create_image(0, 0, anchor="nw", image=self.pdf_photo)
        self.pdf_canvas.configure(scrollregion=(0, 0, pix.width, pix.height))

        self.pdf_page_label.config(
            text=f"{self.pdf_page + 1} / {len(self.pdf_doc)}")
        self.pdf_zoom_label.config(text=f"{self.pdf_zoom:.2f}×")

    def pdf_prev(self) -> None:
        if not self.pdf_doc:
            return
        if self.pdf_page > 0:
            self.pdf_page -= 1
            self._render_pdf_page()

    def pdf_next(self) -> None:
        if not self.pdf_doc:
            return
        if self.pdf_page < len(self.pdf_doc) - 1:
            self.pdf_page += 1
            self._render_pdf_page()

    def pdf_zoom_by(self, factor: float) -> None:
        self.pdf_zoom = max(0.2, min(6.0, self.pdf_zoom * factor))
        self._render_pdf_page()

    def pdf_fit(self) -> None:
        """Fit width of the current page to the canvas."""
        if not self.pdf_doc:
            return
        page = self.pdf_doc[self.pdf_page]
        cw = max(1, self.pdf_canvas.winfo_width())
        pw = max(1, page.rect.width)
        self.pdf_zoom = max(0.2, min(6.0, cw / pw))
        self._render_pdf_page()

    # ============================================================
    #  Artifact tab
    # ============================================================
    def _load_mob_path(self, path: str, silent: bool = False) -> None:
        if not HAS_MOBIUS:
            if not silent:
                self.log("Möbius compiler module not available.", "err")
            return
        try:
            comp = LatexMobiusCompiler()
            art = comp.read_artifact(path)
            self.current_mob = art
            self.mob_path = path
            self._show_artifact(art, path)
        except Exception as e:
            self.log(f"load .mob error: {e}", "err")

    def load_mob(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Möbius artifact",
            filetypes=[("Möbius artifacts", "*.mob"),
                       ("All files", "*.*")])
        if path:
            self._load_mob_path(path)

    def _show_artifact(self, art, path: str) -> None:
        lines = []
        lines.append(f"File:            {path}")
        lines.append(f"Fingerprint:     {art.fingerprint}")
        lines.append(f"K (tokens):      {art.K}")
        lines.append(f"Gate:            {art.gate}")
        lines.append(f"Elliptic TSP:    {art.use_elliptic}")
        lines.append(f"Supertrace S:    {art.S:+.6f}")
        lines.append(f"Entropy H:       {art.H:.6f}")
        lines.append(f"Mass m:          {art.m:.6f}")
        lines.append(f"Vocabulary size: {len(art.vocab)}")
        lines.append(f"Kept coeffs:     {len(art.kept)}")
        lines.append("")
        lines.append("Extra:")
        for k, v in art.extra.items():
            if k == "sub_artifacts":
                lines.append(f"  {k}: {len(v)} sub‑artifacts")
            else:
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("Kept (first 30):")
        for i, v in art.kept[:30]:
            lines.append(f"  idx={i:6d}  val={v:+.8f}")

        self.art_text.delete("1.0", tk.END)
        self.art_text.insert("1.0", "\n".join(lines))
        self._select_tab("Artifact")

    def decompile_mob(self) -> None:
        if not HAS_MOBIUS:
            self.log("Möbius compiler module not available.", "err")
            return
        if self.current_mob is None:
            self.log("No artifact loaded.", "warn")
            return
        try:
            comp = LatexMobiusCompiler()
            text = comp.decompile(self.current_mob, lossless=False)
            self.editor.delete("1.0", tk.END)
            self.editor.insert("1.0", text)
            self.log("Decompiled artifact (lossy) → editor.", "ok")
        except Exception as e:
            self.log(f"Decompile error: {e}", "err")

    # ============================================================
    #  Misc
    # ============================================================
    def _select_tab(self, name: str) -> None:
        for i in range(self.right_nb.index("end")):
            if self.right_nb.tab(i, "text") == name:
                self.right_nb.select(i)
                return

    def _about(self) -> None:
        messagebox.showinfo(
            "About LaTeX Studio",
            "LaTeX Studio · Möbius Compiler\n\n"
            "A split‑pane IDE with a Möbius cross‑compiler and "
            "a .aux file reader.\n\n"
            "Shortcuts:\n"
            "  Ctrl+N  New\n"
            "  Ctrl+O  Open\n"
            "  Ctrl+S  Save\n"
            "  F5      Compile with pdflatex\n"
            "  F6      Möbius compile\n"
            "  F7      Reload PDF")

    def _on_close(self) -> None:
        try:
            if self.pdf_doc:
                self.pdf_doc.close()
        except Exception:
            pass
        self.destroy()


# ============================================================
#  Entry point
# ============================================================
def main(argv: List[str]) -> None:
    app = LatexStudio(initial_file=(argv[0] if argv else None))
    app.mainloop()


if __name__ == "__main__":
    main(sys.argv[1:])