#!/usr/bin/env python3
"""
local_llm_interface.py
======================

A simple local LLM interface with O(K) memory.

Layout on disk
--------------
    <models_root>/
        llama-3.2-3b/
            config.json           or  Modelfile  or  *.gguf
        mistral-7b/
            config.json
        phi-3-mini/
            model.safetensors.index.json
        ...

Everything above this directory level is treated as a "model folder".

Memory model
------------
All storage is bounded by a single constant K:

    • model registry         exactly K slots (μ(k) ≠ 0 placements)
    • tokenizer cache        bounded LRU of size K
    • context window         bounded deque of size K
    • per-token state        fixed K-vector

Nothing is O(K²), nothing grows with the number of models or tokens.

Interface
---------
    iface = LocalLLMInterface(root="~/models", K=512)

    iface.list_models()                 -> [name, ...]
    iface.describe(name)                -> dict
    iface.fingerprint(name)             -> hex
    iface.ask(name, prompt)             -> str
    iface.stream(name, prompt)          -> iterator of tokens

The interface is backend-agnostic.  Two backends are provided:

    MockBackend   — a small deterministic O(K) model that needs no
                    external process.  Useful for testing the pipeline.

    OllamaBackend — talks to a running Ollama server on
                    http://localhost:11434 to drive real models.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import struct
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)                # ≈ 2.362
NORM    = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY = 6.0 / (PI * PI)               # ≈ 0.6079

K_DEFAULT = 512


# ============================================================
#  Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    if K < 1:
        return [0] * (K + 1)
    mu = [1] * (K + 1)
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                is_comp[j] = True
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[0] = 0
    return mu


# ============================================================
#  Model entry
# ============================================================
@dataclass
class ModelEntry:
    """
    A single model discovered under the root directory.

        name         folder name (identifier for the interface)
        path         absolute path to the model folder
        size_bytes   total bytes of all files inside (bounded walk)
        kind         'config' | 'modelfile' | 'gguf' | 'safetensors' | 'unknown'
        metadata     parsed config.json / Modelfile headers / GGUF header
        fingerprint  hex string of the folder contents
    """
    name: str
    path: str
    size_bytes: int = 0
    kind: str = "unknown"
    metadata: Dict = field(default_factory=dict)
    fingerprint: str = ""

    def summary(self) -> Dict:
        return dict(
            name=self.name,
            path=self.path,
            size_bytes=self.size_bytes,
            kind=self.kind,
            fingerprint=self.fingerprint[:16] + "…" if self.fingerprint else "",
            n_files=self.metadata.get("n_files", 0),
        )


# ============================================================
#  Directory scanner
# ============================================================
_MODEL_FILE_HINTS = ("config.json", "Modelfile", "model.safetensors.index.json")
_GGUF_RE     = re.compile(r"\.gguf$", re.IGNORECASE)
_SAFETENSORS = re.compile(r"\.safetensors$", re.IGNORECASE)


def _read_config(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_modelfile(path: str) -> Dict:
    """Parse a Modelfile into a dict of top-level directives."""
    out: Dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2:
                    out.setdefault(parts[0].upper(), []).append(parts[1])
    except Exception:
        pass
    return out


def _read_gguf_header(path: str) -> Dict:
    """Read the first 64 bytes of a GGUF file for a light header probe."""
    out: Dict = {}
    try:
        with open(path, "rb") as f:
            head = f.read(64)
        if len(head) >= 8 and head[:4] == b"GGUF":
            version, n_tensors = struct.unpack_from("<II", head, 4)
            out["gguf_version"]  = version
            out["n_tensors"]     = n_tensors
    except Exception:
        pass
    return out


def _fingerprint_folder(path: str, max_files: int = 64) -> Tuple[str, int, int]:
    """
    SHA-256 fingerprint of a bounded walk of the folder.
    Returns (hex, n_files, total_bytes).
    """
    h = hashlib.sha256()
    n_files = 0
    total_bytes = 0
    for root, dirs, files in os.walk(path):
        dirs.sort()
        files.sort()
        for name in files:
            if n_files >= max_files:
                break
            full = os.path.join(root, name)
            rel  = os.path.relpath(full, path)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            total_bytes += size
            h.update(rel.encode("utf-8"))
            h.update(struct.pack(">Q", size))
            n_files += 1
        if n_files >= max_files:
            break
    return h.hexdigest(), n_files, total_bytes


class ModelDirectoryScanner:
    """
    Walks the root directory and returns a list of ModelEntry.
    Cost is bounded by `max_models` per scan.
    """
    def __init__(self, root: str, max_models: int = 256,
                 max_files_per_model: int = 64):
        self.root = os.path.expanduser(root)
        self.max_models = int(max_models)
        self.max_files_per_model = int(max_files_per_model)

    def scan(self) -> List[ModelEntry]:
        entries: List[ModelEntry] = []
        if not os.path.isdir(self.root):
            return entries
        for name in sorted(os.listdir(self.root)):
            if len(entries) >= self.max_models:
                break
            path = os.path.join(self.root, name)
            if not os.path.isdir(path):
                continue
            entries.append(self._classify(name, path))
        return entries

    def _classify(self, name: str, path: str) -> ModelEntry:
        # 1. look for config.json / Modelfile / safetensors index
        kind = "unknown"
        metadata: Dict = {}
        for hint in _MODEL_FILE_HINTS:
            full = os.path.join(path, hint)
            if os.path.exists(full):
                if hint == "config.json":
                    metadata = _read_config(full) or {}
                    kind = "config"
                elif hint == "Modelfile":
                    metadata = _read_modelfile(full)
                    kind = "modelfile"
                else:
                    metadata = _read_config(full) or {}
                    kind = "safetensors"
                break

        # 2. fall back to scanning for a GGUF file
        if kind == "unknown":
            for fname in os.listdir(path):
                if _GGUF_RE.search(fname):
                    metadata = _read_gguf_header(os.path.join(path, fname))
                    metadata["gguf_file"] = fname
                    kind = "gguf"
                    break

        # 3. fingerprint
        fp, n_files, total_bytes = _fingerprint_folder(
            path, max_files=self.max_files_per_model)
        metadata["n_files"] = n_files

        return ModelEntry(
            name=name, path=path,
            size_bytes=total_bytes, kind=kind,
            metadata=metadata, fingerprint=fp,
        )


# ============================================================
#  Möbius model memory  ·  O(K)
# ============================================================
class MobiusModelMemory:
    """
    Fixed K-slot memory for model entries.

        slot k     ⟷   model entry, placed at the first μ(k) ≠ 0 index
        capacity   = number of square-free positions ≤ K
        cost       O(K) time and space
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)
        self.slots: List[Optional[ModelEntry]] = [None] * (K + 1)
        self.kept_indices: List[int] = [k for k in range(1, K + 1)
                                        if self.mu[k] != 0]
        self._next_slot = 0

    @property
    def capacity(self) -> int:
        return len(self.kept_indices)

    def insert(self, entry: ModelEntry) -> int:
        if self._next_slot >= self.capacity:
            raise RuntimeError("Möbius model memory is full")
        k = self.kept_indices[self._next_slot]
        self.slots[k] = entry
        self._next_slot += 1
        return k

    def get_by_name(self, name: str) -> Optional[ModelEntry]:
        for k in self.kept_indices[:self._next_slot]:
            e = self.slots[k]
            if e is not None and e.name == name:
                return e
        return None

    def entries(self) -> List[Tuple[int, ModelEntry]]:
        out = []
        for k in self.kept_indices[:self._next_slot]:
            e = self.slots[k]
            if e is not None:
                out.append((k, e))
        return out

    def __len__(self) -> int:
        return self._next_slot


# ============================================================
#  Tokenizer  ·  O(K) with a bounded cache
# ============================================================
class BoundedTokenizer:
    """
    A deterministic O(K) tokenizer with a bounded LRU cache.

        tokenize(text) -> List[int]      indices in [0, K)
        detokenize(ids) -> str

    The cache holds at most `K` tokens.  No vocabulary grows past K.
    """
    def __init__(self, K: int = K_DEFAULT, cache_size: Optional[int] = None):
        self.K = K
        self.cache_size = cache_size or K
        self.cache: "OrderedDict[str, int]" = OrderedDict()
        self.reverse: Dict[int, str] = {}
        self._next_id = 1                # 0 = <UNK>

    def _hash_id(self, token: str) -> int:
        h = int(hashlib.md5(token.encode()).digest()[:4].hex(), 16)
        return h % self.K

    def tokenize(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in re.findall(r"\w+|[.,!?;:'\"]", text.lower().strip()):
            if tok in self.cache:
                self.cache.move_to_end(tok)
                ids.append(self.cache[tok])
                continue
            # use hash bucket as the id, keep the map bounded
            tid = self._hash_id(tok)
            self.cache[tok] = tid
            self.reverse[tid] = tok
            if len(self.cache) > self.cache_size:
                old, _ = self.cache.popitem(last=False)
                # do not delete the reverse entry (it is harmless)
            ids.append(tid)
        return ids

    def detokenize(self, ids: List[int]) -> str:
        return " ".join(self.reverse.get(i, f"<{i}>") for i in ids)


# ============================================================
#  Backends
# ============================================================
class MockBackend:
    """
    A deterministic, O(K) mock model.  Uses the chip convolution on
    the token-id sequence and returns the most likely next token ids
    converted back to text.  No external process required.
    """
    def __init__(self, K: int, tokenizer: BoundedTokenizer,
                 seed: int = 0):
        self.K = K
        self.tok = tokenizer
        self.mu = mobius_sieve(K)
        self.rng = __import__("random").Random(seed)
        self._vocab = (
            "the quick brown fox jumps over a lazy dog".split()
            + "it is a truth universally acknowledged that".split()
            + "in the beginning there was a signal".split()
            + "love is the quiet space where two silences agree".split()
        )

    def _next_token(self, context: List[int], step: int) -> str:
        # deterministic hash-based pick that depends on the context
        h = hashlib.md5(
            (",".join(map(str, context[-self.K:])) + f"|{step}").encode()
        ).digest()
        idx = int.from_bytes(h[:4], "big") % len(self._vocab)
        return self._vocab[idx]

    def generate(self, prompt: str, max_tokens: int = 32) -> Iterator[str]:
        ctx = self.tok.tokenize(prompt)
        for step in range(max_tokens):
            nxt = self._next_token(ctx, step)
            ctx.append(self.tok.tokenize(nxt)[0])
            if len(ctx) > self.K:
                ctx = ctx[-self.K:]
            yield nxt


class OllamaBackend:
    """
    Bridges to a running Ollama server.  Requires `requests`.
    The model name is the folder name as registered in Ollama.
    """
    def __init__(self, host: str = "http://localhost:11434",
                 timeout: float = 60.0):
        self.host = host.rstrip("/")
        self.timeout = timeout
        try:
            import requests  # noqa: F401
        except ImportError:
            raise RuntimeError("OllamaBackend requires `pip install requests`")

    def generate(self, model: str, prompt: str,
                 max_tokens: int = 256) -> Iterator[str]:
        import requests
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        with requests.post(url, json=payload,
                           stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("response", "")
                if piece:
                    yield piece
                if chunk.get("done", False):
                    break


# ============================================================
#  Local LLM Interface
# ============================================================
class LocalLLMInterface:
    """
    Main entry point.

        iface = LocalLLMInterface(root="~/models", K=512)
        iface.list_models()
        iface.ask("llama-3.2-3b", "What is 2 + 2?")
        for tok in iface.stream("llama-3.2-3b", "Hello"):
            print(tok, end="", flush=True)
    """

    def __init__(self,
                 root: str,
                 K: int = K_DEFAULT,
                 backend: str = "mock",          # 'mock' | 'ollama'
                 ollama_host: str = "http://localhost:11434",
                 seed: int = 0):
        self.root = os.path.expanduser(root)
        self.K = int(K)
        self.backend_name = backend

        # --- O(K) storage ---
        self.memory   = MobiusModelMemory(K=self.K)
        self.tokenizer = BoundedTokenizer(K=self.K)

        # --- O(K) context window (bounded deque) ---
        self.context: deque = deque(maxlen=self.K)

        # --- backend ---
        if backend == "mock":
            self.backend = MockBackend(self.K, self.tokenizer, seed=seed)
        elif backend == "ollama":
            self.backend = OllamaBackend(host=ollama_host)
        else:
            raise ValueError(f"unknown backend: {backend}")

        # --- discovery ---
        t0 = time.perf_counter()
        self.scanner = ModelDirectoryScanner(self.root)
        self._register()
        self.scan_ms = (time.perf_counter() - t0) * 1e3

    # ---------- discovery ----------
    def _register(self) -> None:
        entries = self.scanner.scan()
        for e in entries:
            try:
                self.memory.insert(e)
            except RuntimeError:
                break        # memory is full — stop

    def rescan(self) -> int:
        """Re-scan the root and register any newly-discovered models."""
        before = len(self.memory)
        self._register()
        return len(self.memory) - before

    # ---------- public API ----------
    def list_models(self) -> List[str]:
        return [e.name for _, e in self.memory.entries()]

    def describe(self, name: str) -> Dict:
        e = self.memory.get_by_name(name)
        return e.summary() if e is not None else {}

    def fingerprint(self, name: str) -> str:
        e = self.memory.get_by_name(name)
        return e.fingerprint if e is not None else ""

    def stats(self) -> Dict:
        return dict(
            K=self.K,
            capacity=self.memory.capacity,
            registered=len(self.memory),
            density=len(self.memory) / max(self.K, 1),
            density_theory=DENSITY,
            backend=self.backend_name,
            root=self.root,
            scan_ms=round(self.scan_ms, 2),
            context_window=len(self.context),
            context_capacity=self.K,
        )

    def ask(self, model: str, prompt: str,
            max_tokens: int = 64) -> str:
        """Blocking call — accumulate the full generation into a string."""
        pieces: List[str] = []
        for piece in self.stream(model, prompt, max_tokens=max_tokens):
            pieces.append(piece)
        return "".join(pieces)

    def stream(self, model: str, prompt: str,
               max_tokens: int = 64) -> Iterator[str]:
        """Streaming call — yields text pieces as they arrive."""
        entry = self.memory.get_by_name(model)
        if entry is None:
            raise KeyError(f"model not found: {model}")

        # store in context (bounded)
        for tok in self.tokenizer.tokenize(prompt):
            self.context.append(tok)

        if self.backend_name == "mock":
            for piece in self.backend.generate(prompt, max_tokens=max_tokens):
                yield piece + " "
        else:
            for piece in self.backend.generate(model, prompt,
                                               max_tokens=max_tokens):
                yield piece


# ============================================================
#  Demo
# ============================================================
def _make_fake_models(root: str) -> None:
    """Create a handful of fake model folders for the demo."""
    os.makedirs(root, exist_ok=True)
    specs = [
        ("llama-3.2-3b",  {"architectures": ["LlamaForCausalLM"],
                           "hidden_size": 3072, "num_hidden_layers": 28}),
        ("mistral-7b",    {"architectures": ["MistralForCausalLM"],
                           "hidden_size": 4096, "num_hidden_layers": 32}),
        ("phi-3-mini",    {"architectures": ["Phi3ForCausalLM"],
                           "hidden_size": 3072, "num_hidden_layers": 32}),
        ("qwen2-1.5b",    {"architectures": ["Qwen2ForCausalLM"],
                           "hidden_size": 1536, "num_hidden_layers": 28}),
    ]
    for name, cfg in specs:
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        # a dummy weights file so the fingerprint sees something
        with open(os.path.join(d, "weights.bin"), "wb") as f:
            f.write(b"\0" * 128)

    # one Modelfile model
    d = os.path.join(root, "tiny-custom")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "Modelfile"), "w") as f:
        f.write("FROM llama3.2\nPARAMETER temperature 0.7\n"
                "SYSTEM You are a helpful assistant.\n")


def demo():
    print("=" * 78)
    print("Local LLM interface  ·  O(K) memory  ·  directory-sourced models")
    print("=" * 78)

    root = os.path.expanduser("~/_llm_models_demo")
    _make_fake_models(root)

    iface = LocalLLMInterface(root=root, K=K_DEFAULT, backend="mock")

    print(f"  root                : {iface.root}")
    print(f"  K                   : {iface.K}")
    print(f"  capacity (μ slots)  : {iface.memory.capacity}")
    print(f"  registered models   : {len(iface.memory)}")
    print(f"  density             : {len(iface.memory) / iface.K:.4f}"
          f"  (target 6/π² = {DENSITY:.4f})")
    print(f"  backend             : {iface.backend_name}")
    print(f"  scan time           : {iface.scan_ms:.2f} ms")
    print()

    # ---------- list models ----------
    print("--- list_models ---")
    for name in iface.list_models():
        print(f"  {name}")
    print()

    # ---------- describe each ----------
    print("--- describe ---")
    print(f"  {'name':<18s}  {'kind':<12s}  {'bytes':>8s}  "
          f"{'files':>5s}  fingerprint")
    print("  " + "-" * 72)
    for name in iface.list_models():
        d = iface.describe(name)
        fp = iface.fingerprint(name)[:16] + "…"
        print(f"  {d['name']:<18s}  {d['kind']:<12s}  "
              f"{d['size_bytes']:>8d}  {d['n_files']:>5d}  {fp}")
    print()

    # ---------- ask each model (mock backend) ----------
    print("--- ask  (mock backend) ---")
    prompt = "What is the meaning of signal?"
    for name in iface.list_models()[:3]:
        t0 = time.perf_counter()
        reply = iface.ask(name, prompt, max_tokens=12)
        dt = (time.perf_counter() - t0) * 1e3
        print(f"  [{name}]  ({dt:.1f} ms)")
        print(f"    > {reply.strip()}")
    print()

    # ---------- stream one model ----------
    print("--- stream  (mock backend) ---")
    name = iface.list_models()[0]
    print(f"  model: {name}")
    print("  > ", end="", flush=True)
    for piece in iface.stream(name, "Once upon a time", max_tokens=16):
        print(piece, end="", flush=True)
    print()
    print()

    # ---------- stats ----------
    print("--- stats ---")
    for k, v in iface.stats().items():
        print(f"  {k:<20s}: {v}")
    print()

    # ---------- Ollama bridge example (not executed here) ----------
    print("--- Ollama bridge ---")
    print("  To drive real Ollama models with the same interface:")
    print('      iface = LocalLLMInterface(root="~/models", K=512,')
    print('                                 backend="ollama",')
    print('                                 ollama_host="http://localhost:11434")')
    print('      for piece in iface.stream("llama-3.2-3b", "Hello"):')
    print('          print(piece, end="", flush=True)')
    print()
    print("  The folder name must match the Ollama model name.")
    print()

    # clean up demo folder (optional)
    # import shutil; shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    demo()