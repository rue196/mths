#!/usr/bin/env python3
"""
https_mosb_vm_inference.py
===========================

HTTPS/HTML inference server.

    · virtual machine  =  512-bit layer (FileCodec512, use_512bit=True)
                          runs Python / JavaScript / Lua through a
                          PolyglotVM; every request and response is
                          512-bit verified *before* it reaches the
                          language executor.
    · hyperref link    =  unchanged wire (MOSB framing + FILE header
                          + UTF-8 UGC ALU packet + weight sig) served
                          as an octet-stream download at
                          GET /frame/{token}

Wire (identical to tcp_mosb_geo.py / cli_512.py):
    [ MOSB header        (48 B) ]
    [ FILE header        (55 B) ]
    [ UTF-8 UGC ALU packet      ]
    [ weight signature   (8 B)  ]

512-bit verification runs on decode:
    particle / ugc / alu / fiber       (64-bit)
    weight_match                       (SHA-256 slice)
    qp_consistent                      Q/P 512-bit
    zwalk_consistent                   Z-walk 512-bit

Usage
-----
    python https_mosb_vm_inference.py                 # HTTPS if openssl, else HTTP
    python https_mosb_vm_inference.py --no-tls        # force HTTP
    python https_mosb_vm_inference.py --port 8443     # custom port
    python https_mosb_vm_inference.py --K-ci 20       # stronger Z-walk
"""

from __future__ import annotations
import argparse, io, json, os, shutil, ssl, subprocess, sys, tempfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html import escape
from urllib.parse import parse_qs, urlparse

# ---- 512-bit + 64-bit codecs (wire identical) ----
from cli_512 import (
    FileCodec512, BITS, PREC_DIGITS,
)
from cli import (
    FileCodec, FileHeader, FILE_SIZE,
    MODE_RAW, MODE_TEXT, MODE_BINARY, MODE_NAMES,
)
from tcp_mosb_geo import MobiusTransport


# ============================================================
#  PolyglotVM  ·  the "virtual machine that runs other languages"
# ============================================================
class PolyglotVM:
    """
    Multi-language code executor.  Every submission is 512-bit
    verified by the surrounding codec *before* it reaches here.
    """

    MAX_CODE_BYTES    = 10_000
    MAX_OUTPUT_BYTES  = 64_000

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.available = self._detect()

    def _detect(self) -> dict:
        avail = {"python": True}
        for lang, cmd in [("javascript", ["node", "--version"]),
                          ("lua",        ["lua",  "-v"])]:
            try:
                subprocess.run(cmd, capture_output=True, timeout=2, check=True)
                avail[lang] = True
            except Exception:
                avail[lang] = False
        return avail

    def execute(self, lang: str, code: str) -> dict:
        if len(code.encode("utf-8", errors="replace")) > self.MAX_CODE_BYTES:
            return {"ok": False, "output": "",
                    "error": f"code exceeds {self.MAX_CODE_BYTES} bytes",
                    "elapsed_ms": 0.0}
        if lang not in self.available or not self.available[lang]:
            return {"ok": False, "output": "",
                    "error": f"language '{lang}' not available on this server",
                    "elapsed_ms": 0.0}
        try:
            if lang == "python":
                return self._run([sys.executable, "-c", code], stdin=None)
            if lang == "javascript":
                return self._run(["node", "-e", code], stdin=None)
            if lang == "lua":
                return self._run(["lua"], stdin=code)
            return {"ok": False, "output": "",
                    "error": f"unknown language '{lang}'", "elapsed_ms": 0.0}
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": "",
                    "error": f"timeout after {self.timeout}s",
                    "elapsed_ms": self.timeout * 1e3}
        except Exception as e:
            return {"ok": False, "output": "",
                    "error": f"{type(e).__name__}: {e}", "elapsed_ms": 0.0}

    def _run(self, cmd, stdin):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, input=stdin,
                              capture_output=True, text=True,
                              timeout=self.timeout)
        return {
            "ok":         proc.returncode == 0,
            "output":     proc.stdout[:self.MAX_OUTPUT_BYTES],
            "error":      proc.stderr[:self.MAX_OUTPUT_BYTES],
            "elapsed_ms": (time.perf_counter() - t0) * 1e3,
        }


# ============================================================
#  MOSB VM codec  ·  unchanged wire
# ============================================================
class MOSBVMCodec:
    """
    Frame layout (byte-identical to cli_512.py / tcp_mosb_geo.py):

        [ MOSB header        (48 B) ]
        [ FILE header        (55 B) ]
        [ UTF-8 UGC ALU packet      ]
        [ weight signature   (8 B)  ]

    encode():  wrap payload, return dict with wire + diagnostics.
    decode():  unpack MOSB, unpack FILE header, run 512-bit verify.
    """

    def __init__(self, use_512bit: bool = True, K_ci: int = 12):
        self.use_512bit = use_512bit
        self.K_ci       = K_ci
        if use_512bit:
            self.codec = FileCodec512(K_table=128, K_alu=64, K_ci=K_ci)
        else:
            self.codec = FileCodec(K_table=128, K_alu=64, K_ci=K_ci)
        self.transport = MobiusTransport(K=512)

    # ---------- encode ----------
    def encode(self, payload: bytes, mode: int = MODE_TEXT) -> dict:
        packet = self.codec.encode(payload)
        header = FileHeader(
            mode=mode, reserved_a=0, reserved_b=0,
            K_ci=self.K_ci,
            threshold=self.codec.threshold,
            timestamp=time.time(),
            file_size=float(len(payload)),
            nonce=os.urandom(12),
            env_size=len(packet),
            msg_len=len(payload),
        )
        mosb = self.transport.frame(header.pack() + packet)
        return {
            "frame":  mosb.bytes_,
            "packet": packet,
            "header": header,
            "mosb_S": mosb.S,
            "mosb_H": mosb.H,
            "mosb_m": mosb.m,
            "size":   mosb.size,
        }

    # ---------- decode + verify ----------
    def decode(self, frame: bytes) -> tuple:
        try:
            inner, mosb = self.transport.deframe(frame)
        except Exception as e:
            return None, None, {"verified": False,
                                "error": f"deframe: {e}"}
        if len(inner) < FILE_SIZE:
            return None, None, {"verified": False,
                                "error": "short payload"}
        try:
            header = FileHeader.unpack(inner[:FILE_SIZE])
        except ValueError as e:
            return None, None, {"verified": False,
                                "error": f"FILE header: {e}"}
        packet = inner[FILE_SIZE:FILE_SIZE + header.env_size]
        if len(packet) != header.env_size:
            return None, None, {"verified": False,
                                "error": "packet size mismatch"}

        recovered, diag = self.codec.decode(packet)
        diag = dict(diag)
        diag["mosb_S"]     = mosb.S
        diag["mosb_H"]     = mosb.H
        diag["mosb_m"]     = mosb.m
        diag["frame_size"] = mosb.size
        diag["packet_size"] = len(packet)
        diag["bits"]       = BITS if self.use_512bit else 64
        return header, recovered, diag


# ============================================================
#  HTML rendering
# ============================================================
_CSS = """
body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;
     padding:0 1em;background:#f7f7f8;color:#222}
h1{color:#2c3e50;border-bottom:2px solid #2c3e50;padding-bottom:.3em}
h2{color:#34495e;margin-top:1.5em}
textarea{width:100%;height:200px;font-family:'SF Mono',Menlo,monospace;
         font-size:13px;padding:.6em;border:1px solid #ccc;border-radius:4px;
         box-sizing:border-box}
select,button{padding:8px 14px;font-size:14px;border-radius:4px;border:1px solid #ccc}
button{background:#2c3e50;color:#fff;border:none;cursor:pointer}
button:hover{background:#34495e}
pre{background:#1e1e1e;color:#d4d4d4;padding:1em;border-radius:4px;
    overflow-x:auto;font-size:13px}
.badge{display:inline-block;padding:3px 10px;border-radius:3px;
       font-size:12px;margin-left:6px;font-weight:600}
.ok{background:#27ae60;color:#fff}
.fail{background:#e74c3c;color:#fff}
.info{background:#3498db;color:#fff}
.card{background:#fff;padding:1em;border-radius:6px;
      box-shadow:0 1px 3px rgba(0,0,0,.08);margin:1em 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #eee}
th{background:#f0f0f2;font-weight:600}
a{color:#2980b9}
code{background:#eef;padding:1px 5px;border-radius:3px;font-size:12px}
"""

def render_form(info: dict, result_html: str = "",
                frame_token: str = "") -> str:
    langs = info["available_languages"]
    lang_opts = "".join(f'<option value="{escape(l)}">{escape(l)}</option>'
                        for l in langs)
    bits    = info["bits"]
    bits_badge = (f'<span class="badge ok">{bits}-bit VM</span>'
                  if info["use_512bit"]
                  else '<span class="badge info">64-bit</span>')
    wire_badge = '<span class="badge ok">wire unchanged</span>'
    hyperref = ""
    if frame_token:
        hyperref = (
            f'<div class="card">'
            f'  <b>Hyperref link — unchanged MOSB wire</b><br>'
            f'  <a href="/frame/{escape(frame_token)}">'
            f'    ⬇ Download raw MOSB frame ({escape(frame_token)}.mosb)'
            f'  </a>'
            f'  <p style="font-size:12px;color:#666;margin:.5em 0 0 0">'
            f'    MOSB framing + FILE header + UTF-8 UGC ALU packet '
            f'+ weight signature. Byte-identical to cli_512.py.'
            f'  </p>'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MOSB HTTPS VM Inference</title>
<style>{_CSS}</style>
</head>
<body>
<h1>MOSB HTTPS VM Inference {bits_badge}{wire_badge}</h1>
<p style="color:#555">
  Virtual machine = 512-bit verification layer
  (<code>use_512bit={info['use_512bit']}</code>, default on).<br>
  Hyperref link = unchanged MOSB wire &mdash; same framing, same DATA payload.
</p>

<div class="card">
  <b>Server</b><br>
  <table>
    <tr><td>verification</td><td>{bits}-bit
        ({'Q/P + Z-walk at 512-bit' if info['use_512bit'] else '64-bit only'})</td></tr>
    <tr><td>K_ci</td><td>{info['K_ci']}</td></tr>
    <tr><td>languages</td><td>{', '.join(escape(l) for l in langs)}</td></tr>
    <tr><td>wire</td><td><code>MOSB &middot; FILE &middot; UTF-8 UGC ALU &middot; weight sig</code></td></tr>
  </table>
</div>

<h2>Submit code</h2>
<form method="POST" action="/run">
  <p>
    <select name="lang">
      {lang_opts}
    </select>
    <button type="submit">Run through 512-bit VM &rarr;</button>
  </p>
  <textarea name="code" placeholder="print('hello from the 512-bit VM')">print("hello from the 512-bit VM")</textarea>
</form>

{hyperref}
{result_html}

<p style="color:#888;font-size:12px;margin-top:3em">
  <a href="/info">/info</a> &middot;
  <a href="/frame/{{token}}">/frame/{{token}}</a> &middot;
  <code>POST /run</code> (form) &middot; <code>GET /</code>
</p>
</body>
</html>"""


def render_result(lang: str, code: str, enc: dict,
                  diag: dict, exec_result: dict,
                  frame_token: str) -> str:
    verified = bool(diag.get("verified"))
    status = ('<span class="badge ok">512-bit VERIFIED</span>'
              if verified
              else '<span class="badge fail">VERIFICATION FAILED</span>')
    bits = diag.get("bits", 64)

    # --- Q/P block ---
    qp = diag.get("qp_diag") or {}
    n_baby  = qp.get("n_baby", 0)
    n_giant = qp.get("n_giant", 0)
    agree   = qp.get("mode_agree", 0)
    dis     = qp.get("mode_disagree", 0)

    # --- Z-walk block ---
    zw = diag.get("zwalk_diag") or {}
    zw_steps = zw.get("n_steps", 0)
    zw_exp   = zw.get("expected", 0)
    zw_rms   = zw.get("Zf_re_rms", 0.0)
    zw_rms_im = zw.get("Zf_im_rms", 0.0)

    # --- execution block ---
    ok_exec = exec_result.get("ok")
    out = exec_result.get("output", "")
    err = exec_result.get("error", "")
    elapsed = exec_result.get("elapsed_ms", 0.0)

    exec_status = ('<span class="badge ok">EXECUTED</span>'
                   if ok_exec else
                   '<span class="badge fail">FAILED</span>')
    out_block = ""
    if out:
        out_block += f"<h3>stdout</h3><pre>{escape(out)}</pre>"
    if err:
        out_block += f"<h3>stderr</h3><pre>{escape(err)}</pre>"
    if not out and not err:
        out_block = '<pre style="color:#888">(no output)</pre>'

    return f"""
<div class="card">
  <h2>Result  {status}</h2>
  <table>
    <tr><td>language</td><td><code>{escape(lang)}</code></td></tr>
    <tr><td>verification bits</td><td>{bits}-bit</td></tr>
    <tr><td>wire unchanged</td>
        <td><span class="badge ok">MOSB framing intact</span></td></tr>
    <tr><td>frame size</td><td>{diag.get('frame_size', 0)} B</td></tr>
    <tr><td>packet size</td><td>{diag.get('packet_size', 0)} B</td></tr>
    <tr><td>MOSB S / H / m</td>
        <td>{diag.get('mosb_S', 0):+.4f} &middot;
            {diag.get('mosb_H', 0):.4f} &middot;
            {diag.get('mosb_m', 0):.4e}</td></tr>
  </table>
</div>

<div class="card">
  <h2>512-bit Q/P</h2>
  <table>
    <tr><td>qp_consistent</td><td>{diag.get('qp_consistent')}</td></tr>
    <tr><td>samples</td><td>{n_baby + n_giant}</td></tr>
    <tr><td>baby</td><td>{n_baby}</td></tr>
    <tr><td>giant</td><td>{n_giant}</td></tr>
    <tr><td>mode agree (64 vs 512)</td>
        <td>{agree}  /  disagree {dis}</td></tr>
  </table>
</div>

<div class="card">
  <h2>512-bit Z-walk</h2>
  <table>
    <tr><td>zwalk_consistent</td>
        <td>{diag.get('zwalk_consistent')}</td></tr>
    <tr><td>n_steps / expected</td>
        <td>{zw_steps} / {zw_exp}</td></tr>
    <tr><td>Zf_re_rms</td><td>{zw_rms:+.6e}</td></tr>
    <tr><td>Zf_im_rms</td><td>{zw_rms_im:+.6e}</td></tr>
  </table>
</div>

<div class="card">
  <h2>Execution {exec_status}</h2>
  <table>
    <tr><td>elapsed</td><td>{elapsed:.2f} ms</td></tr>
  </table>
  {out_block}
</div>

<div class="card">
  <h2>Source</h2>
  <pre>{escape(code)}</pre>
</div>

<div class="card">
  <b>Hyperref link — unchanged MOSB wire</b><br>
  <a href="/frame/{escape(frame_token)}">
    ⬇ Download raw MOSB frame ({escape(frame_token)}.mosb)
  </a>
</div>

<p><a href="/">← submit another</a></p>
"""


# ============================================================
#  HTTP handler
# ============================================================
class InferenceHandler(BaseHTTPRequestHandler):
    server_version = "MOSBVM/1.0"
    app = None    # bound by HTTPSMOSBVMInference

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._serve_html(self.app.render_form())
        elif path == "/info":
            self._serve_json(self.app.info())
        elif path.startswith("/frame/"):
            self._serve_frame(path[len("/frame/"):])
        else:
            self.send_error(404, "not found")

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        params = parse_qs(raw)
        if path == "/run":
            lang = params.get("lang", ["python"])[0]
            code = params.get("code", [""])[0]
            html = self.app.run_and_render(lang, code)
            self._serve_html(html)
        else:
            self.send_error(404, "not found")

    def _serve_html(self, html: str):
        b = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _serve_json(self, obj):
        b = json.dumps(obj, default=str, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _serve_frame(self, token: str):
        entry = self.app.frame_cache.get(token)
        if not entry:
            self.send_error(404, "frame not found or expired")
            return
        b = entry["frame"]
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{token}.mosb"')
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, fmt, *args):
        if not self.app or not self.app.verbose:
            return
        sys.stderr.write(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{self.address_string()} {fmt % args}\n")


# ============================================================
#  HTTPS server
# ============================================================
class HTTPSMOSBVMInference:
    def __init__(self,
                 host: str = "127.0.0.1",
                 port: int = 8443,
                 use_tls: bool = True,
                 use_512bit: bool = True,
                 K_ci: int = 12,
                 timeout: float = 5.0,
                 verbose: bool = True):
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.use_512bit = use_512bit
        self.verbose = verbose

        self.vm     = PolyglotVM(timeout=timeout)
        self.codec  = MOSBVMCodec(use_512bit=use_512bit, K_ci=K_ci)

        self.frame_cache: dict = {}
        self.frame_lock = threading.Lock()

        self._httpd = None

    # ---------- info ----------
    def info(self) -> dict:
        return {
            "server":  "MOSB HTTPS VM Inference",
            "bits":    BITS if self.use_512bit else 64,
            "use_512bit": self.use_512bit,
            "K_ci":    self.codec.K_ci,
            "available_languages": [
                l for l, ok in self.vm.available.items() if ok
            ],
            "wire": (
                "MOSB header (48 B) + FILE header (55 B) + "
                "UTF-8 UGC ALU packet + weight signature (8 B)"
            ),
            "wire_changed": False,
        }

    # ---------- form ----------
    def render_form(self, result_html: str = "",
                    frame_token: str = "") -> str:
        return render_form(self.info(), result_html, frame_token)

    # ---------- run ----------
    def run_and_render(self, lang: str, code: str) -> str:
        # 1. wrap in wire payload (unchanged DATA format)
        payload = f"{lang}\n{code}".encode("utf-8")
        enc = self.codec.encode(payload, mode=MODE_TEXT)

        # 2. decode + 512-bit verify (the VM's compile/parse step)
        header, recovered, diag = self.codec.decode(enc["frame"])

        # 3. only execute if the VM verified
        if diag.get("verified") and recovered is not None:
            exec_result = self.vm.execute(lang, code)
        else:
            exec_result = {
                "ok":     False,
                "output": "",
                "error":  "verification failed — code not executed",
                "elapsed_ms": 0.0,
            }

        # 4. cache the raw frame for the hyperref link
        token = uuid.uuid4().hex[:16]
        with self.frame_lock:
            self.frame_cache[token] = {
                "frame": enc["frame"],
                "diag":  diag,
                "ts":    time.time(),
            }
            if len(self.frame_cache) > 200:
                oldest = min(
                    self.frame_cache,
                    key=lambda k: self.frame_cache[k]["ts"])
                del self.frame_cache[oldest]

        if self.verbose:
            status = "VERIFIED" if diag.get("verified") else "FAILED"
            print(f"[run] lang={lang}  wire={len(enc['frame'])} B  "
                  f"vm={status}  exec={'ok' if exec_result['ok'] else 'fail'}")

        return render_result(
            lang=lang, code=code, enc=enc,
            diag=diag, exec_result=exec_result,
            frame_token=token)

    # ---------- TLS cert ----------
    def _ensure_cert(self) -> tuple:
        base = os.path.join(tempfile.gettempdir(), "mosb_vm_tls")
        os.makedirs(base, exist_ok=True)
        cert = os.path.join(base, "cert.pem")
        key  = os.path.join(base, "key.pem")
        if os.path.exists(cert) and os.path.exists(key):
            return cert, key
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048",
             "-keyout", key, "-out", cert,
             "-days", "365", "-nodes",
             "-subj", "/CN=localhost"],
            check=True, capture_output=True, timeout=10)
        return cert, key

    def _ssl_ctx(self):
        cert, key = self._ensure_cert()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        return ctx

    # ---------- serve ----------
    def serve_forever(self):
        # bind app to handler
        InferenceHandler.app = self

        if self.use_tls:
            self._httpd = ThreadingHTTPServer(
                (self.host, self.port), InferenceHandler)
            self._httpd.socket = self._ssl_ctx().wrap_socket(
                self._httpd.socket, server_side=True)
            scheme = "https"
        else:
            self._httpd = ThreadingHTTPServer(
                (self.host, self.port), InferenceHandler)
            scheme = "http"

        bits = BITS if self.use_512bit else 64
        langs = [l for l, ok in self.vm.available.items() if ok]
        print("=" * 66)
        print("MOSB HTTPS VM Inference")
        print("=" * 66)
        print(f"  URL         : {scheme}://{self.host}:{self.port}/")
        print(f"  bits        : {bits}")
        print(f"  use_512bit  : {self.use_512bit}")
        print(f"  K_ci        : {self.codec.K_ci}")
        print(f"  languages   : {', '.join(langs)}")
        print(f"  wire        : MOSB framing (unchanged)")
        print(f"                + FILE header + UTF-8 UGC ALU packet")
        print(f"                + weight signature (8 B)")
        if scheme == "https":
            print(f"  cert        : self-signed (accept in browser)")
        print()
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            try:
                self._httpd.server_close()
            except Exception:
                pass


# ============================================================
#  Main
# ============================================================
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="https_mosb_vm_inference",
        description="HTTPS/HTML inference server with 512-bit VM "
                    "and unchanged MOSB wire")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--tls", action="store_true",
                   help="use HTTPS (self-signed cert)")
    p.add_argument("--no-tls", action="store_true",
                   help="force plain HTTP")
    p.add_argument("--no-512", action="store_true",
                   help="disable 512-bit verification (64-bit only)")
    p.add_argument("--K-ci", type=int, default=12,
                   help="K_ci for spectral sum (12 = fast, 20 = strong)")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="execution timeout per request (s)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    use_tls = (args.tls
               or (not args.no_tls and shutil.which("openssl") is not None))

    app = HTTPSMOSBVMInference(
        host=args.host,
        port=args.port,
        use_tls=use_tls,
        use_512bit=not args.no_512,
        K_ci=args.K_ci,
        timeout=args.timeout,
        verbose=not args.quiet,
    )
    app.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())