import tkinter as tk
from tkinter import ttk, messagebox
import math

# ------------------------------------------------------------
# 1. Spectral functions
# ------------------------------------------------------------
def zeta(t, c, alpha):
    """Spectral sum ζ(t) = Σ C_i * cos(i*t/α) for symmetric real C_i."""
    K = (len(c) - 1) // 2
    total = c[K]
    for i in range(1, K + 1):
        theta = t * i / alpha
        total += 2.0 * c[K + i] * math.cos(theta)
    return total

def dzeta_dt(t, c, alpha):
    """Derivative dζ/dt = -2 Σ (i/α) C_i sin(i*t/α)."""
    K = (len(c) - 1) // 2
    deriv = 0.0
    for i in range(1, K + 1):
        theta = t * i / alpha
        deriv += -2.0 * c[K + i] * (i / alpha) * math.sin(theta)
    return deriv

def scarcity_multiplier(p, n):
    """Binomial scarcity amplifier (1-p)^(-n)."""
    if p >= 1.0:
        return 1e6
    return (1.0 - p) ** (-n)

def clamp_p(p):
    """Clamp failure probability to [0.01, 0.99]."""
    return min(max(p, 0.01), 0.99)

# ------------------------------------------------------------
# 2. Main application – modular spreadsheet
# ------------------------------------------------------------
class ModularSpreadsheet:
    def __init__(self, root):
        self.root = root
        self.root.title("Modular Spectral Spreadsheet (M = [x⁻¹, y⁻¹, Xⁱ, Yⁱ])")

        # ----- global parameters (defaults) -----
        self.K = 10
        self.alpha = 0.3628
        self.t = 1.5
        self.p0_x = 0.20
        self.p0_y = 0.30
        self.n_x = 2
        self.n_y = 3

        # coefficients will be stored as list of floats
        self.c_vals = [1.0] * (2 * self.K + 1)

        # storage for entry widgets (StringVars)
        self.c_vars = []

        # storage for display labels (will be rebuilt when K changes)
        self.display_labels = []

        # ----- build UI -----
        self.build_controls()
        self.build_table_frame()

        # initial render
        self.rebuild_table()

    # ------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------
    def build_controls(self):
        ctrl = ttk.LabelFrame(self.root, text="Controls", padding=8)
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        # Row 0: K, t, alpha
        ttk.Label(ctrl, text="K:").grid(row=0, column=0, padx=5)
        self.k_spin = ttk.Spinbox(ctrl, from_=1, to=50, width=5,
                                  command=self.on_k_change)
        self.k_spin.set(str(self.K))
        self.k_spin.grid(row=0, column=1, padx=5)

        ttk.Label(ctrl, text="t:").grid(row=0, column=2, padx=5)
        self.t_entry = ttk.Entry(ctrl, width=8)
        self.t_entry.insert(0, str(self.t))
        self.t_entry.bind("<KeyRelease>", self.on_param_change)
        self.t_entry.grid(row=0, column=3, padx=5)

        ttk.Label(ctrl, text="α:").grid(row=0, column=4, padx=5)
        self.alpha_entry = ttk.Entry(ctrl, width=8)
        self.alpha_entry.insert(0, str(self.alpha))
        self.alpha_entry.bind("<KeyRelease>", self.on_param_change)
        self.alpha_entry.grid(row=0, column=5, padx=5)

        # Row 1: binomial parameters
        ttk.Label(ctrl, text="p0_x:").grid(row=1, column=0, padx=5)
        self.px_entry = ttk.Entry(ctrl, width=6)
        self.px_entry.insert(0, str(self.p0_x))
        self.px_entry.bind("<KeyRelease>", self.on_param_change)
        self.px_entry.grid(row=1, column=1, padx=5)

        ttk.Label(ctrl, text="n_x:").grid(row=1, column=2, padx=5)
        self.nx_entry = ttk.Entry(ctrl, width=4)
        self.nx_entry.insert(0, str(self.n_x))
        self.nx_entry.bind("<KeyRelease>", self.on_param_change)
        self.nx_entry.grid(row=1, column=3, padx=5)

        ttk.Label(ctrl, text="p0_y:").grid(row=1, column=4, padx=5)
        self.py_entry = ttk.Entry(ctrl, width=6)
        self.py_entry.insert(0, str(self.p0_y))
        self.py_entry.bind("<KeyRelease>", self.on_param_change)
        self.py_entry.grid(row=1, column=5, padx=5)

        ttk.Label(ctrl, text="n_y:").grid(row=1, column=6, padx=5)
        self.ny_entry = ttk.Entry(ctrl, width=4)
        self.ny_entry.insert(0, str(self.n_y))
        self.ny_entry.bind("<KeyRelease>", self.on_param_change)
        self.ny_entry.grid(row=1, column=7, padx=5)

        # Row 2: summary labels (ζ, dζ/dt, x⁻¹, y⁻¹)
        ttk.Label(ctrl, text="ζ(t) =").grid(row=2, column=0, padx=5, pady=5)
        self.zeta_label = ttk.Label(ctrl, text="0.000", foreground="purple")
        self.zeta_label.grid(row=2, column=1, padx=5)

        ttk.Label(ctrl, text="dζ/dt =").grid(row=2, column=2, padx=5)
        self.dzeta_label = ttk.Label(ctrl, text="0.000", foreground="darkgreen")
        self.dzeta_label.grid(row=2, column=3, padx=5)

        ttk.Label(ctrl, text="x⁻¹ =").grid(row=2, column=4, padx=5)
        self.x_inv_label = ttk.Label(ctrl, text="0.000", foreground="blue")
        self.x_inv_label.grid(row=2, column=5, padx=5)

        ttk.Label(ctrl, text="y⁻¹ =").grid(row=2, column=6, padx=5)
        self.y_inv_label = ttk.Label(ctrl, text="0.000", foreground="red")
        self.y_inv_label.grid(row=2, column=7, padx=5)

        # Reset button
        self.reset_btn = ttk.Button(ctrl, text="Reset Cᵢ = 1", command=self.reset_coeffs)
        self.reset_btn.grid(row=2, column=8, padx=20)

    def build_table_frame(self):
        # main table frame with scrollbar
        table_frame = ttk.LabelFrame(self.root, text="Spectral Coefficients Table", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # canvas + scrollbar
        canvas = tk.Canvas(table_frame)
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # store canvas for later resizing if needed
        self.table_canvas = canvas

        # headers (they will be re‑created when K changes)
        self.headers = []

    # ------------------------------------------------------------
    # Table rebuild (called when K or any parameter changes)
    # ------------------------------------------------------------
    def rebuild_table(self):
        # clear existing table widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # read current K from spinbox
        try:
            new_k = int(self.k_spin.get())
            if new_k < 1:
                new_k = 1
            self.K = new_k
        except ValueError:
            self.K = 10
            self.k_spin.set(str(self.K))

        # resize coefficient array
        self.c_vals = [1.0] * (2 * self.K + 1)
        self.c_vars = [None] * (2 * self.K + 1)
        self.display_labels = []

        # headers
        headers = ["i", "Cᵢ (input)", "Xⁱ = Cᵢ·cos", "Yⁱ = Cᵢ·sin", "x⁻¹", "y⁻¹"]
        for col, text in enumerate(headers):
            lbl = ttk.Label(self.scrollable_frame, text=text, font=('Arial', 10, 'bold'))
            lbl.grid(row=0, column=col, padx=5, pady=5, sticky='w')

        # rows
        for idx in range(2 * self.K + 1):
            i = idx - self.K
            row = idx + 1

            # address
            ttk.Label(self.scrollable_frame, text=str(i)).grid(row=row, column=0, padx=5, sticky='w')

            # input cell for C_i
            var = tk.StringVar()
            var.set(f"{self.c_vals[idx]:.6f}")
            entry = ttk.Entry(self.scrollable_frame, textvariable=var, width=12)
            entry.grid(row=row, column=1, padx=5, pady=2)
            entry.bind("<KeyRelease>", self.on_cell_change)
            self.c_vars[idx] = var

            # Xⁱ, Yⁱ labels
            x_trans = ttk.Label(self.scrollable_frame, text="0.000", width=14)
            x_trans.grid(row=row, column=2, padx=5)
            y_trans = ttk.Label(self.scrollable_frame, text="0.000", width=14)
            y_trans.grid(row=row, column=3, padx=5)

            # algebraic parts (global)
            x_alg = ttk.Label(self.scrollable_frame, text="0.000", width=14, foreground="blue")
            x_alg.grid(row=row, column=4, padx=5)
            y_alg = ttk.Label(self.scrollable_frame, text="0.000", width=14, foreground="red")
            y_alg.grid(row=row, column=5, padx=5)

            self.display_labels.append({
                'x_trans': x_trans,
                'y_trans': y_trans,
                'x_alg': x_alg,
                'y_alg': y_alg
            })

        # update all values
        self.update_all()

    # ------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------
    def on_k_change(self):
        """K spinbox changed – rebuild table."""
        self.rebuild_table()

    def on_param_change(self, event=None):
        """User changed t, α, or binomial parameters."""
        try:
            self.t = float(self.t_entry.get())
            self.alpha = float(self.alpha_entry.get())
            self.p0_x = float(self.px_entry.get())
            self.n_x = float(self.nx_entry.get())
            self.p0_y = float(self.py_entry.get())
            self.n_y = float(self.ny_entry.get())
        except ValueError:
            return
        self.update_all()

    def on_cell_change(self, event=None):
        """User edited a C_i cell."""
        self.read_c_values()
        self.update_all()

    def read_c_values(self):
        """Read all C_i from entry widgets and store in self.c_vals."""
        for idx, var in enumerate(self.c_vars):
            if var is not None:
                try:
                    val = float(var.get())
                    self.c_vals[idx] = val
                except ValueError:
                    # keep previous value (do nothing)
                    pass

    def reset_coeffs(self):
        """Reset all C_i to 1.0."""
        for idx in range(len(self.c_vals)):
            self.c_vals[idx] = 1.0
            if self.c_vars[idx] is not None:
                self.c_vars[idx].set("1.000000")
        self.update_all()

    # ------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------
    def update_all(self):
        # ensure c_vals is up to date
        self.read_c_values()

        # compute spectral sum and derivative
        z = zeta(self.t, self.c_vals, self.alpha)
        dz = dzeta_dt(self.t, self.c_vals, self.alpha)

        # update summary labels
        self.zeta_label.config(text=f"{z:.6f}")
        self.dzeta_label.config(text=f"{dz:.6f}")

        # map derivative to binomial failure probability shift
        def derivative_to_p(dz, base_p, scale=0.02):
            # bounded shift using tanh
            delta = math.tanh(dz * scale)  # in (-1,1)
            new_p = base_p + delta * 0.2
            return clamp_p(new_p)

        p_x = derivative_to_p(dz, self.p0_x, 0.02)
        p_y = derivative_to_p(dz, self.p0_y, 0.03)

        x_inv = scarcity_multiplier(p_x, self.n_x)
        y_inv = scarcity_multiplier(p_y, self.n_y)

        # update global algebraic labels
        self.x_inv_label.config(text=f"{x_inv:.6f}")
        self.y_inv_label.config(text=f"{y_inv:.6f}")

        # update each row's transcendental and algebraic parts
        for idx in range(len(self.c_vals)):
            i = idx - self.K
            c_i = self.c_vals[idx]

            theta = self.t * i / self.alpha
            x_trans = c_i * math.cos(theta)
            y_trans = c_i * math.sin(theta)

            if idx < len(self.display_labels):
                lbls = self.display_labels[idx]
                lbls['x_trans'].config(text=f"{x_trans:.6f}")
                lbls['y_trans'].config(text=f"{y_trans:.6f}")
                lbls['x_alg'].config(text=f"{x_inv:.6f}")
                lbls['y_alg'].config(text=f"{y_inv:.6f}")

# ------------------------------------------------------------
# 5. Launch the app
# ------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x600")
    app = ModularSpreadsheet(root)
    root.mainloop()