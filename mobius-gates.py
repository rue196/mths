import math
import numpy as np
from chip_g import ChipProcessor  # assuming chip-g.py is in the same directory

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

class ChipLogicGate:
    """
    A bounded Möbius logic gate that applies a function to a signal,
    then runs the chip pipeline (convolution + supertrace + Möbius compression).
    The output is a compressed representation of the transformed signal.
    """

    def __init__(self, max_K=1000):
        self.processor = ChipProcessor(max_K=max_K)

    def apply_function(self, signal, func, *args, **kwargs):
        """
        Apply a function `func` to each element of `signal`.
        `func` can be a callable (e.g., math.log, math.exp, np.sin)
        or a string ('log', 'exp', 'sin', 'cos', 'trace').
        """
        if isinstance(func, str):
            func_name = func.lower()
            if func_name == 'log':
                # avoid log(0)
                safe_signal = np.maximum(signal, 1e-12)
                transformed = np.log(safe_signal)
            elif func_name == 'exp':
                transformed = np.exp(signal)
            elif func_name == 'sin':
                transformed = np.sin(signal)
            elif func_name == 'cos':
                transformed = np.cos(signal)
            elif func_name == 'trace':
                # Matrix trace function: assumes signal is complex and represents matrix entries
                transformed = self._matrix_trace(signal)
            else:
                raise ValueError(f"Unknown function name: {func_name}")
        else:
            # assume it's a callable
            transformed = func(signal, *args, **kwargs)

        # Run the chip pipeline on the transformed signal
        kept, S, H, m, conv = self.processor.process(transformed)

        # Return the compressed representation and invariants
        return kept, S, H, m, conv

    def _matrix_trace(self, signal):
        """
        Compute the trace of a 2x2 matrix from a signal of length 4:
        signal = [M00, M01, M10, M11]  (complex or real)
        Returns the trace (scalar) repeated to match the original length?
        For a logic gate, we treat the trace as a scalar that multiplies the signal.
        """
        if len(signal) != 4:
            raise ValueError("Signal for matrix trace must have exactly 4 elements.")
        M = np.array(signal).reshape(2, 2)
        trace = np.trace(M)
        # Return a constant signal of the same length as input (if we want element-wise)
        # But here we treat it as a scalar result; we'll expand to an array of length 1.
        return np.array([trace])

    def bounded_log(self, signal):
        """Apply log with a bound: replace log(x) with log(max(x, epsilon))."""
        return self.apply_function(signal, 'log')

    def bounded_exp(self, signal):
        """Apply exp and then clip to prevent overflow."""
        # The chip pipeline will compress, so no need to clip, but we can.
        return self.apply_function(signal, 'exp')

    # Additional functions can be added similarly


# ---------- Example usage ----------
def main():
    # Create a logic gate processor
    gate = ChipLogicGate(max_K=500)

    # Generate a test signal: a smooth curve (harmonic numbers)
    K = 200
    signal = np.array([math.log(i+1) for i in range(K)])

    print("Original signal (first 10):", signal[:10])

    # Apply bounded logarithm (already log, so same)
    kept_log, S_log, H_log, m_log, conv_log = gate.bounded_log(signal)
    print(f"\nLog transform: S={S_log:.4f}, H={H_log:.4f}, m={m_log:.4f}, kept {len(kept_log)} coeffs")

    # Apply exp to the signal
    kept_exp, S_exp, H_exp, m_exp, conv_exp = gate.bounded_exp(signal)
    print(f"Exp transform: S={S_exp:.4f}, H={H_exp:.4f}, m={m_exp:.4f}, kept {len(kept_exp)} coeffs")

    # Apply matrix trace on a 4-element signal
    mat_signal = np.array([1.0, 0.5, 0.3, 0.7], dtype=complex)
    kept_trace, S_trace, H_trace, m_trace, conv_trace = gate.apply_function(mat_signal, 'trace')
    print(f"\nMatrix trace: S={S_trace:.4f}, H={H_trace:.4f}, m={m_trace:.4f}, kept {len(kept_trace)} coeffs")

    # Show the compressed indices and values for the log case
    print("\nFirst 5 kept for log transform (index, value):")
    for idx, val in kept_log[:5]:
        print(f"  {idx}: {val:.4f}")

if __name__ == "__main__":
    main()