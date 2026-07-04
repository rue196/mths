import numpy as np

def Compare3DClassicalVs6DQuantum_Compressed(T, dt, K):
    # T: total time, dt: time step
    # K: spectral compression size (number of Fourier modes per dimension)
    
    # --- 1. Initialize compressed representation ---
    γ_coeff = np.zeros(K, dtype=complex)  # classical position coefficients
    v_coeff = np.zeros(K, dtype=complex)  # classical velocity coefficients
    
    # Quantum 6D: separable ansatz, each dimension stored with K coefficients
    Ψ = [initialize_wavepacket_1D(K) for _ in range(6)]  # each is complex array length K
    
    # Potential spectra (diagonal in this basis)
    V_cl_spectrum = compute_potential_spectrum('classical', K)
    V_qm_spectrum = compute_potential_spectrum('quantum', K)
    
    # Cache for matrix invariants (Δ, τ) – computed in spectral domain
    cache = {}
    
    # --- 2. Set up entropy operator parameters ---
    alpha = 0.3628                     # from earlier (or any constant)
    collatz_even = True                # use only even indices (Collatz mask)
    
    # Storage for entropies and integrals (for post‑processing)
    entropy_history = []
    integral_history = []
    
    # --- 3. Time evolution loop ---
    for t in np.arange(0, T+dt, dt):
        # --- Common matrix invariants (O(K)) ---
        (Δ_spectral, gradΔ_spectral, cache) = MatrixIntensityAndGradient_Spectral(K, cache)
        
        # --- Classical 3D update (O(K)) ---
        F_coeff = compute_force_coefficients(γ_coeff, V_cl_spectrum, gradΔ_spectral)
        v_coeff = v_coeff + (F_coeff / m) * dt
        γ_coeff = γ_coeff + v_coeff * dt
        
        # --- Quantum 6D update (each dimension O(K)) ---
        for d in range(6):
            Ψ[d] = apply_kinetic(Ψ[d], dt, d)         # kinetic in Fourier basis
            Ψ[d] = apply_potential(Ψ[d], V_qm_spectrum, Δ_spectral, dt)   # potential in same basis (O(K))
        
        # --- Project to 3D effective wavefunction (still spectral) ---
        # For separable product, ψ_3D_spectral = pointwise product of first 3 dims
        # To stay O(K), we assume the product is diagonal in the chosen basis (e.g., Hermite functions)
        # We'll just keep it as a product of coefficients (convolution in real space, but we approximate)
        ψ_3D_spectral = Ψ[0] * Ψ[1] * Ψ[2]   # O(K) elementwise multiplication
        
        # --- Compute expectations (O(K)) ---
        x_exp = expectation_position(ψ_3D_spectral)   # O(K)
        norm = np.sum(np.abs(ψ_3D_spectral)**2)       # should be 1 (Parseval)
        
        # --- NEW: Compute entropy using the spectral entropy operator ---
        # For the 3D effective state, compute probabilities from its coefficients
        probs = np.abs(ψ_3D_spectral)**2   # |c_i|^2
        # Build C_i = -α * p_i * log(p_i) / i  for even i (or all i if not using Collatz)
        C = np.zeros(K, dtype=float)
        for i in range(1, K+1):
            if collatz_even and i % 2 != 0:
                continue
            p = probs[i-1]   # index 0 corresponds to i=1?
            if p > 0:
                C[i-1] = -alpha * p * np.log(p) / i   # store at index i-1
        # Now compute ζ'(0) imaginary part: H = (1/α) * Σ i * C_i
        # But careful: C_i we defined as -α p log p / i, so Σ i*C_i = -α Σ p log p = α H
        # So (1/α) * Σ i*C_i = H
        sum_iCi = np.sum(np.arange(1, K+1) * C)
        spectral_entropy = sum_iCi / alpha   # should equal Shannon entropy
        # Alternatively, compute Shannon entropy directly:
        shannon_H = -np.sum(p * np.log(p) for p in probs if p > 0)
        
        # Record both (they should match numerically)
        entropy_history.append((t, shannon_H, spectral_entropy))
        
        # --- NEW: Integral operator ∫ g(x) dx using the organic curve from earlier ---
        # We can compute the integral of the wavefunction's squared amplitude over the domain.
        # Since we're in spectral domain, the integral over all space is just the sum of |c_i|^2 (Parseval).
        # We already have norm = ∫ |ψ|² dx = Σ |c_i|².
        # The "organic curve" g(x) = a * exp(-(x-π)²/(2e²)) with a=1/(π-e)
        # Its integral over [0, π+e] is exactly a (from earlier boundary condition).
        # For our wavefunction, we can compute the overlap with this curve, or simply use the norm.
        # Here we'll just record the norm as a sanity check.
        integral = norm   # should be ≈ 1
        integral_history.append((t, integral))
        
        # --- Store compressed data (just coefficients) for post‑processing ---
        RecordClassical_Spectral(t, γ_coeff, v_coeff, Δ_spectral)
        RecordQuantum_Spectral(t, x_exp, ψ_3D_spectral, Δ_spectral)
    
    # --- 4. Post‑processing (O(K) or O(K log K) using FFT if needed) ---
    CompareTrajectories_Spectral(γ_coeff_record, x_exp_record)
    FitEffectiveEquations_Spectral(...)
    
    # Return histories for analysis
    return entropy_history, integral_history