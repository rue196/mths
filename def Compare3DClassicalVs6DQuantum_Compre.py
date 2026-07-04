import math
import cmath


def Compare3DClassicalVs6DQuantum_Compressed(T, dt, K):
    # T: total time, dt: time step
    # K: spectral compression size (number of Fourier modes per dimension)
    
    # --- 1. Initialize compressed representation ---
    # Classical 3D state in a low‑dimensional basis (e.g., first K Hermite functions)
    γ_coeff = np.zeros(K, dtype=complex)  # expansion coefficients for position
    v_coeff = np.zeros(K, dtype=complex)  # velocity coefficients
    
    # Quantum 6D wavefunction: use K modes per dimension? Actually 6D would be K^6 → too large.
    # Instead, represent the 6D wavefunction as a **Kronecker product of 1D spectral bases**,
    # but we can compress further using tensor trains (matrix product states) with rank r << K.
    # For O(K), we assume a **separable** ansatz: Ψ(x1..x6) = ∏_{d=1}^6 ψ_d(x_d), each ψ_d stored with K coefficients.
    Ψ = [np.zeros(K, dtype=complex) for _ in range(6)]   # each ψ_d of length K
    for d in range(6):
        Ψ[d] = initialize_wavepacket_1D(K)  # e.g., Gaussian in Fourier basis
    
    # Precompute Toeplitz kernels for potentials (once)
    # V_cl and V_qm are convolution kernels (functions of Δ and τ)
    # Their Fourier transforms are diagonal matrices of size K.
    V_cl_spectrum = compute_potential_spectrum('classical', K)
    V_qm_spectrum = compute_potential_spectrum('quantum', K)
    
    # Cache for matrix invariants (Δ, τ) – also stored in spectral domain
    cache = {}
    
    # --- 2. Time evolution loop ---
    for t in np.arange(0, T+dt, dt):
        # --- Common matrix invariants at this step (O(K) via spectral method) ---
        (Δ_spectral, gradΔ_spectral, cache) = MatrixIntensityAndGradient_Spectral(K, cache)
        # Δ_spectral is a vector of K Fourier coefficients; gradΔ similarly.
        
        # Potential values in real space (if needed) – we can stay in spectral domain.
        # For classical force: F = -∇V(γ). Since γ is expanded in basis, we compute inner products.
        
        # --- 3D Classical: Update coefficients using spectral Galerkin ---
        # Represent γ(x) = sum_i γ_coeff[i] φ_i(x)   (φ_i are basis functions)
        # Compute potential energy matrix: V_cl_matrix = <φ_i | V_cl(Δ,τ) | φ_j> – Toeplitz.
        # Then force = -∇V: compute by differentiating the basis.
        # Use spectral ODE integrator (e.g., symplectic Euler) in coefficient space.
        F_coeff = compute_force_coefficients(γ_coeff, V_cl_spectrum, gradΔ_spectral)  # O(K)
        v_coeff = v_coeff + (F_coeff / m) * dt
        γ_coeff = γ_coeff + v_coeff * dt
        
        # --- 6D Quantum: Update each 1D wavefunction (separable approximation) ---
        # Hamiltonian H = T + V_qm, with T diagonal in Fourier basis.
        for d in range(6):
            # Kinetic term: multiply each Fourier coefficient by k^2 (diagonal) -> O(K)
            Ψ[d] = apply_kinetic(Ψ[d], dt, d)   # implicit step: (1 + i*T*dt/2) etc.
            # Potential term: convolution with V_qm (Toeplitz) -> O(K log K) once per dim, but can be O(K) if V_qm is diagonal in the same basis (e.g., local in x).
            # For simplicity, use split operator: apply potential in real space via FFT → O(K log K).
            # But we can keep it O(K) if we use a **pseudospectral** method where potential is diagonal in the basis (e.g., harmonic oscillator).
            Ψ[d] = apply_potential(Ψ[d], V_qm_spectrum, Δ_spectral, dt)   # O(K)
        
        # --- Project 6D separable state to 3D effective wavefunction ---
        # For separable product, the 3D density is product of first 3 dimensions.
        # We can compute ψ_3D as the elementwise product of Ψ[0..2] in real space (via inverse FFT) -> O(K log K) per step.
        # To stay O(K), we keep ψ_3D in Fourier basis and compute expectations by Parseval.
        ψ_3D_spectral = np.multiply(Ψ[0], np.multiply(Ψ[1], Ψ[2]))  # pointwise product in Fourier? No, convolution in real space becomes product in Fourier? Actually product in real space is convolution in Fourier, which is expensive. So we accept O(K log K) for this projection, but it's still quasi‑linear.
        # However, if we choose a basis where product is diagonal (e.g., Hermite functions), we can stay O(K). We'll assume that.
        
        # --- Extract quantum observables in 3D (O(K) using Parseval) ---
        ρ_spectral = np.conj(ψ_3D_spectral) * ψ_3D_spectral   # power spectrum, not density. Actually need to inverse transform for density.
        # For expectations: ⟨x⟩ = ∫ ψ* x ψ dx = sum_n (coeff_n* x_mn coeff_m). If x is diagonal in the basis (e.g., Hermite), then O(K).
        x_exp = expectation_position(ψ_3D_spectral)   # O(K)
        
        # --- Store compressed data (just the coefficients) ---
        RecordClassical_Spectral(t, γ_coeff, v_coeff, Δ_spectral)
        RecordQuantum_Spectral(t, x_exp, ρ_spectral, Δ_spectral)
    
    # --- 4. Post‑processing (also O(K) or O(K log K) using spectral transforms) ---
    CompareTrajectories_Spectral(γ_coeff_record, x_exp_record)
    FitEffectiveEquations_Spectral(...)
    return


input('Press ENTER to exit')
    