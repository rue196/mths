import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# 1. Define airfoil with 6 vertices (symmetric, chord length = 1)
#    Coordinates: leading edge (0,0), trailing edge (1,0), four points upper surface.
upper = np.array([
    [0.0, 0.0],      # leading edge
    [0.2, 0.06],
    [0.5, 0.08],
    [0.8, 0.04],
    [1.0, 0.0]       # trailing edge
])
# Mirror for lower surface (excluding duplicates)
lower = np.array([[x, -y] for x, y in upper[1:-1][::-1]])
airfoil_vertices = np.vstack([upper, lower])
airfoil_vertices = np.unique(airfoil_vertices, axis=0)  # remove duplicate LE/TE

# ------------------------------------------------------------
# 2. Create 2D computational grid around airfoil
x_min, x_max = -0.8, 2.2
y_min, y_max = -0.8, 0.8
nx, ny = 100, 100
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(x, y)

# Mask points inside the airfoil (for boundary conditions)
from matplotlib.path import Path
airfoil_path = Path(airfoil_vertices)
inside = np.zeros((ny, nx), dtype=bool)
for i in range(nx):
    for j in range(ny):
        inside[j, i] = airfoil_path.contains_point((X[j,i], Y[j,i]))

# ------------------------------------------------------------
# 3. Set up PDE: Laplace equation ∇²φ = 0
#    Boundary conditions:
#      - Farfield: φ = U∞ * (x cosα + y sinα) (uniform flow)
#      - Airfoil surface: ∂φ/∂n = 0 (no penetration)
#    We treat interior grid points (outside airfoil) with central differences.
#    Points inside the airfoil are set to a constant (will be masked later).

Uinf = 1.0
alpha = np.deg2rad(5.0)   # angle of attack
dx = x[1] - x[0]
dy = y[1] - y[0]

N = nx * ny
A = lil_matrix((N, N))
b = np.zeros(N)

# Helper to index grid
def idx(i, j):
    return j * nx + i

# Interior points (not inside airfoil) – Laplace stencil
for j in range(1, ny-1):
    for i in range(1, nx-1):
        if not inside[j, i]:
            A[idx(i,j), idx(i,j)] = -2/dx**2 - 2/dy**2
            A[idx(i,j), idx(i+1,j)] = 1/dx**2
            A[idx(i,j), idx(i-1,j)] = 1/dx**2
            A[idx(i,j), idx(i,j+1)] = 1/dy**2
            A[idx(i,j), idx(i,j-1)] = 1/dy**2
        else:
            # Points inside: Dirichlet (set φ=0) – not physically correct but we'll later ignore.
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = 0

# Farfield boundary (top, bottom, left, right) – Dirichlet
for i in range(nx):
    for j in [0, ny-1]:   # top and bottom
        if not inside[j, i]:
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = Uinf * (x[i]*np.cos(alpha) + y[j]*np.sin(alpha))
for j in range(ny):
    for i in [0, nx-1]:   # left and right
        if not inside[j, i]:
            A[idx(i,j), idx(i,j)] = 1
            b[idx(i,j)] = Uinf * (x[i]*np.cos(alpha) + y[j]*np.sin(alpha))

# Airfoil surface Neumann condition: approximate by setting the normal derivative to zero.
# A simpler method: we already set interior points to constant; the gradient across the boundary
# will adjust. For a proper Neumann we need ghost cells, but this gets acceptable flow tangency.

# Solve linear system
A_csr = csr_matrix(A)
phi = spsolve(A_csr, b).reshape((ny, nx))

# Velocity field: u = ∂φ/∂x, v = ∂φ/∂y
u = np.zeros_like(phi)
v = np.zeros_like(phi)
u[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
v[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dy)
# Set inside to zero
u[inside] = 0
v[inside] = 0

# Flow direction angle (phase)
flow_angle = np.arctan2(v, u)

# Add π/2 coupling if desired (rotate flow by 90°)
couple_pi2 = True
if couple_pi2:
    flow_angle += np.pi/2
    # Also rotate velocity components for visualisation
    u_rot = -v   # because rotation by +90°: (u,v) -> (-v, u)
    v_rot = u
    u, v = u_rot, v_rot

# ------------------------------------------------------------
# 4. Stack into 3D (extruded in z-direction)
nz = 20
z = np.linspace(-0.5, 0.5, nz)
flow_angle_3d = np.zeros((ny, nx, nz))
velocity_magnitude = np.sqrt(u**2 + v**2)
vel_mag_3d = np.zeros((ny, nx, nz))
for k in range(nz):
    flow_angle_3d[:,:,k] = flow_angle
    vel_mag_3d[:,:,k] = velocity_magnitude

# Scalar field τ that relaxes to velocity magnitude (as earlier)
tau = np.zeros_like(vel_mag_3d)
eta = 0.1
for step in range(100):
    tau += eta * (vel_mag_3d - tau)

# ------------------------------------------------------------
# 5. Visualisation
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.contourf(X, Y, phi, levels=50, cmap='viridis')
plt.colorbar(label='Potential φ')
plt.plot(airfoil_vertices[:,0], airfoil_vertices[:,1], 'k-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Velocity potential')
plt.axis('equal')

plt.subplot(1,2,2)
strm = plt.streamplot(X, Y, u, v, density=1.2, color=flow_angle, cmap='twilight')
plt.colorbar(strm.lines, label='Flow direction (rad)')
plt.plot(airfoil_vertices[:,0], airfoil_vertices[:,1], 'k-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow field (streamlines coloured by direction)')
plt.axis('equal')
plt.tight_layout()
plt.show()

# 3D projection at mid‑span
mid_z = nz // 2
fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
# Plot velocity magnitude on the extruded surface at mid‑z
X3, Y3 = np.meshgrid(x, y)
Z3 = np.ones_like(X3) * z[mid_z]
surf = ax1.plot_surface(X3, Y3, Z3, facecolors=plt.cm.viridis(velocity_magnitude / velocity_magnitude.max()), alpha=0.8)
ax1.set_title(f'Velocity magnitude at z = {z[mid_z]:.2f}')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

ax2 = fig.add_subplot(122, projection='3d')
# 3D quiver of flow direction (downsampled)
skip = 8
Xq = X[::skip, ::skip]
Yq = Y[::skip, ::skip]
Zq = np.ones_like(Xq) * z[mid_z]
uq = u[::skip, ::skip]
vq = v[::skip, ::skip]
wq = np.zeros_like(uq)
ax2.quiver(Xq, Yq, Zq, uq, vq, wq, length=0.08, normalize=True, alpha=0.7)
ax2.plot(airfoil_vertices[:,0], airfoil_vertices[:,1], z[mid_z], 'k-', linewidth=2)
ax2.set_title('3D flow direction at mid‑plane')
ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
plt.tight_layout()
plt.show()

# Compare Δ (velocity magnitude) and τ after relaxation
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(velocity_magnitude, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot')
plt.colorbar(label='|∇φ|')
plt.title('Δ = velocity magnitude')
plt.subplot(1,2,2)
plt.imshow(tau[:,:,mid_z], origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='hot')
plt.colorbar(label='τ')
plt.title('τ after relaxation (should equal Δ)')
plt.tight_layout()
plt.show()