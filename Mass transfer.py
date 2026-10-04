import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ARBITRARY SYSTEM PARAMETERS
N = 150                      # Number of gas stream particles
G = 1.0                      # Gravitational constant
M1 = 6.0                     # Donor Mass 
M2 = 4.0                     # Accretor Mass 
mass_transfer_rate = 0.5   # Continuous mass loss rate

x1, y1 = 1.0, 0.0
x2, y2 = -1.0, 0.0
r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) 

# LAGRANGE & ROCHE GEOMETRY SOLVER

def get_system_geometry(M1_c, M2_c):
    '''Computes exact L points'''
    w_c = np.sqrt(G * (M1_c + M2_c) / r**3)
    xcom_c = (M1_c * x1 + M2_c * x2) / (M1_c + M2_c)
    
    # 1D net acceleration along line of centers (y=0)
    def acc_x(x):
        r1 = np.abs(x - x1) + 1e-6
        r2 = np.abs(x - x2) + 1e-6
        
        f_g1 = G * M1_c / r1**2 if x < x1 else -G * M1_c / r1**2
        f_g2 = G * M2_c / r2**2 if x < x2 else -G * M2_c / r2**2
        f_cent = w_c**2 * (x - xcom_c)
        return f_g1 + f_g2 + f_cent

    def bisect(f, a, b, steps=60):
        for _ in range(steps):
            c = (a + b) / 2.0
            if f(a) * f(c) <= 0:
                b = c
            else:
                a = c
        return (a + b) / 2.0

    
    L1_c = bisect(acc_x, -0.98, 0.98)
    L2_c = bisect(acc_x, 1.02, 5.0)
    L3_c = bisect(acc_x, -5.0, -1.02)

    x_L45 = (x1 + x2) / 2.0
    y_L4 = +(np.sqrt(3) / 2.0) * r
    y_L5 = -(np.sqrt(3) / 2.0) * r

    return w_c, xcom_c, L1_c, L2_c, L3_c, x_L45, y_L4, y_L5


w, xcom, L1, L2, L3, x_L45, y_L4, y_L5 = get_system_geometry(M1, M2)


dL1 = np.abs(x1 - L1)
R1 = dL1
R2 = 0.08 * r  


# EFFECTIVE POTENTIAL & JACOBI CONSTANT

def effective_potential_phi(x, y, M1_c, M2_c, w_c, xcom_c):
    eps = 1e-3
    r1 = np.sqrt((x - x1)**2 + (y - y1)**2 + eps)
    r2 = np.sqrt((x - x2)**2 + (y - y2)**2 + eps)
    phi_g = -((G * M1_c) / r1) - ((G * M2_c) / r2)
    phi_f = -0.5 * w_c**2 * ((x - xcom_c)**2 + y**2)
    return phi_g + phi_f

def compute_jacobi(x, y, vx, vy, M1_c, M2_c, w_c, xcom_c):
    phi = effective_potential_phi(x, y, M1_c, M2_c, w_c, xcom_c)
    v_sq = vx**2 + vy**2
    return -2.0 * phi - v_sq


# PARTICLE ARRAYS & NOZZLE RESEEDING

xp = np.zeros(N)
yp = np.zeros(N)
vxp = np.zeros(N)
vyp = np.zeros(N)
C0 = np.zeros(N)

def spawn_particles(mask, dL1_curr, M1_curr, M2_curr, w_curr, xcom_curr):
    count = np.sum(mask)
    if count == 0:
        return
    
    # Spawn EXACTLY at the L1 nozzle radius
    theta = np.random.normal(np.pi, 0.05, count)
    xp[mask] = x1 + dL1_curr * np.cos(theta)
    yp[mask] = y1 + dL1_curr * np.sin(theta)
    
    # Give them a definitive push 
    vxp[mask] = np.random.normal(-0.15, 0.03, count)
    vyp[mask] = np.random.normal(0.0, 0.03, count)
    
    C0[mask] = compute_jacobi(xp[mask], yp[mask], vxp[mask], vyp[mask], M1_curr, M2_curr, w_curr, xcom_curr)

spawn_particles(np.ones(N, dtype=bool), dL1, M1, M2, w, xcom)


# RK4 INTEGRATOR

def acc_particle(xp, yp, vxp, vyp, M1_c, M2_c, w_c, xcom_c):
    eps = 1e-3
    rp1 = np.sqrt((xp - x1)**2 + (yp - y1)**2 + eps)
    rp2 = np.sqrt((xp - x2)**2 + (yp - y2)**2 + eps)
    
    axp = -(G * M1_c * (xp - x1)) / rp1**3 - (G * M2_c * (xp - x2)) / rp2**3 + (2.0 * w_c * vyp) + (w_c**2 * (xp - xcom_c))
    ayp = -(G * M1_c * yp) / rp1**3 - (G * M2_c * yp) / rp2**3 - (2.0 * w_c * vxp) + (w_c**2 * yp)
    return axp, ayp

def runge_kutta_4(x, y, vx, vy, dt, M1_c, M2_c, w_c, xcom_c):
    def derivs(x_s, y_s, vx_s, vy_s):
        ax, ay = acc_particle(x_s, y_s, vx_s, vy_s, M1_c, M2_c, w_c, xcom_c)
        return np.array([vx_s, vy_s, ax, ay])

    state = np.array([x, y, vx, vy])
    k1 = derivs(x, y, vx, vy)
    k2 = derivs(*(state + 0.5 * k1 * dt))
    k3 = derivs(*(state + 0.5 * k2 * dt))
    k4 = derivs(*(state + k3 * dt))
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# MAIN SIMULATION LOOP

t = 0
dt = 0.002
XP, YP = [], []
R1_change, R2_change = [], []
M1_change, M2_change = [], []
L_POINTS = []
DELTA_C = []

while t < 20:
    state = runge_kutta_4(xp, yp, vxp, vyp, dt, M1, M2, w, xcom)
    xp, yp, vxp, vyp = state[0], state[1], state[2], state[3]

    rp1 = np.sqrt((xp - x1)**2 + (yp - y1)**2)
    rp2 = np.sqrt((xp - x2)**2 + (yp - y2)**2)

    captured = (rp2 < R2)
    num_captured = np.sum(captured)
    
    if num_captured > 0:
        dm = mass_transfer_rate * num_captured * dt
        M1 = max(0.5, M1 - dm)
        M2 += dm

    w, xcom, L1, L2, L3, x_L45, y_L4, y_L5 = get_system_geometry(M1, M2)
    dL1 = np.abs(x1 - L1)
    R1 = dL1

    # Reseed if captured
    reseed_mask = captured | (rp1 < 0.50 * R1) | (np.abs(xp) > 4) | (np.abs(yp) > 4)
    if np.any(reseed_mask):
        spawn_particles(reseed_mask, dL1, M1, M2, w, xcom)

    C_curr = compute_jacobi(xp, yp, vxp, vyp, M1, M2, w, xcom)
    delta_C_rel = np.max(np.abs((C_curr - C0) / C0))

    t += dt
    XP.append(xp.copy())
    YP.append(yp.copy())
    R1_change.append(R1)
    R2_change.append(R2)
    M1_change.append(M1)
    M2_change.append(M2)
    L_POINTS.append((L1, L2, L3, x_L45, y_L4, y_L5))
    DELTA_C.append(delta_C_rel)


# PLOT & ANIMATION SETUP

fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-3.5, 3.5)
axes.set_ylim(-3.5, 3.5)
axes.set_aspect('equal')
axes.set_facecolor('#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')
axes.tick_params(colors='white')
axes.set_title(f'Binary Mass Transfer (M1={M1_change[0]:.1f}, M2={M2_change[0]:.1f})', color='white', pad=15)

for spine in axes.spines.values():
    spine.set_edgecolor('#444')

# Equipotential contours
xgrid = np.linspace(-4, 4, 300)
ygrid = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(xgrid, ygrid)
phi = effective_potential_phi(X, Y, M1_change[0], M2_change[0], w, xcom)
phi_l1 = effective_potential_phi(L_POINTS[0][0], 0, M1_change[0], M2_change[0], w, xcom)

axes.contour(X, Y, phi, levels=200, cmap='plasma', alpha=0.5)
axes.contour(X, Y, phi, levels=[phi_l1], colors='white', linestyles='--')

star1 = Circle((x1, y1), R1_change[0], color='tomato', alpha=0.85, zorder=3, label='M1 (Donor)')
star2 = Circle((x2, y2), R2_change[0], color='dodgerblue', alpha=0.85, zorder=3, label='M2 (Accretor)')
axes.add_patch(star1)
axes.add_patch(star2)

l1_0, l2_0, l3_0, xl45_0, yl4_0, yl5_0 = L_POINTS[0]
l_scatter = axes.scatter([l1_0, l2_0, l3_0, xl45_0, xl45_0], [0, 0, 0, yl4_0, yl5_0], 
                         color='cyan', marker='*', s=80, zorder=4, label='L Points')

particles, = axes.plot([], [], 'o', color='crimson', markersize=3, zorder=5, label='Gas Stream')
axes.plot([], [], '--', color='white', label='Roche Lobe Boundary')

axes.legend(loc='lower right', facecolor='#1a1a1a', edgecolor='#444', labelcolor='white', fontsize=8)

info_box = axes.text(0.03, 0.96, '', transform=axes.transAxes, color='white',
                     fontsize=9, verticalalignment='top', zorder=10,
                     bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.85, edgecolor='#555'))

def animate(i):
    particles.set_data(XP[i], YP[i])
    star1.set_radius(R1_change[i])
    star2.set_radius(R2_change[i])
    
    l1_i, l2_i, l3_i, xl45_i, yl4_i, yl5_i = L_POINTS[i]
    l_scatter.set_offsets(np.c_[[l1_i, l2_i, l3_i, xl45_i, xl45_i], [0, 0, 0, yl4_i, yl5_i]])

    info_box.set_text(
        f"M1: {M1_change[i]:.3f} | M2: {M2_change[i]:.3f}\n"
        f"Jacobi Drift (ΔC/C₀): {DELTA_C[i]:.2e}"
    )
    return (particles, star1, star2, l_scatter, info_box)

ani = FuncAnimation(fig, animate, frames=len(XP), interval=10, blit=False)
plt.show()