

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

# Parameters
N                  = 300     # particles
G                  = 1.0     # gravitational constant 
M1_INIT            = 2.0     # initial donor mass
M2_INIT            = 5.0     # initial accretor mass
MASS_TRANSFER_RATE = 2       # mass lost per captured particle per unit time
M1_MIN             = 0.5     # minimum donor mass
EPS_SOFTENING      = 1e-3    # gravitational softening
R_COMPACT_FRAC     = 0.08    # accretor radius as fraction of r
BRACKET_MULT       = 12      # L2/L3 search extent = BRACKET_MULT * r


x1, y1 =  1.0, 0.0
x2, y2 = -1.0, 0.0

r = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)   # orbital separation, constant
R_CAPTURE = R_COMPACT_FRAC * r              # accretor capture


dt                   = 0.002   # RK4 timestep
T_MAX                = 20.0    # max simulation time
CONTOUR_UPDATE_STEPS = 800     # potential contour refresh cadence


# LAGRANGE POINT & ROCHE GEOMETRY

def get_system_geometry(M1_c, M2_c):
    
    M_tot  = M1_c + M2_c
    w_c    = np.sqrt(G * M_tot / r**3)
    xcom_c = (M1_c * x1 + M2_c * x2) / M_tot

    
    def acc_x(x):
        d1    = x - x1
        d2    = x - x2
        r1    = abs(d1) + EPS_SOFTENING
        r2    = abs(d2) + EPS_SOFTENING
        f_g1  = -G * M1_c * d1 / r1**3
        f_g2  = -G * M2_c * d2 / r2**3
        f_cen = w_c**2 * (x - xcom_c)
        return f_g1 + f_g2 + f_cen

    def bisect(f, a, b, steps=60):
        # 60 iterations
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            print(
                f"[WARN] bisect: no sign change in [{a:.4f}, {b:.4f}] "
                f"for M1={M1_c:.3f}, M2={M2_c:.3f}. Using midpoint."
            )
            return (a + b) / 2.0
        for _ in range(steps):
            c  = (a + b) / 2.0
            fc = f(c)
            if fa * fc <= 0:
                b = c;  fb = fc
            else:
                a = c;  fa = fc
        return (a + b) / 2.0

    x_lo      = min(x1, x2)
    x_hi      = max(x1, x2)
    far_limit = BRACKET_MULT * r

    L1_c = bisect(acc_x, x_lo + 0.01,      x_hi - 0.01)
    L2_c = bisect(acc_x, x_hi + 0.01,      x_hi + far_limit)
    L3_c = bisect(acc_x, x_lo - far_limit, x_lo - 0.01)

    
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    ux    = (x2 - x1) / r
    uy    = (y2 - y1) / r
    perp_x = uy
    perp_y = -ux
    h      = (np.sqrt(3.0) / 2.0) * r

    x_L4 = mid_x + h * perp_x
    y_L4 = mid_y + h * perp_y
    x_L5 = mid_x - h * perp_x
    y_L5 = mid_y - h * perp_y

    return w_c, xcom_c, L1_c, L2_c, L3_c, x_L4, y_L4, x_L5, y_L5



# EFFECTIVE POTENTIAL & JACOBI CONSTANT

def effective_potential_phi(x, y, M1_c, M2_c, w_c, xcom_c):
    
    r1    = np.sqrt((x - x1)**2 + (y - y1)**2 + EPS_SOFTENING)
    r2    = np.sqrt((x - x2)**2 + (y - y2)**2 + EPS_SOFTENING)
    phi_g = -G * M1_c / r1 - G * M2_c / r2
    phi_f = -0.5 * w_c**2 * ((x - xcom_c)**2 + y**2)
    return phi_g + phi_f


def compute_jacobi(x, y, vx, vy, M1_c, M2_c, w_c, xcom_c):
    
    phi  = effective_potential_phi(x, y, M1_c, M2_c, w_c, xcom_c)
    v_sq = vx**2 + vy**2
    return -2.0 * phi - v_sq


def get_donor_boundary(M1_c, M2_c, w_c, xcom_c, phi_surface, dL1_curr, num_pts=100):
   
    thetas = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    cos_t  = np.cos(thetas)
    sin_t  = np.sin(thetas)

    rho_plane = np.where(cos_t < -1e-6, dL1_curr / np.maximum(-cos_t, 1e-6), dL1_curr * 1.4)
    rho_hi    = np.minimum(rho_plane, dL1_curr * 1.35)
    rho_lo    = np.full(num_pts, 0.01)

    for _ in range(16):
        rho_mid = 0.5 * (rho_lo + rho_hi)
        x = x1 + rho_mid * cos_t
        y = y1 + rho_mid * sin_t
        phi_val = effective_potential_phi(x, y, M1_c, M2_c, w_c, xcom_c)
        inside  = (phi_val < phi_surface)
        rho_lo  = np.where(inside, rho_mid, rho_lo)
        rho_hi  = np.where(inside, rho_hi, rho_mid)

    rho = 0.5 * (rho_lo + rho_hi)
    return np.column_stack([x1 + rho * cos_t, y1 + rho * sin_t])



# INITIAL GEOMETRY & DONOR PHYSICAL RADIUS

M1, M2 = float(M1_INIT), float(M2_INIT)
w, xcom, L1, L2, L3, x_L4, y_L4, x_L5, y_L5 = get_system_geometry(M1, M2)

dL1 = abs(x1 - L1)            # donor starts filling its Roche lobe at L1
R1_INIT = dL1
ZETA_DONOR = 0.4              # mass-radius index


# PARTICLE ARRAYS & NOZZLE SPAWNING

xp  = np.zeros(N, dtype=float)
yp  = np.zeros(N, dtype=float)
vxp = np.zeros(N, dtype=float)
vyp = np.zeros(N, dtype=float)
C0  = np.zeros(N, dtype=float)   


def spawn_particles(mask, dL1_curr, M1_c, M2_c, w_c, xcom_c):
    
    count = int(np.sum(mask))
    if count == 0:
        return

    theta = np.random.normal(np.pi, 0.05, count)
    xp[mask] = x1 + dL1_curr * np.cos(theta)
    yp[mask] = y1 + dL1_curr * np.sin(theta)

    
    vx_kick   = -0.15
    vxp[mask] = np.random.normal(vx_kick, 0.03, count)
    vyp[mask] = np.random.normal(0.0,     0.03, count)

    C0[mask] = compute_jacobi(
        xp[mask], yp[mask], vxp[mask], vyp[mask],
        M1_c, M2_c, w_c, xcom_c
    )


spawn_particles(np.ones(N, dtype=bool), dL1, M1, M2, w, xcom)



# RK4 INTEGRATOR

def acc_particle(xp_a, yp_a, vxp_a, vyp_a, M1_c, M2_c, w_c, xcom_c):
    """Vectorised co-rotating-frame acceleration: gravity + Coriolis + centrifugal."""
    r1  = np.sqrt((xp_a - x1)**2 + (yp_a - y1)**2 + EPS_SOFTENING)
    r2  = np.sqrt((xp_a - x2)**2 + (yp_a - y2)**2 + EPS_SOFTENING)

    axp = (-(G * M1_c * (xp_a - x1)) / r1**3-(G * M2_c * (xp_a - x2)) / r2**3+ 2.0 * w_c * vyp_a+ w_c**2 * (xp_a - xcom_c))
    ayp = ( -(G * M1_c * (yp_a - y1)) / r1**3-(G * M2_c * (yp_a - y2)) / r2**3- 2.0 * w_c * vxp_a+ w_c**2 * yp_a)
    return axp, ayp


def runge_kutta_4(x, y, vx, vy, dt_step, M1_c, M2_c, w_c, xcom_c):
    """One RK4 step for all N particles at once; state shape (4, N)."""
    def derivs(x_s, y_s, vx_s, vy_s):
        ax, ay = acc_particle(x_s, y_s, vx_s, vy_s, M1_c, M2_c, w_c, xcom_c)
        return np.array([vx_s, vy_s, ax, ay])

    state = np.array([x, y, vx, vy])
    k1    = derivs(*state)
    k2    = derivs(*(state + 0.5 * dt_step * k1))
    k3    = derivs(*(state + 0.5 * dt_step * k2))
    k4    = derivs(*(state +       dt_step * k3))
    return state + (dt_step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)



# POTENTIAL GRID

XGRID  = np.linspace(-4, 4, 300)
YGRID  = np.linspace(-4, 4, 300)
XG, YG = np.meshgrid(XGRID, YGRID)


def compute_phi_grid(M1_c, M2_c, w_c, xcom_c):
    return effective_potential_phi(XG, YG, M1_c, M2_c, w_c, xcom_c)



# MAIN SIMULATION LOOP

t    = 0.0
step = 0

XP, YP           = [], []
STAR1_POLYGONS   = []     # donor's tidal teardrop boundary per frame
R2_history       = []     # accretor display radius per frame
M1_history       = []
M2_history       = []
L_POINTS         = []     # (L1, L2, L3, x_L4, y_L4, x_L5, y_L5) per frame
DELTA_C          = []     # max relative Jacobi drift per frame
OVERFLOW_HISTORY = []     # True while donor overflows its Roche lobe

PHI_SNAPSHOTS          = [(compute_phi_grid(M1, M2, w, xcom),effective_potential_phi(L1, 0.0, M1, M2, w, xcom))]
SNAPSHOT_FRAME_INDICES = [0]

t_detach   = None
R1_detach  = None
overflowing = True

while t < T_MAX:
    state = runge_kutta_4(xp, yp, vxp, vyp, dt, M1, M2, w, xcom)
    xp, yp, vxp, vyp = state[0], state[1], state[2], state[3]

    rp1 = np.sqrt((xp - x1)**2 + (yp - y1)**2)
    rp2 = np.sqrt((xp - x2)**2 + (yp - y2)**2)

    w, xcom, L1, L2, L3, x_L4, y_L4, x_L5, y_L5 = get_system_geometry(M1, M2)
    dL1 = abs(x1 - L1)

    # Mass-radius relation while overflowing
    if overflowing:
        R1_phys = R1_INIT * (M1 / M1_INIT)**ZETA_DONOR
        overflowing = (R1_phys >= dL1) and (M1 > M1_MIN)
        if not overflowing and t_detach is None:
            t_detach  = t
            R1_detach = R1_phys
    else:
        dt_post = t - t_detach
        R1_phys = R1_detach * (0.65 + 0.35 * np.exp(-1.5 * dt_post))

    captured     = rp2 < R_CAPTURE
    num_captured = int(np.sum(captured))

    if num_captured > 0 and overflowing:
        dm  = MASS_TRANSFER_RATE * num_captured * dt
        M1  = max(M1_MIN, M1 - dm)
        M2 += dm

    if overflowing:
        phi_surf = effective_potential_phi(L1, 0.0, M1, M2, w, xcom)
    else:
        phi_surf = effective_potential_phi(x1 - R1_phys, 0.0, M1, M2, w, xcom)
    poly_donor = get_donor_boundary(M1, M2, w, xcom, phi_surf, dL1)

    r_star2 = R_CAPTURE

    # Reseed particles that reached a star
    reseed_mask = (captured| (rp1 < 0.5 * dL1)| (rp2 < 0.5 * R_CAPTURE)| (np.abs(xp) > 4.0)| (np.abs(yp) > 4.0))

    if np.any(reseed_mask):
        if overflowing:
            spawn_particles(reseed_mask, dL1, M1, M2, w, xcom)
        else:
            xp[reseed_mask]  = 100.0
            yp[reseed_mask]  = 100.0
            vxp[reseed_mask] = 0.0
            vyp[reseed_mask] = 0.0

    C_curr = compute_jacobi(xp, yp, vxp, vyp, M1, M2, w, xcom)
    valid = (np.abs(C0) > 1e-10) & (np.abs(xp) <= 4.0) & (np.abs(yp) <= 4.0)
    delta_C_rel = float(np.max(np.abs((C_curr[valid] - C0[valid]) / C0[valid]))) \
                  if np.any(valid) else 0.0

    if step > 0 and step % CONTOUR_UPDATE_STEPS == 0:
        phi_snap    = compute_phi_grid(M1, M2, w, xcom)
        phi_L1_snap = effective_potential_phi(L1, 0.0, M1, M2, w, xcom)
        PHI_SNAPSHOTS.append((phi_snap, phi_L1_snap))
        SNAPSHOT_FRAME_INDICES.append(len(XP))

    t    += dt
    step += 1
    XP.append(xp.copy())
    YP.append(yp.copy())
    STAR1_POLYGONS.append(poly_donor)
    R2_history.append(r_star2)
    M1_history.append(M1)
    M2_history.append(M2)
    L_POINTS.append((L1, L2, L3, x_L4, y_L4, x_L5, y_L5))
    DELTA_C.append(delta_C_rel)
    OVERFLOW_HISTORY.append(overflowing)



# PLOT & ANIMATION 

fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-3.5, 3.5)
axes.set_ylim(-3.5, 3.5)
axes.set_aspect('equal')
axes.set_facecolor('#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')
axes.tick_params(colors='white')
axes.set_title(f'Binary Mass Transfer  (M1={M1_history[0]:.1f}, M2={M2_history[0]:.1f})',color='white', pad=15)
for spine in axes.spines.values():
    spine.set_edgecolor('#444')

phi0, phi_L1_0 = PHI_SNAPSHOTS[0]
contour_bg = [axes.contour(XG, YG, phi0, levels=150, cmap='plasma', alpha=0.35)]
contour_rl = [axes.contour(XG, YG, phi0, levels=[phi_L1_0],
                            colors='white', linestyles='--')]

star1 = Polygon(STAR1_POLYGONS[0], facecolor='tomato', edgecolor='salmon',
                alpha=0.85, zorder=3, label='M1 (Donor)')
star2 = Circle((x2, y2), R2_history[0], color='dodgerblue', alpha=0.85,
               zorder=3, label='M2 (Accretor)')
axes.add_patch(star1)
axes.add_patch(star2)

l1_0, l2_0, l3_0, xl4_0, yl4_0, xl5_0, yl5_0 = L_POINTS[0]
l_scatter = axes.scatter([l1_0, l2_0, l3_0, xl4_0, xl5_0],[0.0,  0.0,  0.0,  yl4_0, yl5_0],color='cyan', marker='*', s=80, zorder=4, label='L Points')

particles, = axes.plot([], [], 'o', color='crimson', markersize=3,zorder=5, label='Gas Stream')
axes.plot([], [], '--', color='white', label='Roche Lobe Boundary')

axes.legend(loc='lower right', facecolor='#1a1a1a', edgecolor='#444',labelcolor='white', fontsize=8)

info_box = axes.text(0.03, 0.96, '', transform=axes.transAxes, color='white',fontsize=9, verticalalignment='top', zorder=10,bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.85, edgecolor='#555'))

_active_snapshot = [0]   


def remove_contour(cs):
    
    if hasattr(cs, 'remove'):
        cs.remove()
    elif hasattr(cs, 'collections'):
        for coll in cs.collections:
            coll.remove()



# ANIMATION FUNCTION

def animate(i):
    particles.set_data(XP[i], YP[i])

    star1.set_xy(STAR1_POLYGONS[i])
    star2.set_radius(R2_history[i])

    l1_i, l2_i, l3_i, xl4_i, yl4_i, xl5_i, yl5_i = L_POINTS[i]
    l_scatter.set_offsets(np.c_[[l1_i, l2_i, l3_i, xl4_i, xl5_i],  [0.0,  0.0,  0.0,  yl4_i, yl5_i]])

    
    snap_idx = 0
    for k, frame_thresh in enumerate(SNAPSHOT_FRAME_INDICES):
        if i >= frame_thresh:
            snap_idx = k
        else:
            break

    if snap_idx != _active_snapshot[0]:
        _active_snapshot[0] = snap_idx
        phi_s, phi_L1_s = PHI_SNAPSHOTS[snap_idx]
        remove_contour(contour_bg[0])
        remove_contour(contour_rl[0])
        contour_bg[0] = axes.contour(XG, YG, phi_s, levels=150, cmap='plasma', alpha=0.35)
        contour_rl[0] = axes.contour(XG, YG, phi_s, levels=[phi_L1_s],colors='white', linestyles='--')

    m1_cur = M1_history[i]
    m2_cur = M2_history[i]
    status = 'Overflowing (Active)' if OVERFLOW_HISTORY[i] else 'Detached (Transfer Ended)'
    info_box.set_text(f"M1 (Donor): {m1_cur:.3f}  |  M2 (Accretor): {m2_cur:.3f}\n"f"State: {status}\n"f"Jacobi drift ΔC/C₀: {DELTA_C[i]:.2e}")

    return (particles, star1, star2, l_scatter, info_box)


ani = FuncAnimation(fig, animate, frames=len(XP), interval=10, blit=False)
plt.show()