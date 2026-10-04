import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#--Physical parameters--
N = 30
G = 1
M1 = 6
M2 = 4

R1 = 1.22
R2 = 1

x1 = 1
y1 = 0

x2 = -1
y2 = 0

xp = np.zeros(N)
yp = np.zeros(N)

if M1 > M2:
    theta = np.random.uniform(0, 2*np.pi, N)

xp = x1 + R1*np.cos(theta)
yp = y1 + R1*np.sin(theta)

vxp = np.zeros(N)
vyp = np.zeros(N)


#--Derived quantities--
r = np.sqrt((x2-x1)**2 + (y2-y1)**2)

w = np.sqrt(G*(M1+M2)/(r**3))

xcom = (M1*x1 + M2*x2)/(M1+M2)
ycom = (M1*y1 + M2*y2)/(M1+M2)


#--Functions--

def acc_particle(xp, yp, vxp, vyp):
    '''Acceleration including gravity, Coriolis, and Centrifugal forces.'''

    eps = 1e-3

    rp1 = np.sqrt((xp-x1)**2 + (yp-y1)**2 + eps)
    rp2 = np.sqrt((xp-x2)**2 + (yp-y2)**2 + eps)

    axp = (
        -(G*M1*(xp-x1))/rp1**3
        -(G*M2*(xp-x2))/rp2**3
        +(2*w*vyp)
        +(w**2*(xp-xcom))
    )

    ayp = (
        -(G*M1*(yp-y1))/rp1**3
        -(G*M2*(yp-y2))/rp2**3
        -(2*w*vxp)
        +(w**2*(yp-ycom))
    )

    return axp, ayp


def effective_potential_phi(x, y):
    '''Potential at every point due to gravity and pseudo force.'''

    eps = 1e-3

    r1 = np.sqrt((x-x1)**2 + (y-y1)**2 + eps)
    r2 = np.sqrt((x-x2)**2 + (y-y2)**2 + eps)

    phi_g = -(G*M1)/r1 -(G*M2)/r2

    phi_f = -0.5*w**2*((x-xcom)**2 + (y-ycom)**2)

    return phi_g + phi_f


def acc_at_xaxis(p):
    '''Acceleration on x axis to find L1, L2, L3.'''

    r1 = np.abs(p-x1) + 0.001
    r2 = np.abs(p-x2) + 0.001

    return (
        -(G*M1*(p-x1))/r1**3
        -(G*M2*(p-x2))/r2**3
        +(w**2*(p-xcom))
    )


def acc_at_xcom(p):
    '''Acceleration on x=xcom to find L4, L5.'''

    r1 = np.sqrt((xcom-x1)**2 + (p-y1)**2 + 0.001)
    r2 = np.sqrt((xcom-x2)**2 + (p-y2)**2 + 0.001)

    return (
        -(G*M1*(p-y1))/r1**3
        -(G*M2*(p-y2))/r2**3
        +(w**2*(p-ycom))
    )


def bisection(f, a, b, n=50):

    if f(a)*f(b) >= 0:
        return (a+b)/2

    for _ in range(n):

        c = (a+b)/2

        if f(c) == 0:
            return c

        elif f(c)*f(a) < 0:
            b = c

        else:
            a = c

    return (a+b)/2


#--Integration by Leapfrog--

t = 0
dt = 0.01

XP = []
YP = []

while t < 20:

    # First half velocity step

    for i in range(N):

        axp, ayp = acc_particle(
            xp[i],
            yp[i],
            vxp[i],
            vyp[i]
        )

        vxp[i] += 0.5*axp*dt
        vyp[i] += 0.5*ayp*dt


    # Full position step

    for i in range(N):

        xp[i] += vxp[i]*dt
        yp[i] += vyp[i]*dt


    # Second half velocity step

    for i in range(N):

        axp, ayp = acc_particle(
            xp[i],
            yp[i],
            vxp[i],
            vyp[i]
        )

        vxp[i] += 0.5*axp*dt
        vyp[i] += 0.5*ayp*dt


    t += dt

    XP.append(xp.copy())
    YP.append(yp.copy())


#--Potential Contour Grid--

xgrid = np.linspace(-7, 7, 1000)
ygrid = np.linspace(-7, 7, 1000)

X, Y = np.meshgrid(xgrid, ygrid)

phi = effective_potential_phi(X, Y)


#--Lagrangian Points--

L1 = bisection(
    acc_at_xaxis,
    min(x1, x2)+0.01,
    max(x1, x2)-0.01,
    50
)

L2 = bisection(
    acc_at_xaxis,
    max(x1, x2)+0.01,
    3,
    50
)

L3 = bisection(
    acc_at_xaxis,
    -3,
    min(x1, x2)-0.01,
    50
)

L4 = bisection(
    acc_at_xcom,
    0.01,
    3,
    50
)

L5 = bisection(
    acc_at_xcom,
    -3,
    -0.01,
    50
)


#--Roche Lobe Potential--

phi_l1 = effective_potential_phi(L1, 0)


#--Plot--

fig, axes = plt.subplots(figsize=(10, 10))

axes.set_xlim(-7, 7)
axes.set_ylim(-7, 7)

axes.set_aspect('equal')

axes.set_facecolor('#0d0d0d')
fig.patch.set_facecolor('#0d0d0d')

axes.tick_params(colors='white')

axes.set_title(
    'Roche Lobe Visualizer + Lagrangian Points',
    color='white'
)

for spine in axes.spines.values():
    spine.set_edgecolor('#444')


#--Potential contours--

axes.contour(
    X,
    Y,
    phi,
    levels=100,
    cmap='plasma'
)


#--Roche Lobe--

axes.contour(
    X,
    Y,
    phi,
    levels=[phi_l1],
    colors='white',
    linestyles='--'
)


#--Stars--

axes.scatter(
    x1,
    y1,
    color='tomato',
    s=100*M1,
    zorder=3,
    label=f'M1={M1}'
)

axes.scatter(
    x2,
    y2,
    color='dodgerblue',
    s=100*M2,
    zorder=3,
    label=f'M2={M2}'
)

axes.legend(
    loc='upper right',
    facecolor='#1a1a1a',
    labelcolor='white',
    fontsize=9
)


#--Marking Lagrangian Points--

plt.scatter(L1, 0, color='cyan', marker='*')
plt.scatter(L2, 0, color='cyan', marker='*')
plt.scatter(L3, 0, color='cyan', marker='*')
plt.scatter(xcom, L4, color='cyan', marker='*')
plt.scatter(xcom, L5, color='cyan', marker='*')


#--Particle Animation--

P = []

XP = np.array(XP)
YP = np.array(YP)

for k in range(N):

    p, = axes.plot([], [], 'o')

    P.append(p)


def animate(i):

    for k in range(N):

        P[k].set_data(
            [XP[i][k]],
            [YP[i][k]]
        )

    return P


ani = FuncAnimation(
    fig,
    animate,
    frames=len(XP),
    interval=10,
    blit=True
)

plt.show()