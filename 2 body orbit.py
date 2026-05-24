import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
G=1
w=1
M1=1
M2=1

x1=1
y1=0

x2=-1
y2=0

xp=0.5
yp=0.5

vx1=0
vy1=-0.5

vx2=0
vy2=0.5

vxp=0
vyp=0

Xcm=[]
Ycm=[]

t=0
X1=[]
X2=[]
Y1=[]
Y2=[]
XP=[]
YP=[]
E=[]
T=[]

dt=0.01

while t<20:
    
    r=np.sqrt((x2-x1)**2+(y2-y1)**2)
    if r<0.001:
        break
    
    ax1=(G*M2*(x2-x1))/r**3
    ay1=(G*M2*(y2-y1))/r**3
    

    ax2=(G*M1*(x1-x2))/r**3
    ay2=(G*M1*(y1-y2))/r**3


    vx1=vx1+ax1*dt
    vy1=vy1+ay1*dt

    vx2=vx2+ax2*dt
    vy2=vy2+ay2*dt

    x1=x1+vx1*dt
    y1=y1+vy1*dt

    x2=x2+vx2*dt
    y2=y2+vy2*dt

    t+=dt

    xcm=(M1*x1+M2*x2)/(M1+M2)
    ycm=(M1*y1+M2*y2)/(M1+M2)

    KE=0.5*((M1*vx1**2)+(M1*vy1**2)+(M2*vx2**2)+(M2*vy2**2))
    U=-(G*M1*M2)/r
    e=KE+U

    X1.append(x1)
    X2.append(x2)
    Y1.append(y1)
    Y2.append(y2)
    E.append(e)
    T.append(t)
    Xcm.append(xcm)
    Ycm.append(ycm)

fig, axes=plt.subplots()
axes.set_xlim(min(X1+X2)-1,max(X1+X2)+1)
axes.set_ylim(min(Y1+Y2)-1,max(Y1+Y2)+1)

planet1,=axes.plot([],[],'o',color='r')
planet2,=axes.plot([],[],'o')

trail1,=axes.plot([],[],color='blue')
trail2,=axes.plot([],[])

cm,=axes.plot([],[],'o',color='green')


def animate(i):
    planet1.set_data([X1[i]],[Y1[i]])
    trail1.set_data([X1[:i]],[Y1[:i]])

    planet2.set_data([X2[i]],[Y2[i]])
    trail2.set_data([X2[:i]],[Y2[:i]])

    cm.set_data([Xcm[i]],[Ycm[i]])

    return planet1,planet2,trail2,trail1

ani=FuncAnimation(fig,animate,frames=len(X1),interval=10)
plt.grid(True)
plt.show()

plt.plot(T,E)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.grid(True)
plt.show()





