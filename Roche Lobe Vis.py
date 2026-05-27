import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
G=1
M1=1
M2=1

x1=1
y1=0

x2=-1
y2=0

xp=0.4
yp=1

vxp=-0.5
vyp=0

t=0

XP=[]
YP=[]


r=np.sqrt((x2-x1)**2+(y2-y1)**2)
w=np.sqrt((G*(M1+M2))/(r**3))

dt=0.01

while t<20:
    
    rp1=np.sqrt((xp-x1)**2+(yp-y1)**2)
    rp2=np.sqrt((xp-x2)**2+(yp-y2)**2)
    if rp1<0.001 or rp2<0.001:
        break

    

    axp=-(G*M1*(xp-x1))/rp1**3-(G*M2*(xp-x2))/rp2**3+(2*w*vyp)+(w**2*xp)
    ayp=-(G*M1*(yp-y1))/rp1**3-(G*M2*(yp-y2))/rp2**3-(2*w*vxp)+(w**2*yp)


    vxp=vxp+0.5*axp*dt
    vyp=vyp+0.5*ayp*dt
    

    xp=xp+vxp*dt
    yp=yp+vyp*dt

    rp1=np.sqrt((xp-x1)**2+(yp-y1)**2)
    rp2=np.sqrt((xp-x2)**2+(yp-y2)**2)
    if rp1<0.001 or rp2<0.001:
        break
    

    axp=-(G*M1*(xp-x1))/rp1**3-(G*M2*(xp-x2))/rp2**3+(2*w*vyp)+(w**2*xp)
    ayp=-(G*M1*(yp-y1))/rp1**3-(G*M2*(yp-y2))/rp2**3-(2*w*vxp)+(w**2*yp)

    vxp=vxp+0.5*axp*dt
    vyp=vyp+0.5*ayp*dt

    t+=dt


    XP.append(xp)
    YP.append(yp)


xcom=(M1*x1+M2*x2)/(M1+M2)
ycom=(M1*y1+M2*y2)/(M1+M2)

x=np.linspace(-5,5,1000)
y=np.linspace(-5,5,1000)
X,Y=np.meshgrid(x,y)
r1=np.sqrt((X-x1)**2+(Y-y1)**2+0.001)
r2=np.sqrt((X-x2)**2+(Y-y2)**2+0.001)

phi_g=-((G*M1)/r1)-((G*M2)/r2)
phi_f=-0.5*w**2*((X-xcom)**2 + (Y-ycom)**2)

phi=phi_g+phi_f
dphidy,dphidx=np.gradient(phi)
gradphi=np.sqrt(dphidy**2+dphidx**2)
mask1=gradphi<0.0005

LX=X[mask1]
LY=Y[mask1]

for i in range(len(LX)):
         if min(x1,x2)<LX[i]<max(x1,x2) and np.abs(LY[i])<0.01:
             phi_l1=phi[mask1][i]



fig,axes=plt.subplots(figsize=(10,10))
axes.set_xlim(-5,5)
axes.set_ylim(-5,5)
axes.scatter(x1,y1,color='r')
axes.scatter(x2,y2,color='b')
axes.contour(X,Y,phi,levels=500)
axes.contour(X,Y,phi,levels=[phi_l1])
plt.scatter(LX,LY,color='cyan')

particle,=axes.plot([],[],'o',color='black')

def animate(i):
    particle.set_data([XP[i]],[YP[i]])
    return particle,

ani=FuncAnimation(fig,animate,frames=len(XP),interval=10)
plt.show()





