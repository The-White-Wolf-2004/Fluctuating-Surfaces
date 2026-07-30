import numpy as np
import matplotlib.pyplot as plt

R=10
u=np.linspace(0,4*np.pi,50)
w=6.0
v=np.linspace(-w/2,w/2,50)
U,V=np.meshgrid(u,v)

t=(R+V*np.cos(U/2))
X=t*np.cos(U)
Y=t*np.sin(U)
Z=V*np.sin(U/2)
   
t2=(R+0.5*w*np.cos(u/2))
x=t2*np.cos(u)
y=t2*np.sin(u)
z=0.5*w*np.sin(u/2)   

fig=plt.figure(figsize=(15,12))
ax=plt.axes(projection='3d')

#surface=ax.plot_surface(X,Y,Z)
line=ax.plot(x,y,z)
plt.tight_layout()

plt.show()



