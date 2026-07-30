import numpy as np
import matplotlib.pyplot as plt

def circle(R=1):
    thetas=np.linspace(0,2*np.pi,100,endpoint=False)
    xs=R*np.cos(thetas)
    ys=R*np.sin(thetas)
    verts=np.vstack([xs,ys]).T
    print(verts[0])
    return verts,thetas
    
verts,thetas=circle(2)

plt.figure()
plt.plot(verts[:,0],verts[:,1],marker="o")
plt.axis('equal')
plt.show()

ind=24
a=verts[ind]
b=verts[ind+1]


mpt=(a+b)/2
mptp1=(verts[ind+2]+b)/2
mptm1=(verts[ind-1]+a)/2
mptp11=(verts[ind+3]+verts[ind+2])/2
mptm11=(verts[ind-1]+verts[ind-2])/2


t1=verts[ind+2]-b
t1/=np.linalg.norm(t1)
t2=a-verts[ind-1]
t2/=np.linalg.norm(t2)

t11=verts[ind+3]-verts[ind+2]
t22=verts[ind-1]-verts[ind-2]
t11/=np.linalg.norm(t11)
t22/=np.linalg.norm(t22)

t=b-a
t/=np.linalg.norm(t)
print("Tangent=",t)
'''
dtds=(-t11+8*t1+8*t2-t22)/(np.linalg.norm(mptp11-verts[ind+2])+np.linalg.norm(verts[ind+2]-b)+np.linalg.norm(verts[ind-1]-mptm11)+np.linalg.norm(a-verts[ind-1])+np.linalg.norm(b-a))
print(np.linalg.norm(dtds))
'''

h = np.linalg.norm(b - a)
d2t_ds2_fo=(t2-t1)/(2*h)
d2t_ds2_so = (-t11 + 8*t1 - 8*t2 + t22) / (12 * (h))
print("At midpoint, three point central diff:",np.linalg.norm(d2t_ds2_fo))
print("At midpoint, five point central diff:",np.linalg.norm(d2t_ds2_so))

dtdsvert=(verts[ind+2]+a-2*b)/(0.5*(np.linalg.norm(verts[ind+2]-b)**2+(np.linalg.norm(b-a)**2)))
print("At vertex,three point central diff:",np.linalg.norm(dtdsvert))