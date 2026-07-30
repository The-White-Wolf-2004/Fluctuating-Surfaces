import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

x=np.linspace(-1,1,20)
y=np.linspace(-1,1,20)

X,Y=np.meshgrid(x,y)

Z=np.sin(X)*np.sin(Y)

pts=np.column_stack((X.flatten(),Y.flatten()))


pts3d=np.column_stack((X.flatten(),Y.flatten(),Z.flatten()))
tri=Delaunay(pts3d)
faces=tri.simplices
points=tri.points
nbrs=tri.neighbors

'''

fig=plt.figure(figsize=(12,10))
ax=plt.axes(projection='3d')


surface=ax.plot_surface(X,Y,Z)

ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), 
           color='red', marker='.', s=30)



ax.plot_trisurf(pts3d[:,0],pts3d[:,1],pts3d[:,2],triangles=faces,alpha=0.5)
        

pt = pts3d[np.random.randint(len(pts3d)-5)]

ax.scatter(pt[0],pt[1],pt[2],marker="o",color="r")

plt.show()
'''
print(faces)
print(points)
print(nbrs)