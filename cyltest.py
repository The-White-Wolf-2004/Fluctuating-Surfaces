import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numba import njit


def threept_fd1(xm1,x,xp1):
    h2=np.linalg.norm(xp1-x)
    h1=np.linalg.norm(x-xm1)
    d1=h2*xm1/(h1*(h1+h2))
    d2=(h1-h2)*x/(h1*h2)
    d3=h1*xp1/(h2*(h1+h2))
    return -d1-d2+d3
   
def threept_fd2(xm1,x,xp1):
    h2=np.linalg.norm(xp1-x)
    h1=np.linalg.norm(x-xm1)
    return 2*(h2*xm1-(h1+h2)*x+h1*xp1)/(h1*h2*(h1+h2))
    
def fivept_fd1(xm2,xm1,x,xp1,xp2):
    h1=np.linalg.norm(xm1-xm2)
    h2=np.linalg.norm(x-xm1)
    h3=np.linalg.norm(xp1-x)
    h4=np.linalg.norm(xp2-xp1)
    H1=h1+h2+h3
    H2=h1+h2+h3+h4
    d1=h2*h3*(h3+h4)*xm2/(h1*(h1+h2)*H1*H2)
    d2=(h1+h2)*h3*(h3+h4)*xm1/(h1*h2*(h2+h3)*(H2-h1))
    d3=((h1+2*h2)*h3*(h3+h4)-(h1+h2)*h2*(2*h3+h4))*x/((h1+h2)*h2*h3*(h3+h4))
    d4=(h1+h2)*h2*(h3+h4)*xp1/(H1*(h2+h3)*h3*h4)
    d5=(h1+h2)*h2*h3*xp2/(H2*(H2-h1)*(h3+h4)*h4)
    return d1-d2+d3+d4-d5
    
def fivept_fd2(xm2,xm1,x,xp1,xp2):
    h1=np.linalg.norm(xm1-xm2)
    h2=np.linalg.norm(x-xm1)
    h3=np.linalg.norm(xp1-x)
    h4=np.linalg.norm(xp2-xp1)
    H1=h1+h2+h3
    H2=h1+h2+h3+h4
    d1=(-2*h2*(2*h3+h4)+2*h3*(h3+h4))*xm2/(h1*(h1+h2)*(h1+h2+h3)*H2)
    d2=(2*(h1+h2)*(2*h3+h4)-2*h3*(h3+h4))*xm1/(h1*h2*(h2+h3)*(h2+h3+h4))
    d3=(2*h2*(h1+h2)-2*(h1+2*h2)*(2*h3+h4)+2*h3*(h3+h4))*x/((h1+h2)*h2*h3*(h3+h4))
    d4=(2*(h1+2*h2)*(h3+h4)-2*h2*(h1+h2))*xp1/((h1+h2+h3)*(h2+h3)*h3*h4)
    d5=(2*(h1+h2)*h2-2*(h1+2*h2)*h3)*xp2/(H2*(h2+h3+h4)*(h3+h4)*h4)
    return d1+d2+d3+d4+d5

#we will test geodesic curvature at outer circle

def cylinder_mesh(Ny=40,Nz=10,R=0.5,H=1.0):
    theta=np.linspace(0,2*np.pi,Ny,endpoint=False)
    z=np.linspace(-H/2,H/2,Nz)
    THETA, Z=np.meshgrid(theta,z)
    pts=np.vstack([THETA.ravel(),Z.ravel()]).T
    tri=Delaunay(pts)
    faces=tri.simplices
    tripts=tri.points
    nbrs=tri.neighbors

    #we have triangulated set of points for theta, z coordintes
    
    x1=R*np.cos(pts[:,0])
    y1=R*np.sin(pts[:,0])
    z1=pts[:,1]
    
    verts=np.vstack([x1,y1,z1]).T
    
    top=np.max(z1) #top height
    toppts=np.where(np.isclose(z1,top))[0]
    
    angles=pts[toppts,0]
    bdry=toppts[np.argsort(angles)]
    
    return faces,verts,tripts,nbrs,bdry

def skew_cylinder_mesh(Ny=101, Nz=10, R=0.5, H=1.0, dz=0.5):
    theta = np.linspace(0, 2 * np.pi, Ny)
    z = np.linspace(-H/2, H/2, Nz)
    THETA, Z = np.meshgrid(theta, z)
    pts = np.vstack([THETA.ravel(), Z.ravel()]).T
    tri = Delaunay(pts)
    faces = tri.simplices
    tripts = tri.points
    nbrs = tri.neighbors
    
    x1 = R * np.cos(pts[:, 0])
    y1 = R * np.sin(pts[:, 0])
    
    z1 = pts[:, 1] + dz*np.sin(pts[:, 0]) #add some wave to the boundary, to check
    
    verts = np.vstack([x1, y1, z1]).T
    
    n_total = len(verts)
    
    # Build a remapping array: remap[i] = canonical index for vertex i
    remap = np.arange(n_total)
    
    # Find vertices at theta=0 and theta=2pi
    at_zero = np.where(np.isclose(pts[:, 0], 0.0))[0]
    at_2pi  = np.where(np.isclose(pts[:, 0], 2*np.pi))[0]
    
    # Sort both by z so they pair up correctly
    at_zero = at_zero[np.argsort(pts[at_zero, 1])]
    at_2pi  = at_2pi[np.argsort(pts[at_2pi,  1])]
    
    # Map every 2pi vertex to its corresponding 0 vertex
    for v_2pi, v_zero in zip(at_2pi, at_zero):
        remap[v_2pi] = v_zero
    
    # Apply remapping to faces
    faces = remap[faces]
    
    # Remove degenerate faces (where two or more vertices merged to same index)
    mask = np.array([len(set(f)) == 3 for f in faces])
    faces = faces[mask]
    
    # Remove the duplicate vertices from verts
    keep = np.setdiff1d(np.arange(n_total), at_2pi)
    
    # Build a new index map for the kept vertices
    new_index = np.full(n_total, -1)
    new_index[keep] = np.arange(len(keep))
    
    # Remap faces to new indices
    faces = new_index[faces]
    verts = verts[keep]
    pts = pts[keep]
    
    top = np.max(pts[:, 1]) 
    toppts = np.where(np.isclose(pts[:, 1], top))[0]
    
    angles = pts[toppts, 0]
    bdry = toppts[np.argsort(angles)]
    
    return faces, verts, tripts, nbrs, bdry
  
  
def find_nbrs(vertind,faces):
    L=faces.shape[0]
    containing=faces[np.any(faces==vertind,axis=1)]
    vsurr=containing.flatten()
    vsurr=np.unique(vsurr)
    vsurr=vsurr[vsurr!=vertind]
    return vsurr
    
def find_tris(vertind,faces):
    L=faces.shape[0]
    containing=faces[np.any(faces==vertind,axis=1)]
    return containing
    
        
  
def normalvec(vertind,faces,verts):
    triangles=find_tris(vertind,faces)
    normal=np.zeros(3)
    for tri in triangles:
        vind1=tri[0]
        vind2=tri[1]
        vind3=tri[2]
        v1=verts[vind1]
        v2=verts[vind2]
        v3=verts[vind3]
        e1=v2-v1
        e2=v3-v1
        ntri=np.cross(e1,e2)
        ntrinorm=np.linalg.norm(ntri)
        if ntrinorm!=0:
            ntri/=ntrinorm #we get the normalized unit vector
        area=ntrinorm/2
        normal+=area*ntri
    normnormal=np.linalg.norm(normal)
    if normnormal!=0:
        normal/=normnormal #supposedly the unit normal
    return normal
    
    
    
'''      
def tangents(vertind,bdry,faces,verts):
    #vsurr=find_nbrs(vertind,faces)
    #print((bdrynbrs))
    nb=len(bdry)
    #pos=np.where(bdry==vertind)[0][0]
    #print(vertind)
    #print(bdry[(vertind-1)%nb])
    #print(bdry[(vertind+1)%nb])
    pvert=verts[bdry[(vertind-1)%nb]]
    nvert=verts[bdry[(vertind+1)%nb]]
    #print("\n\n")
    #print(bdry[(vertind-1)%nb],vertind,bdry[(vertind+1)%nb])
    vert=verts[vertind]
    t_in=vert-pvert
    t_out=nvert-vert
    normin=np.linalg.norm(t_in)
    normout=np.linalg.norm(t_out)
    t_in/=normin
    t_out/=normout
    #thus we get the unit tangent vectors
    ds=0.5*(normin+normout) #average arclength? is this correct? note.
    #print("\nPoints:",pvert,nvert,vert)
    return t_in, t_out, ds
'''
   
#RETRY WITH PROPER FINITE DIFFERENCE METHODS   
def tangents(vertind,bdry,faces,verts):
    nb=len(bdry)
    pvert=verts[bdry[(vertind-1)%nb]]
    nvert=verts[bdry[(vertind+1)%nb]]
    pvert2=verts[bdry[(vertind-2)%nb]]
    nvert2=verts[bdry[(vertind+2)%nb]]
    
    vert=verts[vertind]
    
    t_in=vert-pvert
    t_out=nvert-vert
    t_in2=pvert-pvert2
    t_out2=nvert2-nvert
    
    
    #magnitudes
    normin=np.linalg.norm(t_in)
    normout=np.linalg.norm(t_out)
    normin2=np.linalg.norm(t_in2)
    normout2=np.linalg.norm(t_out2)

    h = (normin + normout + normin2 + normout2) / 4.0
    tangent = (-nvert2 + 8*nvert - 8*pvert + pvert2) / (12 * h)
    dtds = (-nvert2 + 16*nvert - 30*vert + 16*pvert - pvert2) / (12 * h**2)
    
    ds=0.5*(normin+normout) #average arclength? is this correct? note.


    return t_in, t_out, t_in2, t_out2, tangent, dtds, ds
'''   
def bdrylocalgc(vertind):
    global bdry,faces,verts
    t_in,t_out,ds=tangents(vertind,bdry,faces,verts)
    normal=normalvec(vertind,faces,verts)
    kappa=(t_out-t_in)/ds #curvature vector
    tangent=t_in+t_out
    tangent/=np.linalg.norm(tangent)#also unit vector
    #print("\nTangent=",tangent,"\nNormal=",normal,"\nKappa=",kappa,"\nds=",ds)
    temp=np.cross(normal,tangent)
    #print("Cross product=",temp)
    kg=np.dot(kappa,temp) #supposedly gaussian curvature
    return kg
'''

#WITH NEW TANGENTS FUNCTION
def bdrylocalgc(vertind):
    global bdry,faces,verts
    t_in,t_out,t_in2, t_out2, tangent, kappa,ds=tangents(vertind,bdry,faces,verts)
    normal=normalvec(vertind,faces,verts)
    tangent/=np.linalg.norm(tangent)#also unit vector
    #print("\nTangent=",tangent,"\nNormal=",normal,"\nKappa=",kappa,"\nds=",ds)
    temp=np.cross(normal,tangent)
    #print("Cross product=",temp)
    kg=np.dot(kappa,temp) #supposedly gaussian curvature
    return kg,ds



def normal_midpt(vertind1,vertind2,bdry,faces,verts):
    tri1=find_tris(vertind1,faces)
    tri2=find_tris(vertind2,faces)
    tris=[]
    #print("TRIANGLES 1:",tri1)
    #print("TRIANGLES 2:",tri2)
    for tri in tri1:
        matches = np.all(tri2 == tri, axis=1)
        if np.any(matches):
            # Extract the matching face configuration row
            matching_tri = tri2[matches][0]
            tris.append(matching_tri)
    #we need to do only one of these because we are on the boundary and only one unique tri belongs to 2 given pts
    #print(len(tris))
    tri=tris[0]
    normal=np.zeros(3)
    vind1=tri[0]
    vind2=tri[1]
    vind3=tri[2]
    v1=verts[vind1]
    v2=verts[vind2]
    v3=verts[vind3]
    e1=v2-v1
    e2=v3-v1
    ntri=np.cross(e1,e2)
    
    midpt = (verts[vertind1] + verts[vertind2]) / 2
    radial = midpt.copy()
    radial[2] = 0  # project onto xy plane
    radial_norm = np.linalg.norm(radial)
    if radial_norm > 1e-12:
        radial /= radial_norm
    
    # If normal points inward, flip it
    if np.dot(ntri, radial) < 0:
        ntri = -ntri
        
    ntrinorm=np.linalg.norm(ntri)
    if ntrinorm!=0:
        ntri/=ntrinorm #we get the normalized unit vector
        

    return ntri
  
'''  
def tangents_midpt(vertind1,vertind2,bdry,faces,verts):
    nb=len(bdry)
    
    vert1=verts[vertind1]
    vert2=verts[vertind2]
    
    pvert1=verts[bdry[(vertind1-1)%nb]]
    nvert2=verts[bdry[(vertind2+1)%nb]]
    
    ppvert1=verts[bdry[(vertind1-2)%nb]]
    nnvert2=verts[bdry[(vertind2+2)%nb]]
    
    midpt=(vert1+vert2)/2
    
    midpt1=(pvert1+vert1)/2
    midpt2=(vert2+nvert2)/2
    
    midpt11=(pvert1+ppvert1)/2
    midpt22=(nvert2+nnvert2)/2
    
    tan=vert2-vert1
    h=np.linalg.norm(tan)
    
    tan2=nvert2-vert2
    tan2/=np.linalg.norm(tan2)
    tan1=vert1-pvert1
    tan1/=np.linalg.norm(tan1)
    
    tan22=nnvert2-vert2
    tan11=pvert1-ppvert1
    tan11/=np.linalg.norm(tan11)
    tan22/=np.linalg.norm(tan22)
    
    
    #dtds=(tan2-tan1)/(np.linalg.norm(midpt2-vert2)+np.linalg.norm(vert1-midpt1)+np.linalg.norm(tan))
    
    dtds=(-tan22+8*tan2-8*tan1+tan11)/(12*h)
    
    tan/=np.linalg.norm(tan)
    
    return tan,dtds
'''

def tangents_midpt(vertind1,vertind2,bdry,faces,verts):
    nb=len(bdry)
    
    vert1=verts[vertind1]
    vert2=verts[vertind2]
    
    pvert1=verts[bdry[(vertind1-1)%nb]]
    nvert1=verts[bdry[(vertind1+1)%nb]]
    ppvert1=verts[bdry[(vertind1-2)%nb]]
    nnvert1=verts[bdry[(vertind1+2)%nb]]
    
    pvert2=verts[bdry[(vertind2-1)%nb]]
    nvert2=verts[bdry[(vertind2+1)%nb]]
    ppvert2=verts[bdry[(vertind2-2)%nb]]
    nnvert2=verts[bdry[(vertind2+2)%nb]]
    
    tan3_1=threept_fd1(pvert1,vert1,nvert1)
    tan3_2=threept_fd1(pvert2,vert2,nvert2)
    dtds3_1=threept_fd2(pvert1,vert1,nvert1)
    dtds3_2=threept_fd2(pvert2,vert2,nvert2)
    
    tan3=0.5*(tan3_1+tan3_2)
    dtds3=0.5*(dtds3_1+dtds3_2)
    
        
    tan5_1=fivept_fd1(ppvert1,pvert1,vert1,nvert1,nnvert1)
    tan5_2=fivept_fd1(ppvert2,pvert2,vert2,nvert2,nnvert2)
    dtds5_1=fivept_fd2(ppvert1,pvert1,vert1,nvert1,nnvert1)
    dtds5_2=fivept_fd2(ppvert2,pvert2,vert2,nvert2,nnvert2)
    
    tan5=0.5*(tan5_1+tan5_2)
    dtds5=0.5*(dtds5_1+dtds5_2)
    
    tan=tan5
    dtds=dtds5
   
    

  
    
    tan/=np.linalg.norm(tan)
    
    return tan,dtds
    
    
def bdrylocalgc_midpt(vertind):
    global bdry,faces,verts
    nb=len(bdry)
   
    pvertind=bdry[(vertind-1)%nb]
    nvertind=bdry[(vertind+1)%nb]
    
    vert=verts[vertind]
    pvert=verts[pvertind]
    nvert=verts[nvertind]
    
    norm1=normal_midpt(pvertind,vertind,bdry,faces,verts)
    tan1,kappa1=tangents_midpt(pvertind,vertind,bdry,faces,verts)
    temp1=np.cross(norm1,tan1)
    gc1=np.dot(kappa1,temp1)
    
    norm2=normal_midpt(vertind,nvertind,bdry,faces,verts)
    tan2,kappa2=tangents_midpt(vertind,nvertind,bdry,faces,verts)
    temp2=np.cross(norm2,tan2)
    gc2=np.dot(kappa2,temp2)
    
    ds=np.linalg.norm(vert-pvert)/2+np.linalg.norm(nvert-vert)/2
    
    return (gc1*np.linalg.norm(vert-pvert)+gc2*np.linalg.norm(nvert-vert))/(np.linalg.norm(vert-pvert)+np.linalg.norm(nvert-vert)),ds

def bdrylocalgc2_midpt(vertind):
    global bdry,faces,verts
    nb=len(bdry)
   
    nvertind=bdry[(vertind+1)%nb]
    
    vert=verts[vertind]
    nvert=verts[nvertind]
   
    
    norm2=normal_midpt(vertind,nvertind,bdry,faces,verts)
    tan2,kappa2=tangents_midpt(vertind,nvertind,bdry,faces,verts)
    temp2=np.cross(norm2,tan2)
    gc=np.dot(kappa2,temp2)
    
    #print(np.linalg.norm(nvert-vert))
    
    return gc,np.linalg.norm(nvert-vert)

    
    

faces,verts,tripts,nbrs,bdry=skew_cylinder_mesh()

print("Points:")
print(tripts)
print("Faces:")
print(faces)
print("Neighbors:")
print(nbrs)
print("Boundary:")
print(bdry)
  
fig=plt.figure(figsize=(12,10))
ax=plt.axes(projection='3d')

ax.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=faces,color="orange",edgecolor="brown",linewidth=0.3,alpha=0.6)
vertind=np.random.choice(bdry)
pt1=verts[vertind]
print("Random vertex:",vertind)
vsurr=find_nbrs(vertind,faces)
ax.scatter(pt1[0],pt1[1],pt1[2],color="red")

#print("TESTING TANGENTS")
#tangents(vertind)

for ind in vsurr:
    pt=verts[ind]
    ax.scatter(pt[0],pt[1],pt[2],color="blue")
    

plt.show()


print("Iterating through boundary:")
gc=0
gcs=[]
for ind in bdry: #take a look at all boundary points
    ptgc,ds=bdrylocalgc_midpt(ind)
    gcs.append(ptgc)
    print("Local geodesic curvature at point=",ptgc)
    gc+=ds*ptgc

bdryarray=(bdry-bdry[0])
plt.figure()
plt.plot(bdryarray,gcs,marker="o")
plt.title("Geodesic curvature at boundary points")
plt.show()

print("\n\nTotal geodesic curvature integrated over boundary =",gc)   

#testing skewed cylinder
thetas=np.linspace(0,2*np.pi,100,endpoint=False)
R=0.5
func=-np.sin(thetas)/(R*((1+np.cos(thetas)*np.cos(thetas))**1.5))
#func=-4*np.sin(2*thetas)/(R*((1+4*np.cos(2*thetas)*np.cos(2*thetas))**1.5))
plt.figure()
plt.plot(thetas,func,marker="o",color="red")
plt.title("Plot of -sint/(R(1+cos^2t)) over angular coordinate")
plt.show()

plt.figure()

plt.plot(thetas,gcs,marker="o")
plt.plot(thetas,func,marker="o",color="red")
plt.title("Comparative plot of calculated and analytical geodesic curvature")
plt.show()

#Nys=np.asarray([10,20,40,100,500,1000])
Nys=np.asarray([8,16,32,64,128,256,512,1024])
mses=np.zeros_like(Nys,dtype=np.float64)


count=0
for Ny in Nys:
    
    print(Ny)
    
    faces,verts,tripts,nbrs,bdry=skew_cylinder_mesh(Ny+1)

    gcs=[]
    for ind in bdry: #take a look at all boundary points
        ptgc,_=bdrylocalgc_midpt(ind)
        gcs.append(ptgc)

    gcs=np.asarray(gcs)
    
    thetas=np.linspace(0,2*np.pi,Ny)
    R=0.5
    func=-np.sin(thetas)/(R*((1+np.cos(thetas)*np.cos(thetas))**1.5))
    #func=-4*np.sin(2*thetas)/(R*((1+4*np.cos(2*thetas)*np.cos(2*thetas))**1.5))
    
    errors=np.abs(func-gcs)
    
    mse=np.dot(errors,errors)/(Ny-1)
    
    print("MSQERROR AT NY =",Ny,"IS",mse)
    
    print(count)
    mses[count]=mse
    
    count+=1

print(mses)
    
'''
plt.figure(figsize=(12,10))
plt.loglog(Nys,mses,marker="o")
plt.title("MSE with varying Ny")
plt.show()

# Convert your arrays to log scale for the convergence plot
log_Nys = np.log10(Nys)
log_mses = np.log10(mses)

plt.figure(figsize=(10, 8))
plt.plot(log_Nys, log_mses, marker="o", linestyle='-', color='blue', label='Numerical Error')

# Plot a reference slope for 4th-order theoretical convergence O(h^4)
# (Adjust the vertical intercept -2 to align visually with your data)
theoretical_slope = -4 * log_Nys + (log_mses[0] + 4 * log_Nys[0]) 
plt.plot(log_Nys, theoretical_slope, linestyle='--', color='red', label='Theoretical O(h^4) Slope')

plt.title("Log-Log Convergence of Geodesic Curvature Error")
plt.xlabel("Log10(Number of Grid Points Ny)")
plt.ylabel("Log10(Mean Squared Error)")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
'''

   

