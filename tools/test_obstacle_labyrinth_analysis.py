"""Independent analytic sphere/torus fixtures for the topology extractor."""
import numpy as np
from analyze_obstacle_labyrinth import extract, topology

n=18
axis=np.linspace(-1,1,n+1)
points=np.array([(x,y,z) for x in axis for y in axis for z in axis])
index=lambda i,j,k: (i*(n+1)+j)*(n+1)+k
cells=[]
# Six tetrahedra about the cube body diagonal, consistent on adjacent faces.
for i in range(n):
    for j in range(n):
        for k in range(n):
            v=[index(i+a,j+b,k+c) for a,b,c in [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]]
            cells.extend([[v[a] for a in t] for t in [(0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6)]])
cells=np.array(cells)
x,y,z=points.T
for values,expected in [(x*x+y*y+z*z-.63**2,0),((np.sqrt(x*x+y*y)-.53)**2+z*z-.213**2,1)]:
    v,f=extract(points,cells,values,0.)
    components=topology(v,f)
    assert len(components)==1 and components[0]['closed_manifold']
    assert components[0]['genus']==expected, components
    assert components[0]['enclosed_signed_volume']>0
    print(f'Analytic genus-{expected} fixture passed: {len(f)} triangles')

v,f,values=extract(points,cells,z,1e-6,field=x+2*y+3*z)
assert np.max(np.abs(values-(v[:,0]+2*v[:,1]+3*v[:,2])))<1e-13
print('Section interpolation of an affine field passed')
