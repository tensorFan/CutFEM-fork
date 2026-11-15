#!/usr/bin/env python3
"""Extract P1 level sets on the actual tetrahedra; audit topology and plot.

A small positive r-level is an offset proxy, NOT the exact continuum free
boundary. --active uses the midpoint of the nodal contact indicator instead.
Dependencies: NumPy and Matplotlib (no voxel resampling or mesh smoothing).
"""
import argparse
import csv
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np


def extract(points, cells, values, level, field=None):
    vertices, triangles, edge_ids = [], [], {}
    samples = []
    if np.any(values == level):
        raise ValueError('Level equals nodal data; choose a nearby nondegenerate level')
    for tet in cells:
        low = [int(i) for i in tet if values[i] < level]
        high = [int(i) for i in tet if values[i] >= level]
        if not low or not high:
            continue
        polygon = []
        for i in low:
            for j in high:
                edge = tuple(sorted((i, j)))
                if edge not in edge_ids:
                    edge_ids[edge] = len(vertices)
                    t = (level-values[i])/(values[j]-values[i])
                    vertices.append((1-t)*points[i]+t*points[j])
                    if field is not None:
                        samples.append((1-t)*field[i]+t*field[j])
                polygon.append(edge_ids[edge])
        p = np.array([vertices[i] for i in polygon])
        center = p.mean(axis=0)
        normal = np.cross(p[1]-p[0], p[2]-p[0])
        normal /= np.linalg.norm(normal)
        direction = points[high].mean(axis=0)-points[low].mean(axis=0)
        if normal @ direction < 0:
            normal = -normal
        u = p[0]-center
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        angles = np.arctan2((p-center) @ v, (p-center) @ u)
        polygon = [polygon[i] for i in np.argsort(angles)]
        for j in range(1, len(polygon)-1):
            triangles.append([polygon[0], polygon[j], polygon[j+1]])
    result = (np.asarray(vertices).reshape(-1, 3), np.asarray(triangles, dtype=int).reshape(-1, 3))
    return (*result, np.asarray(samples)) if field is not None else result


def topology(vertices, triangles):
    adjacency = defaultdict(set)
    edge_counts = Counter()
    links = defaultdict(list)
    for a, b, c in triangles:
        for u, v in [(a,b), (b,c), (c,a)]:
            adjacency[u].add(v)
            adjacency[v].add(u)
            edge_counts[tuple(sorted((u,v)))] += 1
        links[a].append((b,c)); links[b].append((c,a)); links[c].append((a,b))
    bad_vertices = set()
    for node, pairs in links.items():
        graph = defaultdict(set)
        for u,v in pairs:
            graph[u].add(v); graph[v].add(u)
        seen, todo = set(), [next(iter(graph))]
        while todo:
            u = todo.pop()
            if u not in seen:
                seen.add(u); todo.extend(graph[u]-seen)
        if len(seen)!=len(graph) or any(len(a)!=2 for a in graph.values()):
            bad_vertices.add(node)
    components, unseen = [], set(adjacency)
    while unseen:
        component, todo = set(), [next(iter(unseen))]
        while todo:
            u = todo.pop()
            if u not in component:
                component.add(u); todo.extend(adjacency[u]-component)
        unseen -= component
        faces = np.array([f for f in triangles if f[0] in component])
        edges = {e:n for e,n in edge_counts.items() if e[0] in component}
        chi = len(component)-len(edges)+len(faces)
        closed = all(n==2 for n in edges.values()) and not (component & bad_vertices)
        p = vertices[faces]
        area = .5*np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1).sum()
        volume = np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6
        components.append(dict(vertices=len(component), edges=len(edges), faces=len(faces),
            euler_characteristic=chi, closed_manifold=closed,
            genus=(2-chi)//2 if closed and chi%2==0 else None,
            area=float(area), enclosed_signed_volume=float(volume)))
    return sorted(components, key=lambda c:-c['area'])


def save_vtk(path, vertices, triangles):
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nP1 level-set proxy\nASCII\nDATASET POLYDATA\n')
        f.write(f'POINTS {len(vertices)} double\n')
        np.savetxt(f, vertices, fmt='%.17g')
        f.write(f'POLYGONS {len(triangles)} {4*len(triangles)}\n')
        np.savetxt(f, np.column_stack((np.full(len(triangles),3),triangles)), fmt='%d')


def plot(path, points, cells, r, vertices, triangles, components, level, label, active=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(13,9), facecolor='#fafafa')
    for panel, (elevation, azimuth) in enumerate([(28,-58),(68,18)]):
        ax = fig.add_subplot(2,2,panel+1,projection='3d')
        faces = vertices[triangles]
        normal = np.cross(faces[:,1]-faces[:,0], faces[:,2]-faces[:,0])
        normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-30)
        light = np.abs(normal @ np.array([.25,-.45,.857]))
        colors = plt.get_cmap('viridis')(.25+.65*(faces[:,:,2].mean(axis=1)-points[:,2].min())/np.ptp(points[:,2]))
        colors[:,:3] *= (.55+.45*light[:,None])
        shown=np.ones(len(faces),dtype=bool) if panel==0 else faces[:,:,1].mean(axis=1)<.15
        ax.add_collection3d(Poly3DCollection(faces[shown],facecolors=colors[shown],linewidths=0,rasterized=True))
        for setter, d in [(ax.set_xlim,0),(ax.set_ylim,1),(ax.set_zlim,2)]:
            setter(points[:,d].min(),points[:,d].max())
        ax.set_box_aspect(np.ptp(points,axis=0)); ax.view_init(elevation,azimuth)
        ax.set(xlabel='source x',ylabel='source y',zlabel='source z')
        ax.set_title('Computed contact-boundary proxy' if panel==0 else 'Cutaway: triangles with mean y < 0.15')
    for panel, requested in enumerate([0.,.5]):
        # A tiny offset avoids the exact-node degeneracy of a plane aligned
        # with the mesh. The same P1 interpolation supplies r on the section.
        section,tris,section_r=extract(points,cells,points[:,2],requested+1e-8,field=r)
        ax=fig.add_subplot(2,2,panel+3)
        if len(tris):
            im=ax.tricontourf(section[:,0],section[:,1],tris,section_r,levels=np.linspace(0,max(r.max(),.01),25),cmap='magma')
            if active:
                _,_,contour=extract(points,cells,points[:,2],requested+1e-8,field=(r>1e-7).astype(float))
            else:
                contour=section_r
            contour_level=.5 if active else level
            if contour.min()<contour_level<contour.max():
                ax.tricontour(section[:,0],section[:,1],tris,contour,levels=[contour_level],colors=['cyan'],linewidths=1.)
            fig.colorbar(im,ax=ax,label='normal distance r')
        ax.set(aspect='equal',xlabel='source x',ylabel='source y',title=f'Source section z≈{requested:g}')
    genera=[c['genus'] for c in components]
    fig.suptitle(f'{label}: {len(components)} components, genera {genera}\nFinite-element level-set topology; no continuum or novelty claim',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94),h_pad=3.0); fig.savefig(path,dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefix',type=Path)
    parser.add_argument('--levels',type=float,nargs='+',default=[1e-5,1e-4,1e-3])
    parser.add_argument('--active',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    with open(str(args.prefix)+'_summary.csv') as f:
        summary=next(csv.DictReader(f))
    if summary['status']!='CONVERGED':
        raise ValueError('Refusing to label an unconverged iterate as a candidate surface')
    if not args.levels or any(not np.isfinite(v) or v<=0 for v in args.levels):
        raise ValueError('Use finite positive levels for normal-distance proxies')
    data=np.genfromtxt(str(args.prefix)+'_nodes.csv',delimiter=',',names=True)
    points=np.column_stack([data[a] for a in ['x','y','z']])
    cells=np.loadtxt(str(args.prefix)+'_cells.csv',delimiter=',',skiprows=1,dtype=int).reshape(-1,4)
    records=[]
    for level in args.levels:
        vertices,triangles=extract(points,cells,data['r'],level)
        records.append(dict(level=level,components=topology(vertices,triangles)))
    if args.active:
        vertices,triangles=extract(points,cells,(data['r']>1e-7).astype(float),.5)
        records.append(dict(level='active_midpoint',components=topology(vertices,triangles)))
    with open(str(args.prefix)+'_topology.json','w') as f: json.dump(records,f,indent=2)
    for record in records:
        print(json.dumps(record))
    # Save the final selected proxy. Plot labels always identify it explicitly.
    save_vtk(str(args.prefix)+'_surface.vtk',vertices,triangles)
    if args.plot and len(triangles):
        label=f'r={args.levels[-1]:g}' if not args.active else 'Nodal-contact midpoint'
        plot(str(args.prefix)+'_surface.png',points,cells,data['r'],vertices,triangles,
             records[-1]['components'],args.levels[-1],label,args.active)


if __name__=='__main__': main()
