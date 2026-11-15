#!/usr/bin/env python3
"""Audit and plot the hyperbolic horosphere obstacle-map experiment.

Usage: python tools/plot_obstacle_horosphere.py RUN_PREFIX [--pdf]

Dependencies: NumPy and Matplotlib.  Inputs are RUN_PREFIX_{nodes,cells,
profile,summary}.csv.  The primary surface is the boundary of the union of
fully contacting tetrahedra, when its manifold audit passes; otherwise a
positive P1 level is used.  There is no voxel resampling or smoothing.
Positive r-levels and the midpoint
of a nodal contact indicator are finite-element proxies, not the exact free
boundary.  The reference free boundary is the source sphere rho=a; target
coordinates (s,r) are horospherical coordinates, not a Euclidean embedding.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_obstacle_labyrinth import extract, save_vtk, topology


TEAL = "#237c80"
GOLD = "#b78328"
INK = "#26363a"
GRAY = "#6b777b"


def read_csv(path, fields):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    missing = set(fields) - set(data.dtype.names or ())
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not len(data) or any(not np.isfinite(data[name]).all() for name in fields):
        raise ValueError(f"{path}: empty or nonfinite data")
    return data


def radial_audit(vertices, triangles, reference_radius):
    """Integrate radius errors with the standard three-point triangle rule.

    The rule is exact for polynomials through degree two; radial norms are
    generally nonpolynomial, so these are quadrature estimates.  Vertex-only
    errors would miss the inward displacement inside an inscribed triangle.
    """
    if not len(triangles):
        return {"vertices": 0, "triangles": 0}
    faces = vertices[triangles]
    areas = .5 * np.linalg.norm(
        np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1)
    radii = np.linalg.norm(vertices, axis=1)
    errors = radii - reference_radius
    barycentric = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    samples = np.einsum("qj,tjk->tqk", barycentric, faces)
    sample_radii = np.linalg.norm(samples, axis=2)
    sample_errors = sample_radii - reference_radius
    mean_radius = np.average(sample_radii.mean(axis=1), weights=areas)
    mean_error = np.average(sample_errors.mean(axis=1), weights=areas)
    rms_error = np.sqrt(np.average((sample_errors**2).mean(axis=1), weights=areas))
    return dict(
        vertices=len(vertices), triangles=len(triangles), area=float(areas.sum()),
        radius_min=float(radii.min()), radius_max=float(radii.max()),
        area_weighted_mean_radius=float(mean_radius),
        reference_radius=float(reference_radius),
        area_weighted_mean_radius_error=float(mean_error),
        area_weighted_rms_radius_error=float(rms_error),
        vertex_maximum_absolute_radius_error=float(np.abs(errors).max()),
        quadrature_maximum_absolute_radius_error=float(np.abs(sample_errors).max()),
        maximum_absolute_radius_error=float(max(np.abs(errors).max(), np.abs(sample_errors).max())),
        definitions=("radius_min and radius_max are vertex extrema; integrated means and RMS "
            "use three equal-weight barycentric points (2/3,1/6,1/6) and permutations "
            "on every triangle, weighted by triangle area. The maximum is over vertices "
            "and quadrature points, not a certified continuous maximum."),
    )


def contact_volume_boundary(points, cells, r, threshold):
    """Return oriented exterior faces of the fully contacting tetrahedron union."""
    contacting = cells[np.all(r[cells] <= threshold, axis=1)]
    faces = {}
    multiplicities = {}
    for tet in contacting:
        for opposite in range(4):
            face = [int(tet[i]) for i in range(4) if i != opposite]
            key = tuple(sorted(face))
            multiplicities[key] = multiplicities.get(key, 0) + 1
            if multiplicities[key] > 2:
                raise ValueError("More than two contacting tetrahedra share a face")
            if key in faces:
                del faces[key]
                continue
            a, b, c = points[face]
            if np.dot(np.cross(b-a, c-a), points[tet[opposite]]-a) > 0:
                face[1], face[2] = face[2], face[1]
            faces[key] = face
    indices = np.asarray(list(faces.values()), dtype=int).reshape(-1, 3)
    original_nodes, compact = np.unique(indices, return_inverse=True)
    vertices = points[original_nodes]
    triangles = compact.reshape(-1, 3)
    tetrahedra = points[contacting]
    volume = np.abs(np.einsum("ij,ij->i", tetrahedra[:, 1]-tetrahedra[:, 0],
        np.cross(tetrahedra[:, 2]-tetrahedra[:, 0],
                 tetrahedra[:, 3]-tetrahedra[:, 0]))).sum()/6
    return vertices, triangles, len(contacting), float(volume)


def surface_section(vertices, triangles, z=1e-10):
    """Intersect the displayed triangle surface with a source equatorial plane."""
    segments = []
    for triangle in vertices[triangles]:
        intersections = []
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            p, q = triangle[i], triangle[j]
            if (p[2] < z) != (q[2] < z):
                t = (z-p[2])/(q[2]-p[2])
                intersections.append(((1-t)*p+t*q)[:2])
        if len(intersections) == 2:
            segments.append(intersections)
    return np.asarray(segments).reshape(-1, 2, 2)


def reference_level_radius(profile, level, contact_radius):
    # The radial reference has R=0 in contact and R strictly increasing outside.
    outside = profile["rho"] >= contact_radius
    rho, r = profile["rho"][outside], profile["r"][outside]
    if len(r) < 2 or level > r.max() or np.any(np.diff(r) < -1e-12):
        return None
    return float(np.interp(level, np.r_[0., r], np.r_[contact_radius, rho]))


def make_figure(path, points, cells, nodes, profile, summary, vertices,
                triangles, selected_level, contact_radius, pdf=False,
                solid_contact=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.edgecolor": "#c7cfd0", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": GRAY, "ytick.color": GRAY,
        "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(12.5, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.])
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    axsection = fig.add_subplot(grid[0, 1])
    axr = fig.add_subplot(grid[1, 0])
    axf = fig.add_subplot(grid[1, 1])

    # Show an actual half of the extracted triangle mesh.  The gold wire is
    # the analytically prescribed source sphere and is drawn independently.
    faces = vertices[triangles]
    shown = faces[:, :, 1].mean(axis=1) <= 0
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
    light = np.abs(normals @ np.array([-.35, .55, .76]))
    rgb = np.array(matplotlib.colors.to_rgb(TEAL))
    colors = np.clip(rgb[None, :] * (.72 + .28 * light[:, None]), 0, 1)
    ax3.add_collection3d(Poly3DCollection(
        faces[shown], facecolors=colors[shown], linewidths=0,
        edgecolors="none", antialiaseds=False, rasterized=True))
    phi = np.linspace(0, 2*np.pi, 81)
    theta = np.linspace(0, np.pi, 41)
    xs = contact_radius*np.sin(theta[:, None])*np.cos(phi[None, :])
    ys = contact_radius*np.sin(theta[:, None])*np.sin(phi[None, :])
    zs = contact_radius*np.cos(theta[:, None])*np.ones_like(phi[None, :])
    ax3.plot_wireframe(xs, ys, zs, rstride=5, cstride=10,
                       color=GOLD, linewidth=.55, alpha=.65)
    bound = max(contact_radius * 1.2, np.linalg.norm(vertices, axis=1).max()*1.12)
    ax3.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound),
            xlabel="$x$", ylabel="$y$", zlabel="$z$")
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_proj_type("ortho")
    # Look through the removed +y half, so the visible mesh reads as a
    # hemispherical cutaway rather than the convex front of a full sphere.
    ax3.view_init(elev=24, azim=56)
    title = ("a  Computed contact boundary: cutaway" if solid_contact else
             "a  Source free-boundary proxy: cutaway")
    ax3.set_title(title, loc="left", pad=15)
    ax3.set_xticks([-contact_radius, 0, contact_radius])
    ax3.set_yticks([-contact_radius, 0, contact_radius])
    ax3.set_zticks([-contact_radius, 0, contact_radius])
    for axis in [ax3.xaxis, ax3.yaxis, ax3.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("#eef1f1")
        axis._axinfo["grid"].update(color="#edf0f0", linewidth=.5)
    ax3.legend(handles=[
        Line2D([0], [0], color=TEAL, lw=4,
               label=("Boundary of contacting tetrahedra" if solid_contact else
                      f"Computed P1 level $r={selected_level:g}$")),
        Line2D([0], [0], color=GOLD, lw=1.2,
               label=f"Reference sphere $\\rho={contact_radius:g}$"),
    ], loc="upper left", bbox_to_anchor=(.02, .95), fontsize=9)

    # Extract a source equatorial section from the actual mesh.  A tiny
    # positive offset avoids an exactly degenerate nodal plane.
    section, section_triangles, section_r = extract(
        points, cells, points[:, 2], 1e-10, field=nodes["r"])
    if not len(section_triangles):
        raise ValueError("The source mesh has no equatorial section")
    cmap = LinearSegmentedColormap.from_list(
        "horosphere_distance", ["#f8faf9", "#b5d6d3", TEAL, "#16494e"])
    maximum = max(float(nodes["r"].max()), 1e-12)
    levels = np.linspace(0, maximum*(1+1e-12), 24)
    im = axsection.tricontourf(section[:, 0], section[:, 1], section_triangles,
                             np.clip(section_r, 0, None), levels=levels, cmap=cmap)
    contour = surface_section(vertices, triangles)
    axsection.add_collection(LineCollection(contour, colors=TEAL, linewidths=1.3))
    angle = np.linspace(0, 2*np.pi, 301)
    axsection.plot(contact_radius*np.cos(angle), contact_radius*np.sin(angle),
                   color=GOLD, lw=1.5, ls="--")
    axsection.set(aspect="equal", xlabel="Source $x$", ylabel="Source $y$")
    axsection.set_title("b  Equatorial normal-distance field", loc="left", pad=10)
    axsection.set_xlim(points[:, 0].min(), points[:, 0].max())
    axsection.set_ylim(points[:, 1].min(), points[:, 1].max())
    colorbar = fig.colorbar(im, ax=axsection, fraction=.048, pad=.035)
    colorbar.set_label("Target normal distance $r$")

    rho = np.linalg.norm(points, axis=1)
    s = np.column_stack([nodes[name] for name in ["s0", "s1", "s2"]])
    computed_f = np.divide(np.einsum("ij,ij->i", s, points), rho,
                           out=np.zeros_like(rho), where=rho > 1e-14)
    sample = np.random.default_rng(0).choice(len(points),
                size=min(len(points), 6500), replace=False)
    for ax, computed, reference, ylabel, title in [
        (axr, nodes["r"], profile["r"], "Normal distance $R(\\rho)$",
         "c  Normal profile: contact and release"),
        (axf, computed_f, profile["f"], "Tangential amplitude $f(\\rho)$",
         "d  Tangential profile: radial symmetry"),
    ]:
        ax.axvspan(0, contact_radius, color=GOLD, alpha=.06, lw=0)
        ax.scatter(rho[sample], computed[sample], s=4, color=TEAL,
                   alpha=.35, edgecolors="none", rasterized=True,
                   label="Computed nodal values", zorder=2)
        ax.plot(profile["rho"], reference, color=GOLD, lw=1.7,
                label="Radial KKT reference", zorder=3)
        ax.axvline(contact_radius, color=GOLD, lw=.85, ls="--", alpha=.75)
        ax.set(xlim=(0, max(1., rho.max())), xlabel="Source radius $\\rho$",
               ylabel=ylabel)
        ax.set_title(title, loc="left", pad=10)
        ax.grid(axis="y", color="#edf0f0", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=9)
    axr.annotate("Contact", (.5*contact_radius, .025),
                 xycoords=("data", "axes fraction"), ha="center", color=GRAY)
    axr.annotate("Noncontact", (.5*(1+contact_radius), .025),
                 xycoords=("data", "axes fraction"), ha="center", color=GRAY)
    residual = summary.get("residual", summary.get("kkt_residual", ""))
    subtitle = (r"Target metric $dr^2+e^{2r}|ds|^2$: constant sectional curvature $-1$"
                + f"\nReference free boundary: sphere of radius {contact_radius:g}")
    if residual:
        subtitle += f"  |  KKT residual {float(residual):.2e}"
    fig.suptitle("A spherical free boundary in hyperbolic target geometry\n" + subtitle,
                 fontsize=14, linespacing=1.55)
    fig.savefig(path, dpi=220)
    if pdf:
        fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--levels", type=float, nargs="+", default=[1e-5, 1e-4],
                        help="positive metric-distance levels to audit")
    parser.add_argument("--surface-level", type=float, default=1e-5,
                        help="metric-distance level used in the VTK and figure")
    parser.add_argument("--contact-threshold", type=float, default=1e-7,
                        help="r threshold for the nodal noncontact indicator")
    parser.add_argument("--solid-contact-threshold", type=float, default=1e-12,
                        help="maximum nodal r for a fully contacting tetrahedron")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    with open(str(args.prefix) + "_summary.csv", newline="") as handle:
        summary = next(csv.DictReader(handle))
    if summary.get("status") != "CONVERGED":
        raise ValueError("Refusing to plot an unconverged iterate as a free-boundary candidate")
    levels = sorted(set(args.levels + [args.surface_level]))
    if not levels or any(not np.isfinite(level) or level <= 0 for level in levels):
        raise ValueError("Distance proxy levels must be finite and positive")
    if not np.isfinite(args.contact_threshold) or args.contact_threshold <= 0:
        raise ValueError("The nodal contact threshold must be finite and positive")
    if not np.isfinite(args.solid_contact_threshold) or args.solid_contact_threshold < 0:
        raise ValueError("The solid contact threshold must be finite and nonnegative")
    a = float(summary["contact_radius"])
    if not 0 < a < 1:
        raise ValueError("Expected a contact radius strictly between 0 and 1")
    nodes = read_csv(str(args.prefix) + "_nodes.csv", ["x", "y", "z", "s0", "s1", "s2", "r"])
    profile = read_csv(str(args.prefix) + "_profile.csv", ["rho", "f", "r"])
    if np.any(np.diff(profile["rho"]) <= 0):
        raise ValueError("The reference profile radii must be strictly increasing")
    cells = np.loadtxt(str(args.prefix) + "_cells.csv", delimiter=",", skiprows=1,
                       dtype=int).reshape(-1, 4)
    if not len(cells) or cells.min() < 0 or cells.max() >= len(nodes):
        raise ValueError("Empty or invalid tetrahedral connectivity")
    points = np.column_stack([nodes[name] for name in ["x", "y", "z"]])
    records = []
    selected = None
    for level in levels:
        vertices, triangles = extract(points, cells, nodes["r"], level)
        if not len(triangles):
            raise ValueError(f"No P1 surface at r={level:g}")
        record = dict(kind="positive_normal_distance_level", level=level,
                      components=topology(vertices, triangles),
                      error_to_true_free_boundary=radial_audit(vertices, triangles, a))
        expected = reference_level_radius(profile, level, a)
        if expected is not None:
            record["interpolated_reference_level_radius"] = expected
            record["error_to_reference_level"] = radial_audit(vertices, triangles, expected)
        records.append(record)
        if level == args.surface_level:
            selected = (vertices, triangles)
    vertices, triangles = extract(points, cells,
        (nodes["r"] > args.contact_threshold).astype(float), .5)
    records.append(dict(kind="nodal_noncontact_indicator_midpoint", level=.5,
        normal_distance_threshold=args.contact_threshold,
        components=topology(vertices, triangles),
        error_to_true_free_boundary=radial_audit(vertices, triangles, a)))
    vertices, triangles, contact_cells, contact_volume = contact_volume_boundary(
        points, cells, nodes["r"], args.solid_contact_threshold)
    components = topology(vertices, triangles)
    solid_contact = bool(components) and all(c["closed_manifold"] for c in components)
    records.append(dict(kind="contacting_tetrahedra_boundary", level=None,
        normal_distance_threshold=args.solid_contact_threshold,
        contacting_tetrahedra=contact_cells, contact_volume=contact_volume,
        closed_manifold_audit_passed=solid_contact,
        components=components,
        error_to_true_free_boundary=radial_audit(vertices, triangles, a)))
    if solid_contact:
        selected = (vertices, triangles)
    selected_kind = ("contacting_tetrahedra_boundary" if solid_contact else
                     "positive_normal_distance_level")
    geometry = dict(
        target_metric="dr^2 + exp(2*r)*(ds0^2 + ds1^2 + ds2^2)",
        target_sectional_curvature=-1,
        reference_free_boundary=dict(type="source sphere", radius=a,
                                     area=float(4*np.pi*a*a), volume=float(4*np.pi*a**3/3)),
        summary=summary, selected_surface_kind=selected_kind,
        selected_surface_level=None if solid_contact else args.surface_level,
        interpretation=("Surfaces extracted directly from tetrahedral P1 fields. "
            "The contacting tetrahedra boundary is the exterior mesh faces of the union "
            "of tetrahedra whose four nodal normal distances meet the solid-contact threshold. "
            "Positive distance levels and nodal indicator midpoints are numerical proxies; "
            "radius errors are measured in the Euclidean source metric. "
            "Integrated radial errors use the standard degree-two three-point triangle "
            "quadrature rule and include the interior of each planar face. "
            "No smoothing or Euclidean target embedding is used."),
        surfaces=records,
    )
    with open(str(args.prefix) + "_geometry.json", "w") as handle:
        json.dump(geometry, handle, indent=2, allow_nan=False)
        handle.write("\n")
    save_vtk(str(args.prefix) + "_surface.vtk", *selected)
    if solid_contact:
        vtk_path = Path(str(args.prefix) + "_surface.vtk")
        vtk_path.write_text(vtk_path.read_text().replace(
            "P1 level-set proxy", "Boundary of contacting tetrahedra", 1))
    make_figure(str(args.prefix) + "_figure.png", points, cells, nodes, profile,
                summary, *selected, args.surface_level, a, args.pdf, solid_contact)
    for record in records:
        audit = record["error_to_true_free_boundary"]
        print(json.dumps(dict(kind=record["kind"], level=record["level"],
                              components=record["components"], radial_error=audit)))
    print(f"Saved {args.prefix}_figure.png, _surface.vtk, and _geometry.json")


if __name__ == "__main__":
    main()
