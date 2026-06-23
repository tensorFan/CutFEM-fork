#!/usr/bin/env python3
"""Generate fitted tetrahedral meshes for the Maxwell eigenvalue tests.

The script writes two three-level mesh families:

    cube_0.msh, cube_1.msh, cube_2.msh
    spherical_shell_0.msh, spherical_shell_1.msh, spherical_shell_2.msh

Geometry and default resolution match the comparison driver:

    cube             = [0, pi]^3
    shell centre     = (pi/2, pi/2, pi/2)
    shell inner rad. = pi/6
    shell outer rad. = pi/3
    h_level          = pi / ((nx0 - 1) * 2**level),  nx0 = 7

The files are written as first-order, ASCII Gmsh 2.2 meshes because that is the
format used by Mesh3(..., MeshFormat::mesh_gmsh) in the CutFEM examples.

In build:
(fenicsx-env) [darth@darth-pc build]$ python3 ../cpp/mainFiles/notebooks/generate_fitted_maxwell_meshes.py     --output-dir ../cpp/mainFiles/meshes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

try:
    import gmsh
except ImportError as exc:  # pragma: no cover - depends on user's environment
    raise SystemExit(
        "The gmsh Python module is required. Install Gmsh and its Python module, "
        "for example with `python3 -m pip install gmsh`."
    ) from exc


PI = math.pi
CUBE_LENGTH = PI
SHELL_CENTER = (0.5 * PI, 0.5 * PI, 0.5 * PI)
RADIUS_INNER = PI / 6.0
RADIUS_OUTER = PI / 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three fitted cube meshes and three fitted spherical-shell meshes."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../cpp/mainFiles/meshes"),
        help="directory receiving the .msh files (default: ../cpp/mainFiles/meshes)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="number of refinement levels per geometry (default: 3)",
    )
    parser.add_argument(
        "--nx0",
        type=int,
        default=7,
        help="initial cube-grid point count used to define h0=pi/(nx0-1) (default: 7)",
    )
    parser.add_argument(
        "--only",
        choices=("all", "cube", "spherical_shell"),
        default="all",
        help="restrict generation to one geometry (default: all)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open the Gmsh GUI after each generated mesh",
    )
    return parser.parse_args()


def physical_group(dim: int, tags: Iterable[int], tag: int, name: str) -> None:
    entity_tags = list(tags)
    if not entity_tags:
        raise RuntimeError(f"Cannot create empty physical group {name!r}.")
    gmsh.model.addPhysicalGroup(dim, entity_tags, tag)
    gmsh.model.setPhysicalName(dim, tag, name)


def configure_mesh(h: float) -> None:
    # Keep all refinement levels directly comparable: the prescribed target
    # size is global and is halved from one level to the next.
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)

    # Linear tetrahedra and a conservative 3-D Delaunay mesher.
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    # CutFEM's existing Gmsh reader examples use legacy .msh input.  ASCII 2.2
    # is the most portable variant and preserves triangle boundary labels.
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)


def write_mesh(output_path: Path, show: bool) -> None:
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.write(str(output_path))
    print(f"Wrote {output_path}")
    if show:
        gmsh.fltk.run()


def generate_cube(output_path: Path, h: float, show: bool) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(output_path.stem)

        volume = gmsh.model.occ.addBox(0.0, 0.0, 0.0, PI, PI, PI)
        gmsh.model.occ.synchronize()

        boundary = gmsh.model.getBoundary([(3, volume)], oriented=False, recursive=False)
        boundary_tags = [tag for dim, tag in boundary if dim == 2]

        physical_group(3, [volume], 1, "domain")
        physical_group(2, boundary_tags, 1, "boundary")

        configure_mesh(h)
        write_mesh(output_path, show)
    finally:
        gmsh.finalize()


def generate_spherical_shell(output_path: Path, h: float, show: bool) -> None:
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add(output_path.stem)

        cx, cy, cz = SHELL_CENTER
        outer = gmsh.model.occ.addSphere(cx, cy, cz, RADIUS_OUTER)
        inner = gmsh.model.occ.addSphere(cx, cy, cz, RADIUS_INNER)
        shell_dimtags, _ = gmsh.model.occ.cut(
            [(3, outer)], [(3, inner)], removeObject=True, removeTool=True
        )
        gmsh.model.occ.synchronize()

        volume_tags = [tag for dim, tag in shell_dimtags if dim == 3]
        if len(volume_tags) != 1:
            raise RuntimeError(f"Expected one shell volume, got {shell_dimtags}.")

        boundary = gmsh.model.getBoundary(shell_dimtags, oriented=False, recursive=False)
        inner_tags: list[int] = []
        outer_tags: list[int] = []
        midpoint = 0.5 * (RADIUS_INNER + RADIUS_OUTER)

        for dim, tag in boundary:
            if dim != 2:
                continue
            # The centre of mass of either complete spherical surface is the
            # common shell centre, so classify using the surface bounding box.
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            radius = 0.5 * max(xmax - xmin, ymax - ymin, zmax - zmin)
            if radius < midpoint:
                inner_tags.append(tag)
            else:
                outer_tags.append(tag)

        if not inner_tags or not outer_tags:
            raise RuntimeError(
                "Could not distinguish the inner and outer shell surfaces: "
                f"inner={inner_tags}, outer={outer_tags}."
            )

        physical_group(3, volume_tags, 1, "domain")
        physical_group(2, outer_tags, 1, "outer_boundary")
        physical_group(2, inner_tags, 2, "inner_boundary")

        configure_mesh(h)
        write_mesh(output_path, show)
    finally:
        gmsh.finalize()


def main() -> None:
    args = parse_args()
    if args.levels < 1:
        raise SystemExit("--levels must be at least 1.")
    if args.nx0 < 2:
        raise SystemExit("--nx0 must be at least 2.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    h0 = CUBE_LENGTH / float(args.nx0 - 1)

    for level in range(args.levels):
        h = h0 / (2**level)
        equivalent_nx = (args.nx0 - 1) * (2**level) + 1
        print(
            f"\nlevel={level}, target h={h:.16g}, "
            f"equivalent cube nx={equivalent_nx}"
        )

        if args.only in ("all", "cube"):
            generate_cube(args.output_dir / f"cube_{level}.msh", h, args.show)

        if args.only in ("all", "spherical_shell"):
            generate_spherical_shell(
                args.output_dir / f"spherical_shell_{level}.msh", h, args.show
            )


if __name__ == "__main__":
    main()
