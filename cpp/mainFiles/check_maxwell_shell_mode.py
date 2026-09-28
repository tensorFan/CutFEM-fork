#!/usr/bin/env python3
"""Compare a shell volume VTK with the first PEC Maxwell eigenspace.

Requires NumPy and VTK. Geometry: centre (pi/2, pi/2, pi/2), radii pi/5, pi/3.
The comparison integrates over the exported polyhedral domain; it is a
diagnostic, not a rigorous error bound on the curved shell.
"""

import argparse
import math

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def j1(x):
    return np.sin(x) / x**2 - np.cos(x) / x


def y1(x):
    return -np.cos(x) / x**2 - np.sin(x) / x


def dj1(x):
    """Derivative of x*j_1(x)."""
    return np.sin(x) + np.cos(x) / x - np.sin(x) / x**2


def dy1(x):
    """Derivative of x*y_1(x)."""
    return -np.cos(x) + np.sin(x) / x + np.cos(x) / x**2


def first_wavenumber():
    # PEC TM modes satisfy (r*z_l(k*r))'=0 at both shell boundaries.
    def determinant(k):
        a, b = k * math.pi / 5, k * math.pi / 3
        return dj1(a) * dy1(b) - dy1(a) * dj1(b)

    low, high = 1., 2.
    f_low = determinant(low)
    assert f_low * determinant(high) < 0.
    for _ in range(60):
        mid = (low + high) / 2
        f_mid = determinant(mid)
        if f_low * f_mid <= 0.:
            high = mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def reference_basis(points, k):
    """Columns span the three orientations of the first (l=1, TM) mode.

    For a constant vector a, E_a = curl(z(r)/r * (a cross (x-c))).
    This gives curl curl E_a = k^2 E_a and div E_a = 0.
    Its tangential component is (r*z)'/r * (a - (a.n)*n).
    """
    relative = points - math.pi / 2
    radius = np.linalg.norm(relative, axis=1)
    normal = relative / radius[:, None]
    t = k * radius
    c1, c2 = dy1(k * math.pi / 5), -dj1(k * math.pi / 5)
    z = c1 * j1(t) + c2 * y1(t)
    derivative = c1 * dj1(t) + c2 * dy1(t)
    return (derivative / radius)[:, None, None] * np.eye(3) + (
        (2 * z - derivative) / radius
    )[:, None, None] * normal[:, :, None] * normal[:, None, :]


def check(filename, k):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    grid = reader.GetOutput()
    if grid.GetNumberOfCells() == 0 or grid.GetPointData().GetArray("eigenfunction") is None:
        raise ValueError(f"{filename}: expected a volume VTK with eigenfunction vectors")
    if not np.all(vtk_to_numpy(grid.GetCellTypesArray()) == vtk.VTK_TETRA):
        raise ValueError(f"{filename}: expected tetrahedral cells")
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1, 4)
    points = vtk_to_numpy(grid.GetPoints().GetData())[connectivity]
    field = vtk_to_numpy(grid.GetPointData().GetArray("eigenfunction"))[connectivity]
    volume = abs(np.linalg.det(points[:, 1:] - points[:, :1])) / 6
    gram, load, mass = np.zeros((3, 3)), np.zeros(3), 0.
    # Four-point rule is exact for the discrete field's quadratic L2 integrand.
    for vertex in range(4):
        weights = np.full(4, (5 - np.sqrt(5)) / 20)
        weights[vertex] = (5 + 3 * np.sqrt(5)) / 20
        x = np.einsum("i,nij->nj", weights, points)
        u = np.einsum("i,nij->nj", weights, field)
        exact = reference_basis(x, k)
        gram += np.einsum("n,nij,nik->jk", volume / 4, exact, exact)
        load += np.einsum("n,nij,ni->j", volume / 4, exact, u)
        mass += np.sum(volume / 4 * np.sum(u * u, axis=1))
    if not np.isfinite(mass) or mass <= 0.:
        raise ValueError(f"{filename}: invalid field norm")
    fraction = float(load @ np.linalg.solve(gram, load) / mass)
    eigenvalue = grid.GetFieldData().GetArray("eigenvalue")
    print(filename)
    if eigenvalue is not None:
        value = eigenvalue.GetValue(0)
        print(f"  lambda = {value:.12g}; reference = {k*k:.12g}; relative error = {abs(value/(k*k)-1):.2%}")
    print(f"  physical L2 norm = {math.sqrt(mass):.12g}")
    print(f"  energy in first analytic eigenspace = {fraction:.2%}")
    print(f"  relative L2 distance to eigenspace = {math.sqrt(max(0., 1-fraction)):.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="shell volume .vtk files (not *_surface.vtk)")
    args = parser.parse_args()
    k = first_wavenumber()
    for filename in args.files:
        check(filename, k)
