namespace patch_stabilization_test {
static R plane(double *point, int, int) { return 0.62 - point[0]; }
}

TEST_CASE("Stationary patches include either cut neighbour", "[patch]") {
    // With P0, each off-diagonal jump-mass entry is exactly minus the
    // measure of the two-cell patch. This tests the assembled matrix, including
    // cut/uncut pairs whose uncut cell comes first in the mesh numbering.
    Mesh3 background(3, 3, 3, 0., 0., 0., 1., 1., 1.);
    FESpace3 level_space(background, DataFE<Mesh3>::P1);
    FunFEM<Mesh3> level(level_space, patch_stabilization_test::plane);
    InterfaceLevelSet<Mesh3> interface(background, level);
    ActiveMesh<Mesh3> active(background);
    active.truncate(interface, -1);
    FESpace3 background_space(background, DataFE<Mesh3>::P0);
    CutFESpaceT3 space(active, background_space);
    CutFEM<Mesh3> matrix(space);
    TestFunction<Mesh3> u(space, 1, 0), v(space, 1, 0);
    matrix.addPatchStabilization(innerProduct(jump(u), jump(v)), active);

    int reversed_pairs = 0;
    Matrix expected;
    for (int k = 0; k < active.last_element(); ++k) {
        const auto &element = space[k];
        for (int face = 0; face < Mesh3::Element::nea; ++face) {
            int neighbour_face = face;
            const int neighbour = active.ElementAdj(k, neighbour_face);
            if (neighbour <= k) continue;
            const bool ck = active.isCut(k, 0), cn = active.isCut(neighbour, 0);
            if (!ck && !cn) continue;
            if (!ck && cn) ++reversed_pairs;
            const auto &other = space[neighbour];
            const int i = element(0), j = other(0);
            const double weight = element.T.measure() + other.T.measure();
            expected[{i, i}] += weight;
            expected[{j, j}] += weight;
            expected[{i, j}] -= weight;
            expected[{j, i}] -= weight;
        }
    }
    REQUIRE(reversed_pairs > 0);
    for (const auto &[indices, value] : expected)
        REQUIRE(std::abs(matrix.mat_[0][indices] - value) < 1e-12);
    for (const auto &[indices, value] : matrix.mat_[0])
        REQUIRE(std::abs(expected[indices] - value) < 1e-12);
}
