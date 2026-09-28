#include "../example/obstacle_map/driver.hpp"

// Example DSL only. Weak forms and nonlinear steps: problem/obstacleMap.hpp.
// Geometry, mesh, reference profile and output: example/obstacle_map/.
int main(int argc, char** argv) {
    using namespace cutfem::obstacle;
    using namespace cutfem::obstacle::examples;

    return runExample(argc, argv, [](const ExampleOptions& options) {
        const RotatingProfile profile(/*alpha=*/std::acos(-0.5), /*a=*/0.3, /*k=*/2.712423);//1.0);

        return ObstacleMapExample<3>()
            .on(options.box ? box_mesh_3d(options.subdivisions)
                            : unit_ball_mesh_3d(options.subdivisions))
            .into(profile.chart(options.tube_radius))
            .withDirichlet([profile](const Point<3>& x) { return profile.exact(x); })
            .startingFrom([profile](const Point<3>& x) { return profile.initial(x, false); })
            .compareWith([profile](const Point<3>& x) { return profile.exact(x); })
            .usingSolver(options.solver)
            .writeTo(options.output_prefix)
            .solve();
    });
}
