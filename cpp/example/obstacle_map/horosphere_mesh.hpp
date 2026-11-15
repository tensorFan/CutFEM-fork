#pragma once

#include "benchmarks.hpp"

namespace cutfem::obstacle::examples {

// A fitted polyhedral unit ball with a tetrahedralization aligned to the
// nonsmooth planes of the homogeneous cube-to-ball map
//                 x -> ||x||_infinity * x / ||x||_2.
// Reflecting a Freudenthal subdivision in each octant aligns all planes
// |x_i|=|x_j| with element faces. The mesh therefore avoids both the boundary
// compression of the smooth cube map and elements crossing derivative jumps.
// Even n gives (n+1)^3 vertices, 6*n^3 tetrahedra, and 12*n^2 boundary faces;
// an odd subdivision count is rounded up to the next even count.
inline std::shared_ptr<Mesh3> horosphere_ball_mesh_3d(int subdivisions = 12) {
    if (subdivisions < 2 || subdivisions == std::numeric_limits<int>::max())
        throw std::invalid_argument("Horosphere ball needs at least two subdivisions");
    const int n = subdivisions + subdivisions%2;
    if (6.*n*n*n > std::numeric_limits<int>::max())
        throw std::invalid_argument("Horosphere ball exceeds the mesh index range");
    const int m = n/2, width = n+1;
    const long long vertex_count = 1LL*width*width*width;
    const long long cell_count = 6LL*n*n*n;
    const long long face_count = 12LL*n*n;
    if (vertex_count > std::numeric_limits<int>::max() ||
        cell_count > std::numeric_limits<int>::max() ||
        face_count > std::numeric_limits<int>::max())
        throw std::invalid_argument("Horosphere ball exceeds the mesh index range");
    const int nv = static_cast<int>(vertex_count), nt = static_cast<int>(cell_count);
    const int nbe = static_cast<int>(face_count);
    auto vertices = std::make_unique<Vertex3[]>(nv);
    auto cells = std::make_unique<Tet[]>(nt);
    auto faces = std::make_unique<Triangle3[]>(nbe);
    auto index = [width](const std::array<int,3>& p) {
        return p[0] + width*(p[1] + width*p[2]);
    };
    for (int k=0; k<width; ++k) for (int j=0; j<width; ++j) for (int i=0; i<width; ++i) {
        const std::array<int,3> q{i-m,j-m,k-m};
        const double norm = std::sqrt(1.*q[0]*q[0] + 1.*q[1]*q[1] + 1.*q[2]*q[2]);
        const int maximum = std::max({std::abs(q[0]),std::abs(q[1]),std::abs(q[2])});
        const double scale = norm > 0. ? maximum/(m*norm) : 0.;
        auto& vertex = vertices[index({i,j,k})];
        for (int d=0; d<3; ++d) vertex[d] = scale*q[d];
        vertex.lab = maximum == m ? 1 : 0;
    }
    constexpr int permutations[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    constexpr int parity[6] = {1,-1,-1,1,1,-1};
    int cell = 0;
    for (int octant=0; octant<8; ++octant) {
        const std::array<int,3> sign{(octant&1)?1:-1,(octant&2)?1:-1,(octant&4)?1:-1};
        for (int k=0; k<m; ++k) for (int j=0; j<m; ++j) for (int i=0; i<m; ++i) {
            const std::array<int,3> base{m+sign[0]*i,m+sign[1]*j,m+sign[2]*k};
            for (int permutation=0; permutation<6; ++permutation) {
                auto p = base;
                int ids[4] = {index(p),0,0,0};
                for (int v=0; v<3; ++v) {
                    const int d = permutations[permutation][v];
                    p[d] += sign[d];
                    ids[v+1] = index(p);
                }
                // Orient in the ORIGINAL cube; never hide a mapped inversion
                // by simply reordering an already inverted physical cell.
                if (parity[permutation]*sign[0]*sign[1]*sign[2] < 0)
                    std::swap(ids[1],ids[2]);
                Vertex3* points[4] = {&vertices[ids[0]],&vertices[ids[1]],
                                     &vertices[ids[2]],&vertices[ids[3]]};
                const double volume = DataTet::mesure(points);
                if (!(volume > 0.) || !std::isfinite(volume))
                    throw std::runtime_error("Aligned cube-to-ball map produced an inverted tetrahedron");
                cells[cell++].set(vertices.get(),ids,0);
            }
        }
    }
    int face = 0;
    for (int fixed=0; fixed<3; ++fixed) for (int side : {-1,1}) {
        const int first = (fixed+1)%3, second = (fixed+2)%3;
        for (int first_sign : {-1,1}) for (int second_sign : {-1,1})
            for (int j=0; j<m; ++j) for (int i=0; i<m; ++i) {
                std::array<int,3> p{m,m,m};
                p[fixed] += side*m;
                p[first] += first_sign*i;
                p[second] += second_sign*j;
                const int p00 = index(p);
                p[first] += first_sign;
                const int p10 = index(p);
                p[second] += second_sign;
                const int p11 = index(p);
                p[first] -= first_sign;
                const int p01 = index(p);
                int triangles[2][3] = {{p00,p10,p11},{p00,p11,p01}};
                for (auto& ids : triangles) {
                    const R3 a = vertices[ids[0]], b = vertices[ids[1]], c = vertices[ids[2]];
                    if (((b-a)^(c-a),a) < 0.) std::swap(ids[1],ids[2]);
                    faces[face++].set(vertices.get(),ids,1);
                }
            }
    }
    if (cell != nt || face != nbe)
        throw std::runtime_error("Aligned ball mesh connectivity count mismatch");
    // In a new-expression allocation precedes argument evaluation, so the
    // arrays remain owned if allocation fails. Mesh3's GenericMesh base owns
    // the arrays during construction and deletes them on constructor failure.
    return std::shared_ptr<Mesh3>(new Mesh3(nv,nt,nbe,vertices.release(),
                                          cells.release(),faces.release()));
}

} // namespace cutfem::obstacle::examples
