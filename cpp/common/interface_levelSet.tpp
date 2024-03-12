/*
This file is part of CutFEM-Library.

CutFEM-Library is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

CutFEM-Library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
CutFEM-Library. If not, see <https://www.gnu.org/licenses/>
*/
#ifndef COMMON_LEVELSET_INTERFACE_TPP
#define COMMON_LEVELSET_INTERFACE_TPP

template <typeMesh M>
template <typeFunFEM Fct>
InterfaceLevelSet<M>::InterfaceLevelSet(const M &MM, const Fct &lss, int label)
    : Interface<M>(MM), ls_(lss.getArray()) {
    make_patch(label);
}

template <typeMesh M>
SignElement<typename InterfaceLevelSet<M>::Element> InterfaceLevelSet<M>::get_SignElement(int k) const {
    typedef typename InterfaceLevelSet<M>::Element Element;
    byte loc_ls[Element::nv];
    for (int i = 0; i < Element::nv; ++i) {
        int iglb  = this->backMesh->at(k, i);
        loc_ls[i] = ls_sign[iglb];
    }
    return SignElement<Element>(loc_ls);
}

template <typeMesh M> bool InterfaceLevelSet<M>::isCut(int k) const {
    return (this->face_of_element_.find(k) != this->face_of_element_.end());
}

template <typeMesh M>
Partition<typename InterfaceLevelSet<M>::Element> InterfaceLevelSet<M>::get_partition(int k) const {
    typedef typename InterfaceLevelSet<M>::Element Element;

    double loc_ls[Element::nv];
    for (int i = 0; i < Element::nv; ++i) {
        int iglb  = this->backMesh->at(k, i);
        loc_ls[i] = ls_[iglb];
    }

    return Partition<Element>((*this->backMesh)[k], loc_ls);
}

template <typeMesh M>
Partition<typename InterfaceLevelSet<M>::Element::Face>
InterfaceLevelSet<M>::get_partition_face(const typename Element::Face &face, int k, int ifac) const {
    typedef typename InterfaceLevelSet<M>::Element Element;

    double loc_ls[Element::Face::nv];
    for (int i = 0; i < Element::Face::nv; ++i) {
        int j     = Element::nvhyperFace[ifac][i];
        int iglb  = this->backMesh->at(k, j);
        loc_ls[i] = ls_[iglb];
    }
    return Partition<typename Element::Face>(face, loc_ls);
}

template <typeMesh M>
void InterfaceLevelSet<M>::cut_partition(Physical_Partition<typename InterfaceLevelSet<M>::Element> &local_partition,
                                         std::vector<ElementIdx> &new_element_idx, std::list<int> &erased_element,
                                         int sign_part) const {
    std::cout << " An element might be cut multiplue time, and it is not "
                 "suppose to happen"
              << std::endl;
    exit(EXIT_FAILURE);
    // new_element_idx.resize(0);
    // erased_element.resize(0);
    // byte ls[3];
    // const Element& T(local_partition.T);
    // int kb = (*this->backMesh)(T);
    // for(int k=0; k<local_partition.nb_element(0);++k){ // 0 is just useless. Not needed by this class

    //     // BUILD ELEMENT
    //     const CutElement<Element> K = local_partition.get_element(k);
    //     // for(int i=0;i<3;++i) ls[i] = util::sign(fun.eval(kb, K[i]));
    //     for(int i=0;i<3;++i) ls[i] = ls_[i]; //util::sign(fun.eval(kb, K[i]));

    //     //COMPUTE THE PARTITION
    //     const RefPartition<Triangle2>& patch(RefPartition<Triangle2>::instance(ls));

    //     // interface i CUT THIS LOCAL ELEMENT k
    //     if(patch.is_cut()) {
    //         erased_element.push_back(k);

    //         // LOOP OVER ELEMENTS IN PATCH
    //         // need to get only the part corresponding to the sign
    //         for(auto it = patch.element_begin(); it != patch.element_end(); ++it) {

    //         // if(patch.whatSign(it) != sign_part &&  patch.whatSign(it) != 0) continue;
    //         if(patch.whatSign(it) != sign_part ) continue;

    //         // create the Nodes
    //         // std::cout << " index node to create " << std::endl;
    //         int idx_begin = local_partition.nb_node();
    //         ElementIdx idx_array(idx_begin, idx_begin+1, idx_begin+2);
    //         for(int i=0; i<3;++i) {
    //             Uint idx = (*it)[i];
    //             // std::cout << idx << std::endl;
    //     //
    //             if(idx < 3) {
    //             local_partition.add_node(K[idx]);
    //             // std::cout << K[idx] << std::endl;

    //             }
    //             else{
    //             int i0 = Triangle2::nvedge[idx - 3][0];
    //             int i1 = Triangle2::nvedge[idx - 3][1];
    //             local_partition.add_node(get_intersection_node(kb,K[i0], K[i1]));
    //             // local_partition.add_node(get_intersection_node(kb,i0,i1,K[i0],K[i1]));
    //             // std::cout << get_intersection_node(kb,K[i0], K[i1]) << std::endl;
    //             }
    //         }
    //         // ADD THE INDICES
    //         new_element_idx.push_back(idx_array);
    //         }

    //         // std::cout << " local element " << k << " is cut" << std::endl;
    //     }

    //     else {
    //         // std::cout << " local element " << k << " is not cut" << std::endl;
    //         // has to be removed if not in domain
    //         if(patch.whatSign() != sign_part) {
    //             erased_element.push_back(k);
    //         };
    //     }
    // }
};

template <typeMesh M> R InterfaceLevelSet<M>::measure(const Face &f) const {
    Rd l[nve];
    for (int i = 0; i < nve; ++i)
        l[i] = this->vertices_[f[i]];
    return geometry::measure_hyper_simplex(l);
};

// Rd get_intersection_node(int k, const Rd A, const Rd B) const {
//   double fA = fun.eval(k, A);
//   double fB = fun.eval(k, B);
// }
// Rn get_intersection_node(int k, int iA, int iB, const Rn A, const Rn B) const {
//   double fA = ls_[iA];
//   double fB = ls_[iB];
//   double t = -fA/(fB-fA);
//   return (1-t) * A + t * B;
// }

template <typeMesh M>
typename InterfaceLevelSet<M>::Rd
InterfaceLevelSet<M>::mapToPhysicalFace(int ifac, const typename InterfaceLevelSet<M>::Element::RdHatBord x) const {
    typename InterfaceLevelSet<M>::Rd N[nve];
    for (int i = 0; i < nve; ++i)
        N[i] = this->vertices_[this->faces_[ifac][i]];
    return geometry::map_point_to_simplex(N, x);
}

template <typeMesh M> void InterfaceLevelSet<M>::make_patch(int label) {

    typedef typename InterfaceLevelSet<M>::Element Element;
    assert(this->backMesh);
    this->faces_.resize(0); // reinitialize arrays
    this->vertices_.resize(0);
    this->element_of_face_.resize(0);
    this->outward_normal_.resize(0);
    this->face_of_element_.clear();

    const M &Th = *(this->backMesh);
    util::copy_levelset_sign(ls_, ls_sign);

    const Uint nb_vertex_K = Element::nv;
    double loc_ls[nb_vertex_K];
    byte loc_ls_sign[nb_vertex_K];

    for (int k = 0; k < this->backMesh->nbElmts(); k++) { // loop over elements

        const Element &K(Th[k]);

        for (Uint i = 0; i < K.nv; ++i) {
            loc_ls_sign[i] = ls_sign[Th(K[i])];
            loc_ls[i]      = ls_[Th(K[i])];
        }
        const RefPatch<Element> &cut = RefPatch<Element>::instance(loc_ls_sign);
        if (cut.empty())
            continue;

        for (typename RefPatch<Element>::const_face_iterator it = cut.face_begin(), end = cut.face_end(); it != end;
             ++it) {
            this->face_of_element_[k] = this->element_of_face_.size();
            this->faces_.push_back(make_face(*it, K, loc_ls, label));
            this->element_of_face_.push_back(k);
            this->outward_normal_.push_back(make_normal(K, loc_ls));
        }
    }
}

template <typeMesh M>
const typename InterfaceLevelSet<M>::Face
InterfaceLevelSet<M>::make_face(const typename RefPatch<Element>::FaceIdx &ref_tri, const typename Mesh::Element &K,
                                const double lset[Element::nv], int label) {

    Uint loc_vert_num;
    Uint triIdx[nve];

    for (Uint j = 0; j < nve; ++j) {
        loc_vert_num = ref_tri[j];

        if (loc_vert_num < K.nv) { // zero vertex
            // const Uint idx = (*this->backMesh)(K[loc_vert_num]);
            // Rd Q           = (*this->backMesh)(idx);
            // this->vertices_.push_back(Q);
            // triIdx[j] = this->vertices_.size() - 1;
            std::cout << " Interface cutting through a node " << std::endl;
            exit(EXIT_FAILURE);
            // assert(0);
        } else { // genuine edge vertex

            const Ubyte i0 = Mesh::Element::nvedge[loc_vert_num - K.nv][0],
                        i1 = Mesh::Element::nvedge[loc_vert_num - K.nv][1];

            const double t = lset[i0] / (lset[i0] - lset[i1]);
            Rd Q           = (1.0 - t) * ((Rd)K[i0]) + t * ((Rd)K[i1]); // linear interpolation
            this->vertices_.push_back(Q);
            triIdx[j] = this->vertices_.size() - 1;

            this->edge_of_node_.push_back(loc_vert_num - K.nv);
        }
    }
    return Face(triIdx, label);
}

/**
 * @brief Compute normal from level set function
 *
 * @tparam M Mesh
 * @param K Mesh element
 * @param lset c_i, where the level set function is given by phi(x) = sum_{i=0}^{local DOFs} c_i psi_i(x),
 * where psi_i are the basis functions.
 * @return InterfaceLevelSet<M>::Rd Normalized normal vector on K \cap Gamma_h.
 * @note It holds that grad(phi) = sum_{i=0}^{local DOFs} c_i \grad(psi_i)(x),
 * and for P1 triangular elements, grad(psi_i) is constant.
 */
template <typeMesh M>
typename InterfaceLevelSet<M>::Rd InterfaceLevelSet<M>::make_normal(const typename Mesh::Element &K,
                                                                    const double lset[Element::nv]) {

    Rd grad[Element::nv];
    K.Gradlambda(grad); // compute gradient of the basis functions
    Rd normal_ls;
    for (int i = 0; i < Mesh::Element::nv; ++i) {
        normal_ls += grad[i] * lset[i]; // grad(psi_i) * c_i
    }
    normal_ls /= normal_ls.norm();
    return normal_ls;
}

#endif
