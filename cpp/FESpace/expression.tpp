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

// ============================================================================
// FunFEM Constructors from Expressions (Special Implementations)
// ============================================================================

template <typename M>
FunFEM<M>::FunFEM(const FESpace &vh, const ExpressionVirtual &fh)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh),
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    assert(Vh->N == 1);
    double dataSend[Vh->nbDoF];
    Rn_ fhSend(dataSend, Vh->nbDoF);
    fhSend        = 1e+50;
    const int d   = Vh->N;
    const int nve = Vh->TFE(0)->NbPtforInterpolation;
    KNM<R> Vpf(nve, d);               // value of f at the interpolation points
    KN<R> ggf(Vh->MaxNbDFPerElement); // stock the values of the dof of the
                                      // interpolate

    for (int k = Vh->first_element(); k < Vh->last_element();
         k += Vh->next_element()) {

        const FElement &FK((*Vh)[k]);
        const int nbdf   = FK.NbDoF(); // nof local
        const int domain = FK.whichDomain();
        const int kb     = Vh->idxElementInBackMesh(k);

        for (int p = 0; p < FK.tfe->NbPtforInterpolation;
             p++) {                // all interpolation points
            const Rd &P(FK.Pt(p)); // the coordinate of P in K hat
            for (int i = 0; i < d; ++i) {
                Vpf(p, i) = fh.evalOnBackMesh(kb, domain, P);
            }
        }
        std::cout << Vpf << std::endl;
        FK.Pi_h(Vpf, ggf);
        for (int df = 0; df < nbdf; df++) {
            fhSend(FK.loc2glb(df)) = ggf[df];
            // fh[K(df)] =  ggf[df] ;
        }
        // for(int j=FK.dfcbegin(0);j<FK.dfcend(0);++j) {
        //   Rd mip = FK.Pt(j);
        //   fhSend(FK.loc2glb(j)) = fh.evalOnBackMesh(kb, domain, mip);
        //   // v(FK.loc2glb(j)) = fh.evalOnBackMesh(kb, domain, mip);
        // }
        getchar();
    }
#ifdef USE_MPI
    MPIcf::AllReduce(dataSend, v, fhSend.size(), MPI_MIN);
#else
    assert(0 && "need to fixe the output");
#endif
}

template <typename M>
FunFEM<M>::FunFEM(const FESpace &vh, const ExpressionVirtual &fh1,
                  const ExpressionVirtual &fh2)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh),
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    assert(Vh->N == 2);
    double dataSend[Vh->nbDoF];
    Rn_ fhSend(dataSend, Vh->nbDoF);
    fhSend        = 1e+50;
    const int d   = Vh->N;
    const int nve = Vh->TFE(0)->NbPtforInterpolation;
    KNM<R> Vpf(nve, d);               // value of f at the interpolation points
    KN<R> ggf(Vh->MaxNbDFPerElement); // stock the values of the dof of the
                                      // interpolate

    for (int k = Vh->first_element(); k < Vh->last_element();
         k += Vh->next_element()) {

        const FElement &FK((*Vh)[k]);
        const int nbdf   = FK.NbDoF(); // nof local
        const int domain = FK.whichDomain();
        const int kb     = Vh->idxElementInBackMesh(k);

        for (int p = 0; p < FK.tfe->NbPtforInterpolation;
             p++) {                // all interpolation points
            const Rd &P(FK.Pt(p)); // the coordinate of P in K hat
            for (int i = 0; i < d; ++i) {
                const ExpressionVirtual &fh = (d == 0) ? fh1 : fh2;
                Vpf(p, i)                   = fh.evalOnBackMesh(kb, domain, P);
            }
        }
        // std::cout << Vpf << std::endl;
        FK.Pi_h(Vpf, ggf);
        for (int df = 0; df < nbdf; df++) {
            fhSend(FK.loc2glb(df)) = ggf[df];
            // fh[K(df)] =  ggf[df] ;
        }
        // for(int i=0, j=FK.dfcbegin(ci);j<FK.dfcend(ci);++j,++i) {
        //   Rd mip = FK.Pt(i);
        //   fhSend(FK.loc2glb(j)) = fh.evalOnBackMesh(kb, domain, mip);
        // }
    }
#ifdef USE_MPI
    MPIcf::AllReduce(dataSend, v, fhSend.size(), MPI_MIN);
#else
    assert(0 && "need to fixe the output");
#endif
}

// ============================================================================
// FunFEM Basic Template Implementation
// ============================================================================

template <typename M>
FunFEM<M>::FunFEM(const FESpace &vh)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh), 
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {}

template <typename M>
FunFEM<M>::FunFEM(const FESpace &vh, const TimeSlab &in)
    : FunFEMVirtual(vh.NbDoF() * in.NbDoF()), alloc(true), Vh(&vh), In(&in),
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {}

template <typename M>
template <Vector vector_t>
FunFEM<M>::FunFEM(const FESpace &vh, const TimeSlab &in, vector_t &u)
    : FunFEMVirtual(u.data(), u.size()), alloc(false), Vh(&vh), In(&in),
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {}

template <typename M>
template <Vector vector_t>
FunFEM<M>::FunFEM(const FESpace &vh, vector_t &u)
    : FunFEMVirtual(u.data(), u.size()), alloc(false), Vh(&vh), 
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {}

template <typename M>
template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
FunFEM<M>::FunFEM(const FESpace &vh, fct_t f)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh), 
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    interpolate(*Vh, this->v, f);
}

template <typename M>
FunFEM<M>::FunFEM(const FESpace &vh, double f)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh), 
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    this->v = f;
}

template <typename M>
template <typename fun_t>
FunFEM<M>::FunFEM(const FESpace &vh, fun_t f, R tid)
    : FunFEMVirtual(vh.NbDoF()), alloc(true), Vh(&vh), 
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    interpolate(*Vh, this->v, f, tid);
}

template <typename M>
template <typename fun_t>
FunFEM<M>::FunFEM(const FESpace &vh, const TimeSlab &in, fun_t f)
    : FunFEMVirtual(vh.NbDoF() * in.NbDoF()), alloc(true), Vh(&vh), In(&in),
      databf(new double[10 * vh[0].NbDoF() * vh.N * 4]) {
    interpolate(*Vh, *In, this->v, f);
}

template <typename M>
void FunFEM<M>::init(const FESpace &vh) {
    assert(!data_);
    Vh    = &vh;
    data_ = new double[Vh->NbDoF()];
    v.set(data_, Vh->NbDoF());
    alloc = true;
    v     = 0.;
    if (!databf)
        databf = new double[10 * vh[0].NbDoF() * vh.N * 4];
}

template <typename M>
void FunFEM<M>::init(const KN_<R> &a) {
    assert(v.size() == a.size());
    for (int i = 0; i < v.size(); ++i)
        data_[i] = a(i);
}

template <typename M>
template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionLevelSetTime<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
void FunFEM<M>::init(const FESpace &vh, fct_t f) {
    assert(!data_);
    Vh    = &vh;
    data_ = new double[Vh->NbDoF()];
    v.set(data_, Vh->NbDoF());
    alloc = true;
    interpolate(*Vh, v, f);

    if (!databf)
        databf = new double[10 * vh[0].NbDoF() * vh.N * 4];
}

template <typename M>
void FunFEM<M>::init(const FESpace &vh, R (*f)(double *, int, R), R tid) {
    if (data_)
        delete[] data_;
    Vh    = &vh;
    data_ = new double[Vh->NbDoF()];
    v.set(data_, Vh->NbDoF());
    alloc = true;
    interpolate(*Vh, v, f, tid);
    if (!databf)
        databf = new double[10 * vh[0].NbDoF() * vh.N * 4];
}

template <typename M>
void FunFEM<M>::init(const FESpace &vh, R (*f)(R2, int, R), R tid) {
    if (data_)
        delete[] data_;
    Vh    = &vh;
    data_ = new double[Vh->NbDoF()];
    v.set(data_, Vh->NbDoF());
    alloc = true;
    interpolate(*Vh, v, f, tid);
    if (!databf)
        databf = new double[10 * vh[0].NbDoF() * vh.N * 4];
}

template <typename M>
template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
void FunFEM<M>::init(const FESpace &vh, const TimeSlab &in, fct_t f) {
    if (data_)
        delete[] data_;
    Vh    = &vh;
    In    = &in;
    data_ = new double[Vh->NbDoF() * In->NbDoF()];
    v.set(data_, Vh->NbDoF() * In->NbDoF());
    alloc = true;
    interpolate(*Vh, *In, v, f);
    if (!databf)
        databf = new double[10 * vh[0].NbDoF() * vh.N * 4];
}

template <typename M>
double FunFEM<M>::eval(const int k, const R *x, int cu, int op) const {
    const FElement &FK((*Vh)[k]);
    int ndf = FK.NbDoF();
    // RNMK_ w(databf, ndf, Vh->N, op_dz + 1);
    // FK.BF(FK.T.toKref(x), w);
    const int nops = std::max({op_id, op_dx, op_dy, op_dz, op_dxx, op_dyy, op_dxy}) + 1;

    RNMK_ w(databf, ndf, Vh->N, nops);

    What_d whatd = Fop_D0;

    if (op == op_dx || op == op_dy || op == op_dz ||
        op == op_dxx || op == op_dyy || op == op_dxy)
        whatd |= Fop_D1;

    if (op == op_dxx || op == op_dyy || op == op_dxy)
        whatd |= Fop_D2;

    FK.BF(whatd, FK.T.toKref(x), w);

    double val = 0.;

    for (int j = FK.dfcbegin(cu); j < FK.dfcend(cu); ++j) {
        val += v[FK(j)] * w(j, cu, op);
    }

    return val;
}

template <typename M>
double FunFEM<M>::eval(const int k, const R *x, const R t, int cu, int op, int opt) const {
    if (!In)
        return eval(k, x, cu, op);

    const FElement &FK((*Vh)[k]);
    int ndf = FK.NbDoF();
    RNMK_ w(databf, ndf, Vh->N, op_dz + 1);
    KNMK<R> wt(In->NbDoF(), 1, op_dz);

    FK.BF(FK.T.toKref(x), w);
    In->BF(In->T.toKref(t), wt);

    double val = 0.;
    for (int jt = 0; jt < In->NbDoF(); ++jt) {
        for (int j = FK.dfcbegin(cu); j < FK.dfcend(cu); ++j) {
            val += v[FK(j) + jt * Vh->NbDoF()] * w(j, cu, op) * wt(jt, 0, opt);
        }
    }

    return val;
}

template <typename M> 
void FunFEM<M>::eval(R *u, const int k) const {
    assert(v && u);
    const FElement &FK((*Vh)[k]);
    for (int ci = 0; ci < Vh->N; ++ci) {
        for (int j = FK.dfcbegin(ci); j < FK.dfcend(ci); ++j)
            u[j] = v[FK(j)];
    }
}

template <typename M>
double FunFEM<M>::evalOnBackMesh(const int kb, int dom, const R *x, int cu, int op) const {
    int k = Vh->idxElementFromBackMesh(kb, dom);
    return eval(k, x, cu, op);
}

template <typename M>
double FunFEM<M>::evalOnBackMesh(const int kb, int dom, const R *x, const R t, int cu, int op, int opt) const {
    int k = Vh->idxElementFromBackMesh(kb, dom);
    return eval(k, x, t, cu, op, opt);
}

template <typename M>
void FunFEM<M>::print() const {
    std::cout << v << std::endl;
}

template <typename M>
std::shared_ptr<ExpressionFunFEM<M>> FunFEM<M>::expr(int i0) const {
    assert(i0 < Vh->N);
    return std::make_shared<ExpressionFunFEM<Mesh>>(*this, i0, op_id);
}

template <typename M>
std::list<std::shared_ptr<ExpressionFunFEM<M>>> FunFEM<M>::exprList(int n) const {
    if (n == -1)
        n = Vh->N;
    assert(n <= Vh->N);
    std::list<std::shared_ptr<ExpressionFunFEM<Mesh>>> l;
    for (int i = 0; i < n; ++i) {
        l.push_back(std::make_shared<ExpressionFunFEM<Mesh>>(*this, i, op_id));
    }
    return l;
}

template <typename M>
std::list<std::shared_ptr<ExpressionFunFEM<M>>> FunFEM<M>::exprList(int n, int i0) const {
    assert(n <= Vh->N);
    std::list<std::shared_ptr<ExpressionFunFEM<Mesh>>> l;
    for (int i = 0; i < n; ++i) {
        l.push_back(std::make_shared<ExpressionFunFEM<Mesh>>(*this, i + i0, op_id));
    }
    return l;
}

// ============================================================================
// Differential Operator Templates
// ============================================================================

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dx(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dx, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dy(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dy, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dz(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dz, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dt(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, u->op, op_dx, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dxx(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dxx, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dyy(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dyy, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dzz(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dzz, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dxy(const std::shared_ptr<ExpressionFunFEM<M>> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u->fun, u->cu, op_dxy, u->opt, u->domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dx(const ExpressionFunFEM<M> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u.fun, u.cu, op_dx, u.opt, u.domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dy(const ExpressionFunFEM<M> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u.fun, u.cu, op_dy, u.opt, u.domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dz(const ExpressionFunFEM<M> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u.fun, u.cu, op_dz, u.opt, u.domain);
}

template <typename M> 
std::shared_ptr<ExpressionFunFEM<M>> dt(const ExpressionFunFEM<M> &u) {
    return std::make_shared<ExpressionFunFEM<M>>(u.fun, u.cu, u.op, op_dx, u.domain);
}

// ============================================================================
// 2D Surface Operator Templates
// ============================================================================

template <typeMesh M>
ExpressionDSx2<M>::ExpressionDSx2(const FunFEM<M> &fh1)
    : fun(fh1), dxu1(fh1, 0, op_dx, 0, 0), dxu1nxnx(fh1, 0, op_dx, 0, 0), dyu1nxny(fh1, 0, op_dy, 0, 0) {
    dxu1nxnx.addNormal(0);
    dxu1nxnx.addNormal(0);
    dyu1nxny.addNormal(0);
    dyu1nxny.addNormal(1);
}

template <typeMesh M>
R ExpressionDSx2<M>::operator()(long i) const {
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSx2<M>::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DSx expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSx2<M>::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DSx expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSx2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxu1.evalOnBackMesh(k, dom, x, normal) - dxu1nxnx.evalOnBackMesh(k, dom, x, normal) -
           dyu1nxny.evalOnBackMesh(k, dom, x, normal);
}

template <typeMesh M>
R ExpressionDSx2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxu1.evalOnBackMesh(k, dom, x, t, normal) - dxu1nxnx.evalOnBackMesh(k, dom, x, t, normal) -
           dyu1nxny.evalOnBackMesh(k, dom, x, t, normal);
}

template <typeMesh M>
int ExpressionDSx2<M>::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

template <typeMesh M>
ExpressionDSy2<M>::ExpressionDSy2(const FunFEM<M> &fh1)
    : fun(fh1), dxu2(fh1, 1, op_dy, 0, 0), dxu2nxny(fh1, 1, op_dx, 0, 0), dyu2nyny(fh1, 1, op_dy, 0, 0) {
    dxu2nxny.addNormal(0);
    dxu2nxny.addNormal(1);
    dyu2nyny.addNormal(1);
    dyu2nyny.addNormal(1);
}

template <typeMesh M>
R ExpressionDSy2<M>::operator()(long i) const {
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSy2<M>::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DSy expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSy2<M>::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DSy expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDSy2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxu2.evalOnBackMesh(k, dom, x, normal) - dxu2nxny.evalOnBackMesh(k, dom, x, normal) -
           dyu2nyny.evalOnBackMesh(k, dom, x, normal);
}

template <typeMesh M>
R ExpressionDSy2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxu2.evalOnBackMesh(k, dom, x, t, normal) - dxu2nxny.evalOnBackMesh(k, dom, x, t, normal) -
           dyu2nyny.evalOnBackMesh(k, dom, x, t, normal);
}

template <typeMesh M>
int ExpressionDSy2<M>::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

template <typeMesh M>
ExpressionDivS2<M>::ExpressionDivS2(const FunFEM<M> &fh1) 
    : fun(fh1), dx(dxS(fh1)), dy(dyS(fh1)) {}

template <typeMesh M>
R ExpressionDivS2<M>::operator()(long i) const {
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDivS2<M>::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DivS expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDivS2<M>::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DivS expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

template <typeMesh M>
R ExpressionDivS2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dx->evalOnBackMesh(k, dom, x, normal) + dy->evalOnBackMesh(k, dom, x, normal);
}

template <typeMesh M>
R ExpressionDivS2<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dx->evalOnBackMesh(k, dom, x, t, normal) + dy->evalOnBackMesh(k, dom, x, t, normal);
}

template <typeMesh M>
int ExpressionDivS2<M>::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

template<typeMesh M> 
std::shared_ptr<ExpressionDSx2<M>> dxS(const FunFEM<M> &f1) { 
    return std::make_shared<ExpressionDSx2<M>>(f1); 
}

template<typeMesh M> 
std::shared_ptr<ExpressionDSy2<M>> dyS(const FunFEM<M> &f1) { 
    return std::make_shared<ExpressionDSy2<M>>(f1); 
}

template<typeMesh M> 
std::shared_ptr<ExpressionDivS2<M>> divS(const FunFEM<M> &f1) { 
    return std::make_shared<ExpressionDivS2<M>>(f1); 
}

// ============================================================================
// Surface Tension Templates
// ============================================================================

template <typename M> 
ExpressionLinearSurfaceTension<M>::ExpressionLinearSurfaceTension(const FunFEM<M> &fh, double ssigma0, double bbeta, double ttid)
    : fun(fh), sigma0(ssigma0), beta(bbeta), tid(ttid) {}

template <typename M> 
R ExpressionLinearSurfaceTension<M>::operator()(long i) const { 
    return fabs(fun(i)); 
}

template <typename M> 
R ExpressionLinearSurfaceTension<M>::eval(const int k, const R *x, const R *normal) const {
    assert(0);
    return 0.;
}

template <typename M> 
R ExpressionLinearSurfaceTension<M>::eval(const int k, const R *x, const R t, const R *normal) const {
    assert(0);
    return 0.;
}

template <typename M> 
R ExpressionLinearSurfaceTension<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    double val = fun.evalOnBackMesh(k, dom, x, tid, 0, op_id, op_id);
    return sigma0 * (1 - beta * val);
}

template <typename M> 
R ExpressionLinearSurfaceTension<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    double val = fun.evalOnBackMesh(k, dom, x, tid, 0, op_id, op_id);
    return sigma0 * (1 - beta * val);
}

template <typename M> 
int ExpressionLinearSurfaceTension<M>::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

template <typename M> 
ExpressionNonLinearSurfaceTension<M>::ExpressionNonLinearSurfaceTension(const FunFEM<M> &fh, double ssigma0, double bbeta, double ttid)
    : fun(fh), sigma0(ssigma0), beta(bbeta), tid(ttid) {}

template <typename M> 
R ExpressionNonLinearSurfaceTension<M>::operator()(long i) const { 
    return fabs(fun(i)); 
}

template <typename M> 
R ExpressionNonLinearSurfaceTension<M>::eval(const int k, const R *x, const R *normal) const {
    assert(0);
    return 0.;
}

template <typename M> 
R ExpressionNonLinearSurfaceTension<M>::eval(const int k, const R *x, const R t, const R *normal) const {
    assert(0);
    return 0.;
}

template <typename M> 
R ExpressionNonLinearSurfaceTension<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    double val = fun.evalOnBackMesh(k, dom, x, tid, 0, op_id, op_id);
    return sigma0 * (1 + beta * std::log(1 - val));
}

template <typename M> 
R ExpressionNonLinearSurfaceTension<M>::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    double val = fun.evalOnBackMesh(k, dom, x, tid, 0, op_id, op_id);
    return sigma0 * (1 + beta * std::log(1 - val));
}

template <typename M> 
int ExpressionNonLinearSurfaceTension<M>::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// Jump Template
// ============================================================================

template <class Mesh>
std::list<std::shared_ptr<ExpressionAverage>> jump(const FunFEM<Mesh> &fh, double kk1, double kk2) {
    std::list<std::shared_ptr<ExpressionAverage>> res;
    for (auto &expr : fh.exprList()) {
        res.push_back(jump(expr, kk1, kk2));
    }
    return res;
}