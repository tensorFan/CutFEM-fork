#ifndef _EXPRESSION_HPP
#define _EXPRESSION_HPP

#include "FESpace.hpp"
#include <cmath>
#include <memory>
#include <list>
#include <vector>

// ============================================================================
// Forward Declarations
// ============================================================================

class ExpressionVirtual;
template <typename M> class ExpressionFunFEM;
class ParameterCutFEM;

// ============================================================================
// Basic Types and Components
// ============================================================================

struct Normal_Component {
    virtual ~Normal_Component() = default;
    virtual int component() const = 0;
};

struct Normal_Component_X : public Normal_Component {
    int component() const override { return 0; }
};

struct Normal_Component_Y : public Normal_Component {
    int component() const override { return 1; }
};

struct Normal_Component_Z : public Normal_Component {
    int component() const override { return 2; }
};

struct BaseVector {
    virtual ~BaseVector() = default;
    virtual int operator[](int i) const = 0;
    virtual int cst(int i) const = 0;
};

struct Normal : public BaseVector {
    Normal_Component_X x;
    Normal_Component_Y y;
    Normal_Component_Z z;
    int operator[](int i) const override { return i; }
    int cst(int i) const override { return 1; }
};

struct Conormal : public BaseVector {
    Normal_Component_X x;
    Normal_Component_Y y;
    Normal_Component_Z z;
    int operator[](int i) const override { return i; }
    int cst(int i) const override { return 1; }
};

struct Tangent : public BaseVector {
    int operator[](int i) const override { return (i == 0); }
    int cst(int i) const override { return (i == 0) ? -1 : 1; }
};

struct Projection {
    Normal normal;
    KN<int> operator()(int i, int j) const {
        KN<int> ar(2);
        ar(0) = i;
        ar(1) = j;
        return ar;
    }
};

// ============================================================================
// Base Virtual Classes
// ============================================================================

class FunFEMVirtual {
public:
    double *data_ = nullptr;
    KN_<double> v;

    FunFEMVirtual() : v(data_, 0) {}
    FunFEMVirtual(int df) : data_(new double[df]), v(data_, df) { v = 0.; }
    FunFEMVirtual(KN_<double> &u) : v(u) {}
    FunFEMVirtual(double *u, int n) : v(u, n) {}

    virtual ~FunFEMVirtual() = default;

    virtual double eval(const int k, const R *x, int cu = 0, int op = 0) const {
        assert(0);
        return 0.;
    }
    
    virtual double eval(const int k, const R *x, const R t, int cu, int op, int opt) const {
        assert(0);
        return 0.;
    }
    
    virtual double evalOnBackMesh(const int k, int dom, const R *x, int cu = 0, int op = 0) const {
        assert(0);
        return 0.;
    }
    
    virtual double evalOnBackMesh(const int k, int dom, const R *x, const R t, int cu, int op, int opt) const {
        assert(0);
        return 0.;
    }
    
    virtual int idxElementFromBackMesh(int, int = 0) const {
        assert(0);
        return 0;
    }

    const KN_<double> &getArray() const { return v; }
    const KN_<double> &array() const { return v; }
    const double *data() const { return v.data(); }
};

class ExpressionVirtual {
public:
    int cu, op, opt;
    KN<int> ar_normal;
    int domain = -1;

    ExpressionVirtual() : cu(0), op(0), opt(0) {}
    ExpressionVirtual(int cc, int opp) : cu(cc), op(opp), opt(0) {}
    ExpressionVirtual(int cc, int opp, int oppt) : cu(cc), op(opp), opt(oppt) {}
    ExpressionVirtual(int cc, int opp, int oppt, int dom) : cu(cc), op(opp), opt(oppt), domain(dom) {}
    
    virtual ~ExpressionVirtual() = default;

    virtual R operator()(long i) const = 0;
    virtual R eval(const int k, const R *x, const R *normal = nullptr) const = 0;
    virtual R eval(const int k, const R *x, const R t, const R *normal = nullptr) const = 0;
    virtual R evalOnBackMesh(const int k, int dom, const R *x, const R *normal = nullptr) const = 0;
    virtual R evalOnBackMesh(const int k, int dom, const R *x, const R t, const R *normal = nullptr) const = 0;
    virtual int idxElementFromBackMesh(int kb, int dd = 0) const = 0;
    virtual std::vector<int> getAllDomainId(int k) const { assert(0); };

    virtual int size() const { assert(0); };

    R GevalOnBackMesh(const int k, int dom, const R *x, const R *normal) const {
        int theDomain = (domain == -1) ? dom : domain;
        return evalOnBackMesh(k, theDomain, x, normal);
    }
    
    R GevalOnBackMesh(const int k, int dom, const R *x, const R t, const R *normal) const {
        int theDomain = (domain == -1) ? dom : domain;
        return evalOnBackMesh(k, theDomain, x, t, normal);
    }

    ExpressionVirtual &operator=(const ExpressionVirtual &L) {
        cu = L.cu;
        op = L.op;
        opt = L.opt;
        ar_normal.init(L.ar_normal);
        return *this;
    }

    double computeNormal(const R *normal) const {
        if (normal == nullptr) return 1.;
        R val = 1;
        for (int i = 0; i < ar_normal.size(); ++i)
            val *= normal[ar_normal(i)];
        return val;
    }
    
    void addNormal(int i) {
        int l = ar_normal.size();
        ar_normal.resize(l + 1);
        ar_normal(l) = i;
    }

    virtual void whoAmI() const { 
        std::cout << " I am virtual class Expression" << std::endl; 
    }
};

// ============================================================================
// Main Template Classes
// ============================================================================

template <typename M> 
class FunFEM : public FunFEMVirtual {
public:
    typedef M Mesh;
    typedef GFESpace<Mesh> FESpace;
    typedef typename FESpace::FElement FElement;
    typedef typename Mesh::Rd Rd;

    bool alloc = false;
    double *databf = nullptr;
    FESpace const *Vh = nullptr;
    TimeSlab const *In = nullptr;

    // Constructors
    FunFEM() : FunFEMVirtual() {}
    explicit FunFEM(const FESpace &vh);
    FunFEM(const FESpace &vh, const TimeSlab &in);
    
    template <Vector vector_t>
    FunFEM(const FESpace &vh, const TimeSlab &in, vector_t &u);
    
    template <Vector vector_t>
    FunFEM(const FESpace &vh, vector_t &u);

    template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
    FunFEM(const FESpace &vh, fct_t f);
    
    FunFEM(const FESpace &vh, double f);
    
    template <typename fun_t>
    FunFEM(const FESpace &vh, fun_t f, R tid);
    
    template <typename fun_t>
    FunFEM(const FESpace &vh, const TimeSlab &in, fun_t f);
    
    FunFEM(const FESpace &vh, const ExpressionVirtual &fh);
    FunFEM(const FESpace &vh, const ExpressionVirtual &fh1, const ExpressionVirtual &fh2);

    // Initialization methods
    void init(const FESpace &vh);
    void init(const KN_<R> &a);
    
    template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionLevelSetTime<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
    void init(const FESpace &vh, fct_t f);
    
    void init(const FESpace &vh, R (*f)(double *, int, R), R tid);
    void init(const FESpace &vh, R (*f)(R2, int, R), R tid);
    
    template <typename fct_t> requires FunctionLevelSet<fct_t> || FunctionDomain<fct_t> || FunctionScalar<fct_t>
    void init(const FESpace &vh, const TimeSlab &in, fct_t f);

    // Accessors and operators
    double &operator()(int i) { return v(i); }
    double operator()(int i) const { return v(i); }
    operator Rn() const { return Rn(v); }

    // Evaluation methods
    double eval(const int k, const R *x, int cu = 0, int op = 0) const;
    double eval(const int k, const R *x, const R t, int cu = 0, int op = 0, int opt = 0) const;
    void eval(R *u, const int k) const;

    double evalOnBackMesh(const int kb, int dom, const R *x, int cu, int op) const;
    double evalOnBackMesh(const int kb, int dom, const R *x, const R t, int cu, int op, int opt) const;

    // Utility methods
    int size() const { return Vh->NbDoF() * ((In) ? In->NbDoF() : 1); }
    int size(int k) const { return (*Vh)[k].NbDoF(); }
    void print() const;
    int idxElementFromBackMesh(int kb, int dd = 0) const { return Vh->idxElementFromBackMesh(kb, dd); }
    std::vector<int> getAllDomainId(int k) const { return Vh->getAllDomainId(k); };

    const FESpace &getSpace() const { return *Vh; }
    BasisFctType getBasisFctType() const { return Vh->basisFctType; }
    int getPolynomialOrder() const { return Vh->polynomialOrder; }
    
    std::shared_ptr<ExpressionFunFEM<M>> expr(int i0 = 0) const;
    std::list<std::shared_ptr<ExpressionFunFEM<M>>> exprList(int n = -1) const;
    std::list<std::shared_ptr<ExpressionFunFEM<M>>> exprList(int n, int i0) const;

    friend void swap(FunFEM &f, FunFEM &g) {
        assert(g.v.size() == f.v.size());
        std::swap(g.data_, f.data_);
        g.v.set(g.data_, g.Vh->NbDoF());
        f.v.set(f.data_, f.Vh->NbDoF());
    }

    ~FunFEM() {
        if (databf) delete[] databf;
        if (alloc) delete[] data_;
    }

private:
    FunFEM(const FunFEM &f) = delete;
    void operator=(const FunFEM &f) = delete;
};

template <typename M> 
class ExpressionFunFEM : public ExpressionVirtual {
public:
    const FunFEM<M> &fun;

    ExpressionFunFEM(const FunFEM<M> &fh, int cc, int opp) : ExpressionVirtual(cc, opp), fun(fh) {}
    ExpressionFunFEM(const FunFEM<M> &fh, int cc, int opp, int oppt) : ExpressionVirtual(cc, opp, oppt), fun(fh) {}
    ExpressionFunFEM(const FunFEM<M> &fh, int cc, int opp, int oppt, int dom) : ExpressionVirtual(cc, opp, oppt, dom), fun(fh) {}

    // Copy operations
    ExpressionFunFEM(const ExpressionFunFEM &L) : ExpressionVirtual(L.cu, L.op, L.opt, L.domain), fun(L.fun) {
        ar_normal.init(L.ar_normal);
    }

    ExpressionFunFEM &operator=(const ExpressionFunFEM &L) {
        if (this != &L) {
            cu = L.cu;
            op = L.op;
            opt = L.opt;
            ar_normal.init(L.ar_normal);
            domain = L.domain;
        }
        return *this;
    }

    // Virtual method implementations
    R operator()(long i) const override { return fun(i); }
    int size() const override { return fun.size(); }
    void whoAmI() const override { std::cout << " I am class ExpressionFunFEM" << std::endl; }

    R eval(const int k, const R *x, const R *normal) const override { 
        return fun.eval(k, x, cu, op) * computeNormal(normal); 
    }
    
    R eval(const int k, const R *x, const R t, const R *normal) const override {
        return fun.eval(k, x, t, cu, op, opt) * computeNormal(normal);
    }

    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override {
        return fun.evalOnBackMesh(k, dom, x, cu, op) * computeNormal(normal);
    }
    
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override {
        return fun.evalOnBackMesh(k, dom, x, t, cu, op, opt) * computeNormal(normal);
    }

    int idxElementFromBackMesh(int kb, int dd = 0) const override { 
        return fun.idxElementFromBackMesh(kb, dd); 
    }

    const GFESpace<M> &getSpace() const { return *fun.Vh; }

    ExpressionFunFEM operator*(const Normal &n) {
        ExpressionFunFEM ff(*this);
        ff.addNormal(cu);
        return ff;
    }
};

// ============================================================================
// Expression Types
// ============================================================================

class ExpressionMultConst : public ExpressionVirtual {
    const std::shared_ptr<ExpressionVirtual> fun1;
    const double c;
    const bool nx, ny, nz;
    const R2 p;

public:
    ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const double &cc);
    ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_X &nnx);
    ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_Y &nny);
    ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_Z &nnz);

    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionAbs : public ExpressionVirtual {
    const std::shared_ptr<ExpressionVirtual> fun1;
public:
    ExpressionAbs(const std::shared_ptr<ExpressionVirtual> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionProduct : public ExpressionVirtual {
    const std::shared_ptr<ExpressionVirtual> fun1;
    const std::shared_ptr<ExpressionVirtual> fun2;
public:
    ExpressionProduct(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionPow : public ExpressionVirtual {
public:
    typedef std::shared_ptr<ExpressionVirtual> ptr_expr_t;
private:
    const ptr_expr_t fun1;
    const double n;
public:
    ExpressionPow(const ptr_expr_t &f1h, const double nn);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionDivision : public ExpressionVirtual {
    const std::shared_ptr<ExpressionVirtual> fun1;
    const std::shared_ptr<ExpressionVirtual> fun2;
public:
    ExpressionDivision(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionSum : public ExpressionVirtual {
    const std::shared_ptr<ExpressionVirtual> fun1;
    const std::shared_ptr<ExpressionVirtual> fun2;
public:
    ExpressionSum(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// ============================================================================
// Vector Calculus Expressions
// ============================================================================

class ExpressionNormal2 : public ExpressionVirtual {
    typedef Mesh2 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> uxnx, uyny;
    double c0 = 1;
public:
    ExpressionNormal2(const FunFEM<M> &fh1, const Normal n);
    ExpressionNormal2(const FunFEM<M> &fh1, const Tangent t);
    ExpressionNormal2(const FunFEM<M> &fh1, const Conormal n);
    
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionNormal3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> uxnx, uyny, uznz;
public:
    ExpressionNormal3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// Cross product expressions for 3D
class ExpressionNormalCrossX3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> uzny, uynz;
public:
    ExpressionNormalCrossX3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionNormalCrossY3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> uxnz, uznx;
public:
    ExpressionNormalCrossY3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionNormalCrossZ3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> uynx, uxny;
public:
    ExpressionNormalCrossZ3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionCurl3D {
public:
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionCurl3D(const FunFEM<M> &fh1) : fun(fh1) {}
    std::vector<std::shared_ptr<ExpressionVirtual>> operator()() const;
};

// ============================================================================
// Normal Derivative Expressions
// ============================================================================

class ExpressionDNormal2 : public ExpressionVirtual {
    typedef Mesh2 M;
    const FunFEM<M> &fun;
    const int comp;
    ExpressionFunFEM<M> dxux, dyuy;
public:
    ExpressionDNormal2(const FunFEM<M> &fh1, int component);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionDNormal3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    const int comp;
    ExpressionFunFEM<M> dxux, dyuy, dzuz;
public:
    ExpressionDNormal3(const FunFEM<M> &fh1, int component);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// Replace the existing dnormal template declaration with these overloads:
std::list<std::shared_ptr<ExpressionVirtual>> dnormal(const FunFEM<Mesh2>& uh);
std::list<std::shared_ptr<ExpressionVirtual>> dnormal(const FunFEM<Mesh3>& uh);


// ============================================================================
// Surface Operators
// ============================================================================

template <typeMesh M>
class ExpressionDSx2 : public ExpressionVirtual {
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> dxu1, dxu1nxnx, dyu1nxny;
public:
    ExpressionDSx2(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

template <typeMesh M>
class ExpressionDSy2 : public ExpressionVirtual {
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> dxu2, dxu2nxny, dyu2nyny;
public:
    ExpressionDSy2(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

template <typeMesh M>
class ExpressionDivS2 : public ExpressionVirtual {
    const FunFEM<M> &fun;
    const std::shared_ptr<ExpressionDSx2<M>> dx;
    const std::shared_ptr<ExpressionDSy2<M>> dy;
public:
    ExpressionDivS2(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// 3D surface operators
class ExpressionDSx3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> dxu1, dxu1nxnx, dyu1nxny, dzu1nxnz;
public:
    ExpressionDSx3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionDSy3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> dxu2, dxu2nxny, dyu2nyny, dzu2nynz;
public:
    ExpressionDSy3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionDSz3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    ExpressionFunFEM<M> dxu3, dxu3nxnz, dyu3nynz, dzu3nznz;
public:
    ExpressionDSz3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionDivS3 : public ExpressionVirtual {
    typedef Mesh3 M;
    const FunFEM<M> &fun;
    const std::shared_ptr<ExpressionDSx3> dx;
    const std::shared_ptr<ExpressionDSy3> dy;
    const std::shared_ptr<ExpressionDSz3> dz;
public:
    ExpressionDivS3(const FunFEM<M> &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// ============================================================================
// Specialized Expressions
// ============================================================================

class ExpressionAverage : public ExpressionVirtual{
public:
    std::shared_ptr<ExpressionVirtual> fun1;
    const R k1, k2;
    ExpressionAverage(const std::shared_ptr<ExpressionVirtual> &fh1, double kk1, double kk2);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionBurgerFlux : public ExpressionVirtual {
    const ExpressionVirtual &fun1;
public:
    ExpressionBurgerFlux(const ExpressionVirtual &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

class ExpressionNormalBurgerFlux : public ExpressionVirtual {
    const ExpressionVirtual &fun1;
public:
    ExpressionNormalBurgerFlux(const ExpressionVirtual &fh1);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

template <typename M> 
class ExpressionLinearSurfaceTension : public ExpressionVirtual {
    const FunFEM<M> &fun;
    const double sigma0;
    const double beta;
    const double tid;
public:
    ExpressionLinearSurfaceTension(const FunFEM<M> &fh, double ssigma0, double bbeta, double ttid);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

template <typename M> 
class ExpressionNonLinearSurfaceTension : public ExpressionVirtual {
    const FunFEM<M> &fun;
    const double sigma0;
    const double beta;
    const double tid;
public:
    ExpressionNonLinearSurfaceTension(const FunFEM<M> &fh, double ssigma0, double bbeta, double ttid);
    R operator()(long i) const override;
    R eval(const int k, const R *x, const R *normal) const override;
    R eval(const int k, const R *x, const R t, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const override;
    R evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const override;
    int idxElementFromBackMesh(int kb, int dd = 0) const override;
};

// ============================================================================
// Operator Declarations
// ============================================================================

// Arithmetic operators
std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, double cc);
std::shared_ptr<ExpressionMultConst> operator*(double cc, const std::shared_ptr<ExpressionVirtual> &f1);
std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_X &cc);
std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_Y &cc);
std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_Z &cc);
std::shared_ptr<ExpressionMultConst> operator*(const ParameterCutFEM &v, const std::shared_ptr<ExpressionVirtual> &f1);

std::shared_ptr<ExpressionAbs> fabs(const std::shared_ptr<ExpressionVirtual> &f1);
std::shared_ptr<ExpressionProduct> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2);
std::shared_ptr<ExpressionPow> pow(const std::shared_ptr<ExpressionVirtual> &f1, const double nn);
std::shared_ptr<ExpressionPow> operator^(const std::shared_ptr<ExpressionVirtual> &f1, const double nn);
std::shared_ptr<ExpressionPow> sqrt(const std::shared_ptr<ExpressionVirtual> &f1);
std::shared_ptr<ExpressionDivision> operator/(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2);
std::shared_ptr<ExpressionSum> operator+(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2);
std::shared_ptr<ExpressionSum> operator-(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2);

// Vector operators
std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Normal &n);
std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Tangent &n);
std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Conormal &n);
std::shared_ptr<ExpressionNormal3> operator*(const FunFEM<Mesh3> &f1, const Normal &n);

// Cross product
std::list<std::shared_ptr<ExpressionVirtual>> cross(const Normal &n, const FunFEM<Mesh3> &f1);

// Curl
std::vector<std::shared_ptr<ExpressionVirtual>> curl(const FunFEM<Mesh3>& uh);

// Surface operators
std::shared_ptr<ExpressionDSx3> dxS(const FunFEM<Mesh3> &f1);
std::shared_ptr<ExpressionDSy3> dyS(const FunFEM<Mesh3> &f1);
std::shared_ptr<ExpressionDSz3> dzS(const FunFEM<Mesh3> &f1);
std::shared_ptr<ExpressionDivS3> divS(const FunFEM<Mesh3> &f1);

template<typeMesh M> std::shared_ptr<ExpressionDSx2<M>> dxS(const FunFEM<M> &f1);
template<typeMesh M> std::shared_ptr<ExpressionDSy2<M>> dyS(const FunFEM<M> &f1);
template<typeMesh M> std::shared_ptr<ExpressionDivS2<M>> divS(const FunFEM<M> &f1);

// Average and jump
std::shared_ptr<ExpressionAverage> average(const std::shared_ptr<ExpressionVirtual> &fh1, const double kk1 = 0.5, const double kk2 = 0.5);
std::shared_ptr<ExpressionAverage> jump(const std::shared_ptr<ExpressionVirtual> &fh1, const double kk1 = 1, const double kk2 = -1);
template <class Mesh>
std::list<std::shared_ptr<ExpressionAverage>> jump(const FunFEM<Mesh> &fh, double kk1 = 1.0, double kk2 = -1.0);
std::list<std::shared_ptr<ExpressionAverage>> jump(const std::list<std::shared_ptr<ExpressionVirtual>> &exprs,
     const double kk1 = 1.0, const double kk2 = -1.0);

std::shared_ptr<ExpressionAverage> operator*(double c, const ExpressionAverage &fh);
std::shared_ptr<ExpressionAverage> operator*(const ExpressionAverage &fh, double c);

// Burger flux
ExpressionBurgerFlux burgerFlux(const ExpressionVirtual &f1);
ExpressionNormalBurgerFlux burgerFlux(const ExpressionVirtual &f1, const Normal &n);

// Differential operators
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dx(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dy(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dz(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dt(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dxx(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dyy(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dzz(const std::shared_ptr<ExpressionFunFEM<M>> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dxy(const std::shared_ptr<ExpressionFunFEM<M>> &u);

template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dx(const ExpressionFunFEM<M> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dy(const ExpressionFunFEM<M> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dz(const ExpressionFunFEM<M> &u);
template <typename M> std::shared_ptr<ExpressionFunFEM<M>> dt(const ExpressionFunFEM<M> &u);

// ============================================================================
// Type Aliases
// ============================================================================

typedef FunFEM<Mesh2> Fun2_h;
typedef ExpressionFunFEM<Mesh2> Expression2;
typedef FunFEM<Mesh3> Fun3_h;
typedef ExpressionFunFEM<Mesh3> Expression3;

// ============================================================================
// Template Implementation Includes
// ============================================================================

#include "expression.tpp"

#endif