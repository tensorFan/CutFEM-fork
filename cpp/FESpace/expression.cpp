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

#include "expression.hpp"
#include "../problem/CutFEM_parameter.hpp"

// ============================================================================
// ExpressionMultConst Implementation
// ============================================================================

ExpressionMultConst::ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const double &cc)
    : fun1(fh1), c(cc), nx(false), ny(false), nz(false), p(R2(1, 1)) {}

ExpressionMultConst::ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_X &nnx)
    : fun1(fh1), c(1.), nx(true), ny(false), nz(false), p(R2(1, 1)) {}

ExpressionMultConst::ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_Y &nny)
    : fun1(fh1), c(1.), nx(false), ny(true), nz(false), p(R2(1, 1)) {}

ExpressionMultConst::ExpressionMultConst(const std::shared_ptr<ExpressionVirtual> &fh1, const Normal_Component_Z &nnz)
    : fun1(fh1), c(1.), nx(false), ny(false), nz(true), p(R2(1, 1)) {}

R ExpressionMultConst::operator()(long i) const { return c * ((*fun1)(i)); }

R ExpressionMultConst::eval(const int k, const R *x, const R *normal) const {
    double compN = ((nx) ? normal[0] : 1) * ((ny) ? normal[1] : 1) * ((nz) ? normal[2] : 1);
    return fun1->eval(k, x, normal) * c * compN * p[0];
}

R ExpressionMultConst::eval(const int k, const R *x, const R t, const R *normal) const {
    double compN = ((nx) ? normal[0] : 1) * ((ny) ? normal[1] : 1) * ((nz) ? normal[2] : 1);
    return fun1->eval(k, x, t, normal) * c * compN * p[0];
}

R ExpressionMultConst::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    double compN = ((nx) ? normal[0] : 1) * ((ny) ? normal[1] : 1) * ((nz) ? normal[2] : 1);
    return fun1->evalOnBackMesh(k, dom, x, normal) * c * compN * p[dom == 1];
}

R ExpressionMultConst::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    double compN = ((nx) ? normal[0] : 1) * ((ny) ? normal[1] : 1) * ((nz) ? normal[2] : 1);
    return fun1->evalOnBackMesh(k, dom, x, t, normal) * c * compN * p[dom == 1];
}

int ExpressionMultConst::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// ExpressionAbs Implementation
// ============================================================================

ExpressionAbs::ExpressionAbs(const std::shared_ptr<ExpressionVirtual> &fh1) : fun1(fh1) {}

R ExpressionAbs::operator()(long i) const { return fabs((*fun1)(i)); }

R ExpressionAbs::eval(const int k, const R *x, const R *normal) const { 
    return fabs(fun1->eval(k, x, normal)); 
}

R ExpressionAbs::eval(const int k, const R *x, const R t, const R *normal) const { 
    return fabs(fun1->eval(k, x, t, normal)); 
}

R ExpressionAbs::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    return fabs(fun1->evalOnBackMesh(k, dom, x, normal));
}

R ExpressionAbs::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    return fabs(fun1->evalOnBackMesh(k, dom, x, t, normal));
}

int ExpressionAbs::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// ExpressionProduct Implementation
// ============================================================================

ExpressionProduct::ExpressionProduct(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2)
    : fun1(fh1), fun2(fh2) {}

R ExpressionProduct::operator()(long i) const { return (*fun1)(i) * (*fun2)(i); }

R ExpressionProduct::eval(const int k, const R *x, const R *normal) const {
    return fun1->eval(k, x, normal) * fun2->eval(k, x, normal);
}

R ExpressionProduct::eval(const int k, const R *x, const R t, const R *normal) const {
    return fun1->eval(k, x, t, normal) * fun2->eval(k, x, t, normal);
}

R ExpressionProduct::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    return fun1->evalOnBackMesh(k, dom, x, normal) * fun2->evalOnBackMesh(k, dom, x, normal);
}

R ExpressionProduct::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    return fun1->evalOnBackMesh(k, dom, x, t, normal) * fun2->evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionProduct::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// ExpressionPow Implementation
// ============================================================================

ExpressionPow::ExpressionPow(const ptr_expr_t &f1h, const double nn) : fun1(f1h), n(nn) {}

R ExpressionPow::operator()(long i) const { return pow((*fun1)(i), n); }

R ExpressionPow::eval(const int k, const R *x, const R *normal) const {
    const double val = fun1->eval(k, x, normal);
    return pow(val, n);
}

R ExpressionPow::eval(const int k, const R *x, const R t, const R *normal) const {
    const double val = fun1->eval(k, x, t, normal);
    return pow(val, n);
}

R ExpressionPow::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    const double val = fun1->evalOnBackMesh(k, dom, x, normal);
    return pow(val, n);
}

R ExpressionPow::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    const double val = fun1->evalOnBackMesh(k, dom, x, t, normal);
    return pow(val, n);
}

int ExpressionPow::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// ExpressionDivision Implementation
// ============================================================================

ExpressionDivision::ExpressionDivision(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2)
    : fun1(fh1), fun2(fh2) {}

R ExpressionDivision::operator()(long i) const { return (*fun1)(i) / ((*fun2)(i)); }

R ExpressionDivision::eval(const int k, const R *x, const R *normal) const {
    double v = fun2->eval(k, x, normal);
    assert(fabs(v) > 1e-15);
    return fun1->eval(k, x, normal) / v;
}

R ExpressionDivision::eval(const int k, const R *x, const R t, const R *normal) const {
    double v = fun2->eval(k, x, t, normal);
    assert(fabs(v) > 1e-15);
    return fun1->eval(k, x, t, normal) / v;
}

R ExpressionDivision::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    double v = fun2->evalOnBackMesh(k, dom, x, normal);
    assert(fabs(v) > 1e-15);
    return fun1->evalOnBackMesh(k, dom, x, normal) / v;
}

R ExpressionDivision::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    double v = fun2->evalOnBackMesh(k, dom, x, t, normal);
    assert(fabs(v) > 1e-15);
    return fun1->evalOnBackMesh(k, dom, x, t, normal) / v;
}

int ExpressionDivision::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// ExpressionSum Implementation
// ============================================================================

ExpressionSum::ExpressionSum(const std::shared_ptr<ExpressionVirtual> &fh1, const std::shared_ptr<ExpressionVirtual> &fh2)
    : fun1(fh1), fun2(fh2) {}

R ExpressionSum::operator()(long i) const { return (*fun1)(i) + (*fun2)(i); }

R ExpressionSum::eval(const int k, const R *x, const R *normal) const {
    return fun1->eval(k, x, normal) + fun2->eval(k, x, normal);
}

R ExpressionSum::eval(const int k, const R *x, const R t, const R *normal) const {
    return fun1->eval(k, x, t, normal) + fun2->eval(k, x, t, normal);
}

R ExpressionSum::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    return fun1->evalOnBackMesh(k, dom, x, normal) + fun2->evalOnBackMesh(k, dom, x, normal);
}

R ExpressionSum::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    return fun1->evalOnBackMesh(k, dom, x, t, normal) + fun2->evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionSum::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// Vector Calculus Expressions Implementation
// ============================================================================

ExpressionNormal2::ExpressionNormal2(const FunFEM<M> &fh1, const Normal n)
    : fun(fh1), uxnx(fh1, 0, op_id, 0, 0), uyny(fh1, 1, op_id, 0, 0) {
    assert(fh1.Vh->N != 1);
    uxnx.addNormal(0);
    uyny.addNormal(1);
}

ExpressionNormal2::ExpressionNormal2(const FunFEM<M> &fh1, const Tangent t)
    : fun(fh1), uxnx(fh1, 0, op_id, 0, 0), uyny(fh1, 1, op_id, 0, 0) {
    assert(fh1.Vh->N != 1);
    uxnx.addNormal(1);
    uyny.addNormal(0);
    c0 = -1;
}

ExpressionNormal2::ExpressionNormal2(const FunFEM<M> &fh1, const Conormal n)
    : fun(fh1), uxnx(fh1, 0, op_id, 0, 0), uyny(fh1, 1, op_id, 0, 0) {
    assert(fh1.Vh->N != 1);
    uxnx.addNormal(0);
    uyny.addNormal(1);
}

R ExpressionNormal2::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionNormal2::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating f*n expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionNormal2::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating f*n expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionNormal2::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return c0 * uxnx.evalOnBackMesh(k, dom, x, normal) + uyny.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionNormal2::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return c0 * uxnx.evalOnBackMesh(k, dom, x, t, normal) + uyny.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionNormal2::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionNormal3::ExpressionNormal3(const FunFEM<M> &fh1)
    : fun(fh1), uxnx(fh1, 0, op_id, 0, 0), uyny(fh1, 1, op_id, 0, 0), uznz(fh1, 2, op_id, 0, 0) {
    assert(fh1.Vh->N != 1);
    uxnx.addNormal(0);
    uyny.addNormal(1);
    uznz.addNormal(2);
}

R ExpressionNormal3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionNormal3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating f*n expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionNormal3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating f*n expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionNormal3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return uxnx.evalOnBackMesh(k, dom, x, normal) + uyny.evalOnBackMesh(k, dom, x, normal) +
           uznz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionNormal3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return uxnx.evalOnBackMesh(k, dom, x, t, normal) + uyny.evalOnBackMesh(k, dom, x, t, normal) +
           uznz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionNormal3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// Cross Product Expressions Implementation
// ============================================================================

ExpressionNormalCrossX3::ExpressionNormalCrossX3(const FunFEM<M> &fh1) 
    : fun(fh1), uynz(fh1, 1, op_id, 0, 0), uzny(fh1, 2, op_id, 0, 0) {
    assert(fh1.Vh->N == 3);
    uzny.addNormal(1);
    uynz.addNormal(2);
}

R ExpressionNormalCrossX3::operator()(long i) const { 
    assert(0);
    return 0;
}

R ExpressionNormalCrossX3::eval(const int k, const R *x, const R *normal) const {
    return uzny.eval(k, x, normal) - uynz.eval(k, x, normal);
}

R ExpressionNormalCrossX3::eval(const int k, const R *x, const R t, const R *normal) const {
    return uzny.eval(k, x, t, normal) - uynz.eval(k, x, t, normal);
}

R ExpressionNormalCrossX3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return uzny.evalOnBackMesh(k, dom, x, normal) - uynz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionNormalCrossX3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return uzny.evalOnBackMesh(k, dom, x, t, normal) - uynz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionNormalCrossX3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionNormalCrossY3::ExpressionNormalCrossY3(const FunFEM<M> &fh1) 
    : fun(fh1), uxnz(fh1, 0, op_id, 0, 0), uznx(fh1, 2, op_id, 0, 0) {
    assert(fh1.Vh->N == 3);
    uznx.addNormal(0);
    uxnz.addNormal(2);
}

R ExpressionNormalCrossY3::operator()(long i) const { 
    assert(0);
    return 0;
}

R ExpressionNormalCrossY3::eval(const int k, const R *x, const R *normal) const {
    return uxnz.eval(k, x, normal) - uznx.eval(k, x, normal);
}

R ExpressionNormalCrossY3::eval(const int k, const R *x, const R t, const R *normal) const {
    return uxnz.eval(k, x, t, normal) - uznx.eval(k, x, t, normal);
}

R ExpressionNormalCrossY3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return uxnz.evalOnBackMesh(k, dom, x, normal) - uznx.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionNormalCrossY3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return uxnz.evalOnBackMesh(k, dom, x, t, normal) - uznx.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionNormalCrossY3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionNormalCrossZ3::ExpressionNormalCrossZ3(const FunFEM<M> &fh1) 
    : fun(fh1), uynx(fh1, 1, op_id, 0, 0), uxny(fh1, 0, op_id, 0, 0) {
    assert(fh1.Vh->N == 3);
    uynx.addNormal(0);
    uxny.addNormal(1);
}

R ExpressionNormalCrossZ3::operator()(long i) const { 
    assert(0);
    return 0;
}

R ExpressionNormalCrossZ3::eval(const int k, const R *x, const R *normal) const {
    return uynx.eval(k, x, normal) - uxny.eval(k, x, normal);
}

R ExpressionNormalCrossZ3::eval(const int k, const R *x, const R t, const R *normal) const {
    return uynx.eval(k, x, t, normal) - uxny.eval(k, x, t, normal);
}

R ExpressionNormalCrossZ3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return uynx.evalOnBackMesh(k, dom, x, normal) - uxny.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionNormalCrossZ3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return uynx.evalOnBackMesh(k, dom, x, t, normal) - uxny.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionNormalCrossZ3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// 3D Surface Operators Implementation
// ============================================================================

ExpressionDSx3::ExpressionDSx3(const FunFEM<M> &fh1)
    : fun(fh1), dxu1(fh1, 0, op_dx, 0, 0), dxu1nxnx(fh1, 0, op_dx, 0, 0), 
      dyu1nxny(fh1, 0, op_dy, 0, 0), dzu1nxnz(fh1, 0, op_dz, 0, 0) {
    dxu1nxnx.addNormal(0);
    dxu1nxnx.addNormal(0);
    dyu1nxny.addNormal(0);
    dyu1nxny.addNormal(1);
    dzu1nxnz.addNormal(0);
    dzu1nxnz.addNormal(2);
}

R ExpressionDSx3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDSx3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DSx expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSx3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DSx expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSx3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxu1.evalOnBackMesh(k, dom, x, normal) - dxu1nxnx.evalOnBackMesh(k, dom, x, normal) -
           dyu1nxny.evalOnBackMesh(k, dom, x, normal) - dzu1nxnz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDSx3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxu1.evalOnBackMesh(k, dom, x, t, normal) - dxu1nxnx.evalOnBackMesh(k, dom, x, t, normal) -
           dyu1nxny.evalOnBackMesh(k, dom, x, t, normal) - dzu1nxnz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDSx3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionDSy3::ExpressionDSy3(const FunFEM<M> &fh1)
    : fun(fh1), dxu2(fh1, 1, op_dy, 0, 0), dxu2nxny(fh1, 1, op_dx, 0, 0), 
      dyu2nyny(fh1, 1, op_dy, 0, 0), dzu2nynz(fh1, 1, op_dz, 0, 0) {
    dxu2nxny.addNormal(0);
    dxu2nxny.addNormal(1);
    dyu2nyny.addNormal(1);
    dyu2nyny.addNormal(1);
    dzu2nynz.addNormal(1);
    dzu2nynz.addNormal(2);
}

R ExpressionDSy3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDSy3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DSy expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSy3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DSy expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSy3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxu2.evalOnBackMesh(k, dom, x, normal) - dxu2nxny.evalOnBackMesh(k, dom, x, normal) -
           dyu2nyny.evalOnBackMesh(k, dom, x, normal) - dzu2nynz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDSy3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxu2.evalOnBackMesh(k, dom, x, t, normal) - dxu2nxny.evalOnBackMesh(k, dom, x, t, normal) -
           dyu2nyny.evalOnBackMesh(k, dom, x, t, normal) - dzu2nynz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDSy3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionDSz3::ExpressionDSz3(const FunFEM<M> &fh1)
    : fun(fh1), dxu3(fh1, 2, op_dz, 0, 0), dxu3nxnz(fh1, 2, op_dx, 0, 0), 
      dyu3nynz(fh1, 2, op_dy, 0, 0), dzu3nznz(fh1, 2, op_dz, 0, 0) {
    dxu3nxnz.addNormal(0);
    dxu3nxnz.addNormal(2);
    dyu3nynz.addNormal(1);
    dyu3nynz.addNormal(2);
    dzu3nznz.addNormal(2);
    dzu3nznz.addNormal(2);
}

R ExpressionDSz3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDSz3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DSz expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSz3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DSz expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDSz3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxu3.evalOnBackMesh(k, dom, x, normal) - dxu3nxnz.evalOnBackMesh(k, dom, x, normal) -
           dyu3nynz.evalOnBackMesh(k, dom, x, normal) - dzu3nznz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDSz3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxu3.evalOnBackMesh(k, dom, x, t, normal) - dxu3nxnz.evalOnBackMesh(k, dom, x, t, normal) -
           dyu3nynz.evalOnBackMesh(k, dom, x, t, normal) - dzu3nznz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDSz3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

ExpressionDivS3::ExpressionDivS3(const FunFEM<M> &fh1) 
    : fun(fh1), dx(dxS(fh1)), dy(dyS(fh1)), dz(dzS(fh1)) {}

R ExpressionDivS3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDivS3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating DivS expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDivS3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating DivS expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDivS3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dx->evalOnBackMesh(k, dom, x, normal) + dy->evalOnBackMesh(k, dom, x, normal) +
           dz->evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDivS3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dx->evalOnBackMesh(k, dom, x, t, normal) + dy->evalOnBackMesh(k, dom, x, t, normal) +
           dz->evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDivS3::idxElementFromBackMesh(int kb, int dd) const { 
    return fun.idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// Specialized Expressions Implementation
// ============================================================================

ExpressionAverage::ExpressionAverage(const std::shared_ptr<ExpressionVirtual> &fh1, double kk1, double kk2)
    : fun1(fh1), k1(kk1), k2(kk2) {}

R ExpressionAverage::operator()(long i) const {
    assert(0 && " cannot use this ");
    return k1 * (*fun1)(i);
}

R ExpressionAverage::eval(const int k, const R *x, const R *normal) const {
    assert(0 && " need to be evaluated on backmesh");
    return fun1->eval(k, x, normal) + fun1->eval(k, x, normal);
}

R ExpressionAverage::eval(const int k, const R *x, const R t, const R *normal) const {
    assert(0 && " need to be evaluated on backmesh");
    return fun1->eval(k, x, t, normal) + fun1->eval(k, x, t, normal);
}

R ExpressionAverage::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    return k1 * fun1->evalOnBackMesh(k, 0, x, normal) + k2 * fun1->evalOnBackMesh(k, 1, x, normal);
}

R ExpressionAverage::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    return k1 * fun1->evalOnBackMesh(k, 0, x, t, normal) + k2 * fun1->evalOnBackMesh(k, 1, x, t, normal);
}

int ExpressionAverage::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1->idxElementFromBackMesh(kb, dd); 
}

ExpressionBurgerFlux::ExpressionBurgerFlux(const ExpressionVirtual &fh1) : fun1(fh1) {}

R ExpressionBurgerFlux::operator()(long i) const { return fabs(fun1(i)); }

R ExpressionBurgerFlux::eval(const int k, const R *x, const R *normal) const {
    double val = fun1.eval(k, x, normal);
    return 0.5 * val * val;
}

R ExpressionBurgerFlux::eval(const int k, const R *x, const R t, const R *normal) const {
    double val = fun1.eval(k, x, t, normal);
    return 0.5 * val * val;
}

R ExpressionBurgerFlux::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    double val = fun1.evalOnBackMesh(k, dom, x, normal);
    return 0.5 * val * val;
}

R ExpressionBurgerFlux::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    double val = fun1.evalOnBackMesh(k, dom, x, t, normal);
    return 0.5 * val * val;
}

int ExpressionBurgerFlux::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1.idxElementFromBackMesh(kb, dd); 
}

ExpressionNormalBurgerFlux::ExpressionNormalBurgerFlux(const ExpressionVirtual &fh1) : fun1(fh1) {}

R ExpressionNormalBurgerFlux::operator()(long i) const { return fabs(fun1(i)); }

R ExpressionNormalBurgerFlux::eval(const int k, const R *x, const R *normal) const {
    assert(normal);
    double val = fun1.eval(k, x, normal);
    return 0.5 * val * val * (normal[0] + normal[1]);
}

R ExpressionNormalBurgerFlux::eval(const int k, const R *x, const R t, const R *normal) const {
    assert(normal);
    double val = fun1.eval(k, x, t, normal);
    return 0.5 * val * val * (normal[0] + normal[1]);
}

R ExpressionNormalBurgerFlux::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    double val = fun1.evalOnBackMesh(k, dom, x, normal);
    return 0.5 * val * val * (normal[0] + normal[1]);
}

R ExpressionNormalBurgerFlux::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    double val = fun1.evalOnBackMesh(k, dom, x, t, normal);
    return 0.5 * val * val * (normal[0] + normal[1]);
}

int ExpressionNormalBurgerFlux::idxElementFromBackMesh(int kb, int dd) const { 
    return fun1.idxElementFromBackMesh(kb, dd); 
}

// ============================================================================
// Operator Implementations
// ============================================================================

std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, double cc) {
    return std::make_shared<ExpressionMultConst>(f1, cc);
}

std::shared_ptr<ExpressionMultConst> operator*(double cc, const std::shared_ptr<ExpressionVirtual> &f1) {
    return std::make_shared<ExpressionMultConst>(f1, cc);
}

std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_X &cc) {
    return std::make_shared<ExpressionMultConst>(f1, cc);
}

std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_Y &cc) {
    return std::make_shared<ExpressionMultConst>(f1, cc);
}

std::shared_ptr<ExpressionMultConst> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const Normal_Component_Z &cc) {
    return std::make_shared<ExpressionMultConst>(f1, cc);
}

std::shared_ptr<ExpressionAbs> fabs(const std::shared_ptr<ExpressionVirtual> &f1) {
    return std::make_shared<ExpressionAbs>(f1);
}

std::shared_ptr<ExpressionProduct> operator*(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2) {
    return std::make_shared<ExpressionProduct>(f1, f2);
}

std::shared_ptr<ExpressionPow> pow(const std::shared_ptr<ExpressionVirtual> &f1, const double nn) {
    return std::make_shared<ExpressionPow>(f1, nn);
}

std::shared_ptr<ExpressionPow> operator^(const std::shared_ptr<ExpressionVirtual> &f1, const double nn) {
    return std::make_shared<ExpressionPow>(f1, nn);
}

std::shared_ptr<ExpressionPow> sqrt(const std::shared_ptr<ExpressionVirtual> &f1) { 
    return pow(f1, 1. / 2); 
}

std::shared_ptr<ExpressionDivision> operator/(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2) {
    return std::make_shared<ExpressionDivision>(f1, f2);
}

std::shared_ptr<ExpressionSum> operator+(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2) {
    return std::make_shared<ExpressionSum>(f1, f2);
}

std::shared_ptr<ExpressionSum> operator-(const std::shared_ptr<ExpressionVirtual> &f1, const std::shared_ptr<ExpressionVirtual> &f2) {
    return f1 + (-1. * f2);
}

std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Normal &n) {
    return std::make_shared<ExpressionNormal2>(f1, n);
}

std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Tangent &n) {
    return std::make_shared<ExpressionNormal2>(f1, n);
}

std::shared_ptr<ExpressionNormal2> operator*(const FunFEM<Mesh2> &f1, const Conormal &n) {
    return std::make_shared<ExpressionNormal2>(f1, n);
}

std::shared_ptr<ExpressionNormal3> operator*(const FunFEM<Mesh3> &f1, const Normal &n) {
    return std::make_shared<ExpressionNormal3>(f1);
}

std::shared_ptr<ExpressionAverage> average(const std::shared_ptr<ExpressionVirtual> &f1, const double kk1, const double kk2) {
    return std::make_shared<ExpressionAverage>(f1, kk1, kk2);
}

std::shared_ptr<ExpressionAverage> jump(const std::shared_ptr<ExpressionVirtual> &f1, const double kk1, const double kk2) {
    return std::make_shared<ExpressionAverage>(f1, 1, -1);
}
std::list<std::shared_ptr<ExpressionAverage>>
jump(const std::list<std::shared_ptr<ExpressionVirtual>> &exprs,
     const double kk1, const double kk2) {
    std::list<std::shared_ptr<ExpressionAverage>> res;
    for (auto &expr : exprs) {
        res.push_back(jump(expr, kk1, kk2));  // uses the scalar overload
    }
    return res;
}


std::shared_ptr<ExpressionAverage> operator*(double c, const ExpressionAverage &fh) {
    return std::make_shared<ExpressionAverage>(fh.fun1, c * fh.k1, c * fh.k2);
}

std::shared_ptr<ExpressionAverage> operator*(const ExpressionAverage &fh, double c) {
    return std::make_shared<ExpressionAverage>(fh.fun1, c * fh.k1, c * fh.k2);
}

std::list<std::shared_ptr<ExpressionVirtual>> cross(const Normal &n, const FunFEM<Mesh3> &f1) {
    return {std::make_shared<ExpressionNormalCrossX3>(f1), 
            std::make_shared<ExpressionNormalCrossY3>(f1),
            std::make_shared<ExpressionNormalCrossZ3>(f1)};
}

std::vector<std::shared_ptr<ExpressionVirtual>> ExpressionCurl3D::operator()() const {
    return {
        dy(fun.expr(2)) - dz(fun.expr(1)),  // d/dy(u_z) - d/dz(u_y)
        dz(fun.expr(0)) - dx(fun.expr(2)),  // d/dz(u_x) - d/dx(u_z)
        dx(fun.expr(1)) - dy(fun.expr(0))   // d/dx(u_y) - d/dy(u_x)
    };
}

ExpressionBurgerFlux burgerFlux(const ExpressionVirtual &f1) { 
    return ExpressionBurgerFlux(f1); 
}

ExpressionNormalBurgerFlux burgerFlux(const ExpressionVirtual &f1, const Normal &n) {
    return ExpressionNormalBurgerFlux(f1);
}

std::shared_ptr<ExpressionDSx3> dxS(const FunFEM<Mesh3> &f1) { 
    return std::make_shared<ExpressionDSx3>(f1); 
}

std::shared_ptr<ExpressionDSy3> dyS(const FunFEM<Mesh3> &f1) { 
    return std::make_shared<ExpressionDSy3>(f1); 
}

std::shared_ptr<ExpressionDSz3> dzS(const FunFEM<Mesh3> &f1) { 
    return std::make_shared<ExpressionDSz3>(f1); 
}

std::shared_ptr<ExpressionDivS3> divS(const FunFEM<Mesh3> &f1) { 
    return std::make_shared<ExpressionDivS3>(f1); 
}

// ============================================================================
// Normal Derivative Expressions Implementation
// ============================================================================

ExpressionDNormal2::ExpressionDNormal2(const FunFEM<M> &fh1, int component)
    : fun(fh1), comp(component), dxux(fh1, component, op_dx, 0, 0), dyuy(fh1, component, op_dy, 0, 0) {
    assert(component < fh1.Vh->N);
    dxux.addNormal(0);
    dyuy.addNormal(1);
}

R ExpressionDNormal2::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDNormal2::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating dnormal expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDNormal2::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating dnormal expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDNormal2::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxux.evalOnBackMesh(k, dom, x, normal) + dyuy.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDNormal2::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxux.evalOnBackMesh(k, dom, x, t, normal) + dyuy.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDNormal2::idxElementFromBackMesh(int kb, int dd) const {
    return fun.idxElementFromBackMesh(kb, dd);
}

ExpressionDNormal3::ExpressionDNormal3(const FunFEM<M> &fh1, int component)
    : fun(fh1), comp(component), dxux(fh1, component, op_dx, 0, 0), 
      dyuy(fh1, component, op_dy, 0, 0), dzuz(fh1, component, op_dz, 0, 0) {
    assert(component < fh1.Vh->N);
    dxux.addNormal(0);
    dyuy.addNormal(1);
    dzuz.addNormal(2);
}

R ExpressionDNormal3::operator()(long i) const {
    assert(0);
    return 0;
}

R ExpressionDNormal3::eval(const int k, const R *x, const R *normal) const {
    std::cout << " evaluating dnormal expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDNormal3::eval(const int k, const R *x, const R t, const R *normal) const {
    std::cout << " evaluating dnormal expression without giving the normal as input " << std::endl;
    assert(0);
    return 0;
}

R ExpressionDNormal3::evalOnBackMesh(const int k, const int dom, const R *x, const R *normal) const {
    assert(normal);
    return dxux.evalOnBackMesh(k, dom, x, normal) + dyuy.evalOnBackMesh(k, dom, x, normal) +
           dzuz.evalOnBackMesh(k, dom, x, normal);
}

R ExpressionDNormal3::evalOnBackMesh(const int k, const int dom, const R *x, const R t, const R *normal) const {
    assert(normal);
    return dxux.evalOnBackMesh(k, dom, x, t, normal) + dyuy.evalOnBackMesh(k, dom, x, t, normal) +
           dzuz.evalOnBackMesh(k, dom, x, t, normal);
}

int ExpressionDNormal3::idxElementFromBackMesh(int kb, int dd) const {
    return fun.idxElementFromBackMesh(kb, dd);
}

// Factory functions
std::list<std::shared_ptr<ExpressionVirtual>> dnormal(const FunFEM<Mesh2>& uh) {
    std::list<std::shared_ptr<ExpressionVirtual>> res;
    for (int i = 0; i < uh.Vh->N; ++i) {
        res.push_back(std::make_shared<ExpressionDNormal2>(uh, i));
    }
    return res;
}

std::list<std::shared_ptr<ExpressionVirtual>> dnormal(const FunFEM<Mesh3>& uh) {
    std::list<std::shared_ptr<ExpressionVirtual>> res;
    for (int i = 0; i < uh.Vh->N; ++i) {
        res.push_back(std::make_shared<ExpressionDNormal3>(uh, i));
    }
    return res;
}