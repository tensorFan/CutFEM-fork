using DelimitedFiles, SparseArrays, LinearAlgebra, Printf

using ArnoldiMethod, LinearMaps
# using KrylovKit
# using NonlinearEigenproblems
# using JacobiDavidson

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A, B)
    a = ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A, 1)))
    LinearMap{eltype(A)}(a, size(A, 1), ismutating = true)
end

A = 0
for i in 0:0
    mat = readdlm("build/"*"mat"*string(i)*".dat");
    matrhs = readdlm("build/"*"mat"*string(i)*"RHS.dat");

    # mat = readdlm("build/"*"mat"*string(i)*"Cut.dat");
    # matrhs = readdlm("build/"*"mat"*string(i)*"CutRHS.dat");

    # Create sparse matrix
    A = sparse(mat[:, 1], mat[:, 2], mat[:, 3]);
    B = sparse(matrhs[:, 1], matrhs[:, 2], matrhs[:, 3]);
    # print(size(A))
    # print(size(B))

    # Compute the eigenvalues of Ax=λBx
    # Target the largest eigenvalues of the inverted problem
    decomp, = partialschur(
        construct_linear_map(A, B),
        nev = 4, # nbr of eigenvalues
        tol = 1e-6,
        restarts = 200,
        which = LM(),
    )
    λs_inv, X = partialeigen(decomp)

    # Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
    λs = 1 ./ λs_inv
    print(sort(abs.(λs)))

    @printf "\n"
end
