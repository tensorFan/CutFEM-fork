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

num_nodes = [343, 2197, 15625];
A = 0;
eig_list = 0;
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
    diagC = diagm(diag(A+B));
    A = diagC \ A;
    B = diagC \ B;

    ## Compute the eigenvalues of Ax=λBx
    # Target the largest eigenvalues of the inverted problem, then invert to find smallest of non-inverted problem.
    decomp, = partialschur(
        construct_linear_map(A, B),
        nev = num_nodes[i+1]+20+15, # For standard wave formulation: nbr of NONZERO eigenvalues (remove the first #dim_lagrange eigenvalues)
        # nev = size(A,1), 
        tol = 1e-16,
        restarts = fld(size(A,1), 2),
        # restarts = size(A,1),
        which = LM(),
    )
    λs_inv, X = partialeigen(decomp)
    λs = 1 ./ λs_inv
    λs = abs.(λs)

    λs = λs[ λs .> 10^(-10)] # sort out the eigenvalues that are too close to zero
    eig_list = sort( λs )
    print((eig_list/eig_list[1]).^2)

    @printf "\n"
end

