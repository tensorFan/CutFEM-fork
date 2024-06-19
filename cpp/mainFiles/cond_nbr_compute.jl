using DelimitedFiles, SparseArrays, LinearAlgebra, Printf

for i in 0:4
    mat = readdlm("build/"*"mat"*string(i)*"Cut.dat");
    # mat = readdlm("build/"*"mat"*string(i)*"Cut.dat");

    # Create sparse matrix from mat0Cut
    sparse_mat = sparse(mat[:, 1], mat[:, 2], mat[:, 3]);

    # Compute the condition number of the sparse matrix
    cond_nbr = cond(sparse_mat,1);    @printf "%.2E" cond_nbr
    @printf "\n"
end
