using DelimitedFiles, SparseArrays, LinearAlgebra, Printf

for i in 0:3
    mat = readdlm("build/"*"mat"*string(i)*"Cut.dat");
    # mat = readdlm("build/"*"mat"*string(i)*"Cut.dat");

    # Create sparse matrix
    row_indices = Int64.(mat[:, 1])  # Convert to Int64
    col_indices = Int64.(mat[:, 2])  # Convert to Int64
    values = mat[:, 3]               # Keep values as they are
    sparse_mat = sparse(row_indices, col_indices, values)
    # sparse_mat = sparse(mat[:, 1], mat[:, 2], mat[:, 3]);

    # Compute the condition number of the sparse matrix
    cond_nbr = cond(sparse_mat,1);    @printf "%.2E" cond_nbr
    @printf "\n"
end
