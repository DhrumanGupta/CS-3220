from helpers import bidiagonalize, diagonalize_bidiagonal_matrix
from matrix import Matrix


def main():
    # 1. Take n as an input
    # 2. Read a nxm matrix from the user, each line needs to be read as a list of n floats with spaces

    # Try-catch are not needed, that is not the goal of this.
    n = int(input("Enter the number of rows: "))
    m = int(input("Enter the number of columns: "))
    matrix = [
        [float(x) for x in input(f"Enter the {i+1}th row of the matrix: ").split()]
        for i in range(n)
    ]

    A = Matrix(matrix)

    print("\nOriginal matrix A:")
    print(A)

    # Perform bidiagonalization
    U, B, V = bidiagonalize(A)

    print("\nU (orthogonal, left reflectors):")
    print(U)

    print("\nB (upper bidiagonal):")
    print(B)

    print("\nV (orthogonal, right reflectors):")
    print(V)

    # Verify: U * B * V^T should equal A
    V_T = V.transpose()
    reconstructed = U * B * V_T

    print("\nReconstructed A = U * B * V^T:")
    print(reconstructed)

    # Compute reconstruction error
    diff = A - reconstructed
    max_error = max(abs(diff.matrix[i][j]) for i in range(n) for j in range(m))
    print(f"\nMax reconstruction error: {max_error:.2e}")


def test_svd(A: Matrix):
    """Test function to verify SVD decomposition with a sample matrix."""
    print("=" * 60)
    print("Testing bidiagonalization...")
    print("=" * 60)

    print("\nOriginal matrix A (m x n):")
    print(A)

    U, B, V = bidiagonalize(A)

    print("\nU (m x m orthogonal matrix):")
    print(U)

    print("\nB (m x n upper bidiagonal matrix):")
    print(B)

    print("\nV (n x n orthogonal matrix):")
    print(V)

    # Verify reconstruction
    V_T = V.transpose()
    reconstructed = U * B * V_T

    print("\nReconstructed A = U * B * V^T:")
    print(reconstructed)

    # Check error
    rows, cols = A.size
    diff = A - reconstructed
    max_error = max(abs(diff.matrix[i][j]) for i in range(rows) for j in range(cols))
    print(f"\nMax reconstruction error: {max_error:.2e}")

    # Verify U is orthogonal: U^T * U = I
    U_T = U.transpose()
    UTU = U_T * U
    print("\nU^T * U (should be identity):")
    print(UTU)

    # Verify V is orthogonal: V^T * V = I
    VTV = V_T * V
    print("\nV^T * V (should be identity):")
    print(VTV)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    # Diagonalize the bidiagonal matrix B (handles rectangular m x n matrices)
    # Returns U_b (m x m), D (m x n), V_b (n x n)
    U_b, D, V_b = diagonalize_bidiagonal_matrix(B, sort=True)

    # Compute the full SVD decomposition: A = U * B * V^T = U * (U_b * D * V_b^T) * V^T
    #                                       = (U * U_b) * D * (V * V_b)^T
    U_final = U * U_b
    V_final = V * V_b

    # Verify the full SVD decomposition
    reconstructed = U_final * D * V_final.transpose()
    diff = A - reconstructed
    max_error = max(
        abs(diff.matrix[i][j]) for i in range(A.rows) for j in range(A.cols)
    )

    print("\n" + "=" * 60)
    print("SVD decomposition: A = U * D * V^T")
    print("=" * 60)
    print("\nU (m x m orthogonal):")
    print(U_final)
    print("\nD (m x n diagonal with singular values):")
    print(D)
    print("\nV (n x n orthogonal):")
    print(V_final)
    print("\nReconstructed A = U * D * V^T:")
    print(reconstructed)
    print(f"\nMax reconstruction error: {max_error:.2e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        A = Matrix([[1, 2, 3], [4, 5, 6]])
        test_svd(A)
    else:
        main()
