"""
Unit Tests for Bidiagonal Matrix Diagonalization

This module contains comprehensive tests for the SVD implementation,
including tests for Givens rotations, bidiagonal diagonalization,
and integration with the bidiagonalize function.

Run with: python tests.py
"""

import random

from helpers import (
    bidiagonalize,
    check_orthogonality,
    diagonalize_bidiagonal_matrix,
    givens,
)
from matrix import Matrix


def make_bidiagonal(diag: list[float], superdiag: list[float]) -> Matrix:
    """Create a bidiagonal matrix from diagonal and superdiagonal lists."""
    n = len(diag)
    data = [[0.0] * n for _ in range(n)]
    for i in range(n):
        data[i][i] = diag[i]
    for i in range(n - 1):
        data[i][i + 1] = superdiag[i]
    return Matrix(data)


def max_off_diagonal(M: Matrix) -> float:
    """Return the maximum absolute value of off-diagonal elements."""
    max_val = 0.0
    for i in range(M.rows):
        for j in range(M.cols):
            if i != j:
                max_val = max(max_val, abs(M.matrix[i][j]))
    return max_val


def test_givens():
    """Test the Givens rotation computation."""
    print("=" * 60)
    print("Test 1: Givens rotation computation")
    print("=" * 60)

    # Test case 1: Simple rotation
    # The rotation [[c, s], [-s, c]] * [a, b]^T = [r, 0]^T
    a, b = 3.0, 4.0
    c, s, r = givens(a, b)
    print(f"givens({a}, {b}) = (c={c:.6f}, s={s:.6f}, r={r:.6f})")
    # Check: c*a + s*b = r, -s*a + c*b = 0
    result1 = c * a + s * b
    result2 = -s * a + c * b
    print(f"  c*a + s*b = {result1:.6f} (should be {r:.6f})")
    print(f"  -s*a + c*b = {result2:.6f} (should be 0)")
    assert abs(result1 - r) < 1e-10, "Givens r check failed"
    assert abs(result2) < 1e-10, "Givens zero check failed"

    # Test case 2: Zero second component
    c, s, r = givens(5.0, 0.0)
    print(f"\ngivens(5, 0) = (c={c:.6f}, s={s:.6f}, r={r:.6f})")
    assert abs(c - 1.0) < 1e-10 and abs(s) < 1e-10, "Zero b case failed"

    # Test case 3: Both zero
    c, s, r = givens(0.0, 0.0)
    print(f"givens(0, 0) = (c={c:.6f}, s={s:.6f}, r={r:.6f})")
    assert c == 1.0 and s == 0.0 and r == 0.0, "Both zero case failed"

    print("\n[PASS] Givens rotation tests passed!\n")


def test_small_bidiagonal():
    """Test diagonalization on small bidiagonal matrices."""
    print("=" * 60)
    print("Test 2: Small 3x3 bidiagonal matrix diagonalization")
    print("=" * 60)

    # Create a simple 3x3 bidiagonal matrix
    B = make_bidiagonal([4.0, 3.0, 2.0], [1.0, 0.5])
    print("Original bidiagonal matrix B:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nU_b (left orthogonal matrix):")
    print(U_b)

    print("\nD (diagonal matrix with singular values):")
    print(D)

    print("\nV_b (right orthogonal matrix):")
    print(V_b)

    # Check 1: D should be diagonal
    off_diag = max_off_diagonal(D)
    print(f"\nMax off-diagonal in D: {off_diag:.2e}")
    assert off_diag < 1e-10, f"D is not diagonal! Off-diagonal max: {off_diag}"

    # Check 2: D should have non-negative diagonal
    for i in range(D.rows):
        assert D.matrix[i][i] >= -1e-12, f"Negative singular value at D[{i},{i}]"
    print("All diagonal entries are non-negative: [PASS]")

    # Check 3: U_b should be orthogonal
    U_orth_err = check_orthogonality(U_b)
    print(f"||I - U_b^T U_b||_F = {U_orth_err:.2e}")
    assert U_orth_err < 1e-10, f"U_b is not orthogonal! Error: {U_orth_err}"

    # Check 4: V_b should be orthogonal
    V_orth_err = check_orthogonality(V_b)
    print(f"||I - V_b^T V_b||_F = {V_orth_err:.2e}")
    assert V_orth_err < 1e-10, f"V_b is not orthogonal! Error: {V_orth_err}"

    # Check 5: U_b * D * V_b^T should equal B
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"Max reconstruction error ||B - U_b*D*V_b^T||_max = {max_err:.2e}")
    assert max_err < 1e-10, f"Reconstruction failed! Error: {max_err}"

    print("\n[PASS] 3x3 bidiagonal test passed!\n")


def test_4x4_bidiagonal():
    """Test diagonalization on a 4x4 bidiagonal matrix."""
    print("=" * 60)
    print("Test 3: 4x4 bidiagonal matrix diagonalization")
    print("=" * 60)

    B = make_bidiagonal([5.0, 4.0, 3.0, 2.0], [1.2, 0.8, 0.4])
    print("Original bidiagonal matrix B:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nDiagonal entries of D (singular values):")
    diag_vals = [D.matrix[i][i] for i in range(D.rows)]
    print(f"  {diag_vals}")

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"\nMax reconstruction error: {max_err:.2e}")

    # Verify orthogonality
    U_orth_err = check_orthogonality(U_b)
    V_orth_err = check_orthogonality(V_b)
    print(f"U_b orthogonality error: {U_orth_err:.2e}")
    print(f"V_b orthogonality error: {V_orth_err:.2e}")

    assert max_err < 1e-10, f"Reconstruction failed! Error: {max_err}"
    assert U_orth_err < 1e-10, f"U_b not orthogonal! Error: {U_orth_err}"
    assert V_orth_err < 1e-10, f"V_b not orthogonal! Error: {V_orth_err}"

    print("\n[PASS] 4x4 bidiagonal test passed!\n")


def test_deflation():
    """Test that deflation works when a superdiagonal element is zero."""
    print("=" * 60)
    print("Test 4: Deflation test (zero superdiagonal element)")
    print("=" * 60)

    # Create a bidiagonal matrix with one zero superdiagonal
    B = make_bidiagonal([4.0, 3.0, 2.0, 1.0], [1.0, 0.0, 0.5])
    print("Bidiagonal matrix with f[1] = 0:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nDiagonal D:")
    print(D)

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"\nMax reconstruction error: {max_err:.2e}")

    assert max_err < 1e-10, f"Reconstruction failed with deflation! Error: {max_err}"
    print("\n[PASS] Deflation test passed!\n")


def test_sorted_output():
    """Test that sorting option works correctly."""
    print("=" * 60)
    print("Test 5: Sorted singular values")
    print("=" * 60)

    B = make_bidiagonal([1.0, 5.0, 2.0, 4.0], [0.3, 0.2, 0.1])
    print("Original bidiagonal matrix B:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B, sort=True)

    diag_vals = [D.matrix[i][i] for i in range(D.rows)]
    print(f"\nSorted singular values: {diag_vals}")

    # Check that values are in descending order
    for i in range(len(diag_vals) - 1):
        assert diag_vals[i] >= diag_vals[i + 1], "Singular values not sorted!"

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"Max reconstruction error: {max_err:.2e}")

    assert max_err < 1e-10, f"Sorted reconstruction failed! Error: {max_err}"
    print("\n[PASS] Sorted output test passed!\n")


def test_integration_with_bidiagonalize():
    """
    Integration test: full SVD pipeline.
    A = U * B * V^T  (from bidiagonalize)
    B = U_b * D * V_b^T  (from diagonalize_bidiagonal_matrix)
    Therefore: A = (U * U_b) * D * (V * V_b)^T
    """
    print("=" * 60)
    print("Test 6: Integration with bidiagonalize (full SVD)")
    print("=" * 60)

    # Create a test matrix
    A = Matrix(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    print("Original matrix A:")
    print(A)

    # Step 1: Bidiagonalize
    U, B, V = bidiagonalize(A)
    print("\nBidiagonal matrix B:")
    print(B)

    # Step 2: Diagonalize the bidiagonal matrix
    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nDiagonal matrix D (singular values):")
    diag_vals = [D.matrix[i][i] for i in range(D.rows)]
    print(f"  {diag_vals}")

    # Compute the final SVD factors
    U_final = U * U_b
    V_final = V * V_b

    # Verify: A = U_final * D * V_final^T
    reconstructed = U_final * D * V_final.transpose()
    diff = A - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(A.rows) for j in range(A.cols))
    print(f"\nMax reconstruction error ||A - U*D*V^T||_max: {max_err:.2e}")

    # Verify orthogonality of final factors
    U_orth_err = check_orthogonality(U_final)
    V_orth_err = check_orthogonality(V_final)
    print(f"U_final orthogonality error: {U_orth_err:.2e}")
    print(f"V_final orthogonality error: {V_orth_err:.2e}")

    assert max_err < 1e-9, f"Full SVD reconstruction failed! Error: {max_err}"
    print("\n[PASS] Integration test passed!\n")


def test_random_matrix():
    """Test with a random matrix to ensure robustness."""
    print("=" * 60)
    print("Test 7: Random 5x5 bidiagonal matrix")
    print("=" * 60)

    random.seed(42)  # For reproducibility
    n = 5
    diag = [random.uniform(0.5, 5.0) for _ in range(n)]
    superdiag = [random.uniform(0.1, 2.0) for _ in range(n - 1)]

    B = make_bidiagonal(diag, superdiag)
    print("Random bidiagonal matrix B:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nSingular values:")
    diag_vals = [D.matrix[i][i] for i in range(D.rows)]
    print(f"  {diag_vals}")

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"\nMax reconstruction error: {max_err:.2e}")

    # Verify orthogonality
    U_orth_err = check_orthogonality(U_b)
    V_orth_err = check_orthogonality(V_b)
    print(f"U_b orthogonality error: {U_orth_err:.2e}")
    print(f"V_b orthogonality error: {V_orth_err:.2e}")

    assert max_err < 1e-9, f"Random test reconstruction failed! Error: {max_err}"
    assert U_orth_err < 1e-9, f"Random test U_b not orthogonal! Error: {U_orth_err}"
    assert V_orth_err < 1e-9, f"Random test V_b not orthogonal! Error: {V_orth_err}"

    print("\n[PASS] Random matrix test passed!\n")


def test_negative_diagonal():
    """Test handling of negative diagonal entries."""
    print("=" * 60)
    print("Test 8: Negative diagonal entries")
    print("=" * 60)

    B = make_bidiagonal([-3.0, 2.0, -1.0], [0.5, 0.3])
    print("Bidiagonal matrix with negative diagonal entries:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nDiagonal D (should be non-negative):")
    print(D)

    # Check all diagonal entries are non-negative
    for i in range(D.rows):
        assert D.matrix[i][i] >= -1e-12, f"Negative singular value at D[{i},{i}]"
    print("All singular values are non-negative: [PASS]")

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"Max reconstruction error: {max_err:.2e}")

    assert max_err < 1e-10, f"Negative diagonal test failed! Error: {max_err}"
    print("\n[PASS] Negative diagonal test passed!\n")


def test_2x2():
    """Test the simplest non-trivial case: 2x2 matrix."""
    print("=" * 60)
    print("Test 9: 2x2 bidiagonal matrix")
    print("=" * 60)

    B = make_bidiagonal([3.0, 4.0], [2.0])
    print("2x2 bidiagonal matrix B:")
    print(B)

    U_b, D, V_b = diagonalize_bidiagonal_matrix(B)

    print("\nD:")
    print(D)

    # Verify reconstruction
    reconstructed = U_b * D * V_b.transpose()
    diff = B - reconstructed
    max_err = max(abs(diff.matrix[i][j]) for i in range(B.rows) for j in range(B.cols))
    print(f"Max reconstruction error: {max_err:.2e}")

    assert max_err < 1e-10, f"2x2 test failed! Error: {max_err}"
    print("\n[PASS] 2x2 test passed!\n")


def run_all_tests():
    """Run all tests in the test suite."""
    print("\n" + "=" * 60)
    print("BIDIAGONAL DIAGONALIZATION TEST SUITE")
    print("=" * 60 + "\n")

    test_givens()
    test_small_bidiagonal()
    test_4x4_bidiagonal()
    test_deflation()
    test_sorted_output()
    test_integration_with_bidiagonalize()
    test_random_matrix()
    test_negative_diagonal()
    test_2x2()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
