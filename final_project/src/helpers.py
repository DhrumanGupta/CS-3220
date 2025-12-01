import math

from matrix import Matrix


def householder_vector_from_vector(vector: Matrix) -> Matrix | None:
    """
    Compute the Householder vector that zeros out elements after the first.

    Args:
        vector: A Matrix of dimension 1*n (row vector) or n*1 (column vector)

    Returns:
        A normalized column vector (n*1 Matrix), or None if no reflection is needed.
    """
    # Handle both row and column vectors - work with row form internally
    if vector.cols == 1:
        # Column vector - transpose to row for processing
        row_vec = vector.transpose()
    else:
        row_vec = vector

    # Compute norm using the Matrix method
    norm_x = row_vec.norm()

    if norm_x < 1e-15:
        return None

    # Extract as list for modification
    x = row_vec.matrix[0][:]

    # Choose sign to avoid cancellation
    sign = 1.0 if x[0] >= 0 else -1.0

    # v = x + sign(x[0]) * ||x|| * e_1
    v = x[:]
    v[0] = v[0] + sign * norm_x

    # Normalize v
    v_matrix = Matrix([v])
    norm_v = v_matrix.norm()

    if norm_v < 1e-15:
        return None

    # Return as column vector (n*1)
    return Matrix([[vi / norm_v] for vi in v])


def bidiagonalize(
    matrix: Matrix,
) -> tuple[Matrix, Matrix, Matrix]:
    """
    Compute the bidiagonalization A = U * B * V^T using Householder reflectors.

    Args:
        matrix: Input matrix A (m x n)

    Returns:
        Tuple (U, B, V) where:
        - U is an m x m orthogonal matrix
        - B is an m x n upper bidiagonal matrix
        - V is an n x n orthogonal matrix
    """
    m, n = matrix.size

    # Create working copy of A
    A = Matrix([row[:] for row in matrix.matrix])

    # Initialize U (m x m) and V (n x n) as identity matrices
    U = Matrix.Identity(m)
    V = Matrix.Identity(n)

    # -------------------------------------------------------------------------
    # Helper functions using Matrix operations
    # -------------------------------------------------------------------------

    def apply_left_reflector(M: Matrix, u: Matrix | None, k: int) -> None:
        """
        Apply left Householder reflector in-place: M[k:, :] ← (I - 2uu^T) * M[k:, :]
        Equivalent to: M[k:, :] ← M[k:, :] - 2*u*(u^T*M[k:, :])
        u is a column vector (m-k)*1, acts on rows k:m
        """
        if u is None:
            return

        # Extract submatrix M[k:m, :]
        M_sub = M.submatrix(k, M.rows, 0, M.cols)

        # Compute: M_sub - 2 * u * (u^T * M_sub)
        uT_M = u.transpose() * M_sub  # 1 x cols
        update = 2.0 * (u * uT_M)  # (m-k) x cols

        M_new = M_sub - update
        M.set_submatrix(k, 0, M_new)

    def apply_right_reflector(M: Matrix, v: Matrix | None, k: int) -> None:
        """
        Apply right Householder reflector in-place: M[:, k:] ← M[:, k:] * (I - 2vv^T)
        Equivalent to: M[:, k:] ← M[:, k:] - 2*(M[:, k:]*v)*v^T
        v is a column vector (n-k)*1, acts on columns k:n
        """
        if v is None:
            return

        # Extract submatrix M[:, k:n]
        M_sub = M.submatrix(0, M.rows, k, M.cols)

        # Compute: M_sub - 2 * (M_sub * v) * v^T
        M_v = M_sub * v  # rows x 1
        update = 2.0 * (M_v * v.transpose())  # rows x (n-k)

        M_new = M_sub - update
        M.set_submatrix(0, k, M_new)

    def accumulate_reflector(M: Matrix, u: Matrix | None, k: int) -> None:
        """
        Accumulate reflector into orthogonal matrix: M[:, k:] ← M[:, k:] * (I - 2uu^T)
        Equivalent to: M[:, k:] ← M[:, k:] - 2*(M[:, k:]*u)*u^T
        u is a column vector, acts on columns k onwards
        """
        if u is None:
            return

        # Extract submatrix M[:, k:]
        M_sub = M.submatrix(0, M.rows, k, M.cols)

        # Compute: M_sub - 2 * (M_sub * u) * u^T
        M_u = M_sub * u  # rows x 1
        update = 2.0 * (M_u * u.transpose())  # rows x len(u)

        M_new = M_sub - update
        M.set_submatrix(0, k, M_new)

    # -------------------------------------------------------------------------
    # Main bidiagonalization algorithm
    # -------------------------------------------------------------------------

    num_steps = min(m, n)

    for k in range(num_steps):
        # Step 1: Left Householder to zero out A[k+1:m, k]
        col_vec = A.get_column(k, k)  # Column vector (m-k) x 1
        u = householder_vector_from_vector(col_vec)

        apply_left_reflector(A, u, k)  # A ← H * A
        accumulate_reflector(U, u, k)  # U ← U * H

        # Step 2: Right Householder to zero out A[k, k+2:n]
        if k < n - 2:
            row_vec = A.get_row(k, k + 1)  # Row vector 1 x (n-k-1)
            v = householder_vector_from_vector(row_vec)

            apply_right_reflector(A, v, k + 1)  # A ← A * G
            accumulate_reflector(V, v, k + 1)  # V ← V * G

    return (U, A, V)


def is_bidiagonal_matrix(B: Matrix, tol: float = 1e-14) -> bool:
    """
    Check if matrix B is upper bidiagonal.

    An upper bidiagonal matrix has nonzero entries only on the main diagonal
    and the superdiagonal (the diagonal immediately above the main diagonal).

    Args:
        B: Matrix to check
        tol: Tolerance for considering an element as zero (default 1e-14)

    Returns:
        True if B is upper bidiagonal, False otherwise.
    """
    for i in range(B.rows):
        for j in range(B.cols):
            # Valid positions: diagonal (i == j) or superdiagonal (j == i + 1)
            if i != j and j != i + 1:
                if abs(B.matrix[i][j]) > tol:
                    return False
    return True


# =============================================================================
# Givens Rotation Utilities
# =============================================================================


def givens(a: float, b: float, tol: float = 1e-14) -> tuple[float, float, float]:
    """
    Compute Givens rotation parameters (c, s, r) such that:
        [ c   s ] [ a ]   [ r ]
        [-s   c ] [ b ] = [ 0 ]

    Uses math.hypot for numerical stability to avoid overflow/underflow.

    Args:
        a: First component of the vector
        b: Second component of the vector (to be zeroed)

    Returns:
        Tuple (c, s, r) where:
        - c = cos(theta) = a / r
        - s = b / r
        - r = hypot(a, b) = sqrt(a^2 + b^2)

    Verification: c*a + s*b = a^2/r + b^2/r = r ✓
                  -s*a + c*b = -ab/r + ab/r = 0 ✓

    If both a and b are zero, returns (1.0, 0.0, 0.0) (identity rotation).
    """
    r = math.hypot(a, b)
    if r < tol:  # Effectively zero
        return (1.0, 0.0, 0.0)
    c = a / r
    s = b / r
    return (c, s, r)


def apply_givens_left_inplace(A: Matrix, i: int, j: int, c: float, s: float) -> None:
    """
    Apply a Givens rotation from the left to rows i and j of matrix A in-place.

    This computes:
        [ A[i, :] ]     [ c   s ] [ A[i, :] ]
        [ A[j, :] ]  <- [-s   c ] [ A[j, :] ]

    For each column k:
        t_i = c * A[i][k] + s * A[j][k]
        t_j = -s * A[i][k] + c * A[j][k]
        A[i][k] = t_i
        A[j][k] = t_j

    Args:
        A: Matrix to modify in-place
        i: First row index
        j: Second row index (should be i+1 for adjacent rotations)
        c: Cosine of rotation angle
        s: Sine of rotation angle (with sign convention from givens())
    """
    for k in range(A.cols):
        t_i = c * A.matrix[i][k] + s * A.matrix[j][k]
        t_j = -s * A.matrix[i][k] + c * A.matrix[j][k]
        A.matrix[i][k] = t_i
        A.matrix[j][k] = t_j


def apply_givens_right_inplace(A: Matrix, i: int, j: int, c: float, s: float) -> None:
    """
    Apply a Givens rotation from the right to columns i and j of matrix A in-place.

    This computes:
        [ A[:, i]  A[:, j] ] <- [ A[:, i]  A[:, j] ] [ c   s ]
                                                     [ -s  c ]

    For each row k:
        t_i = c * A[k][i] + s * A[k][j]
        t_j = -s * A[k][i] + c * A[k][j]
        A[k][i] = t_i
        A[k][j] = t_j

    Args:
        A: Matrix to modify in-place
        i: First column index
        j: Second column index (should be i+1 for adjacent rotations)
        c: Cosine of rotation angle
        s: Sine of rotation angle (with sign convention from givens())
    """
    for k in range(A.rows):
        t_i = c * A.matrix[k][i] + s * A.matrix[k][j]
        t_j = -s * A.matrix[k][i] + c * A.matrix[k][j]
        A.matrix[k][i] = t_i
        A.matrix[k][j] = t_j


# =============================================================================
# Bidiagonal Array Update Functions
# =============================================================================


def apply_left_givens_to_bf(
    b: list[float],
    f: list[float],
    bulge: float,
    i: int,
    c: float,
    s: float,
) -> float:
    """
    Apply a left Givens rotation to rows i and i+1 of a bidiagonal matrix,
    represented by diagonal b and superdiagonal f arrays.

    The bidiagonal structure before rotation (showing rows i and i+1):
        Row i:      [..., b[i],   f[i],   0,      ...]
        Row i+1:    [..., bulge,  b[i+1], f[i+1], ...]

    The 'bulge' is the element at position (i+1, i) introduced by a previous
    right rotation. For the first rotation, bulge = 0.

    After left rotation on rows i and i+1:
        - The bulge at (i+1, i) is zeroed
        - A new bulge appears at (i, i+2) in the superdiagonal structure

    Args:
        b: Diagonal elements [b[0], b[1], ..., b[n-1]]
        f: Superdiagonal elements [f[0], f[1], ..., f[n-2]] where f[k] = B[k, k+1]
        bulge: Current bulge value at position (i+1, i)
        i: Row index (rotation affects rows i and i+1)
        c: Cosine from givens()
        s: Sine from givens()

    Returns:
        New bulge value that appears at position (i, i+2), or 0 if i+1 >= len(f).
    """
    n = len(b)

    # Current values
    b_i = b[i]
    f_i = f[i] if i < len(f) else 0.0
    b_i1 = b[i + 1] if i + 1 < n else 0.0
    f_i1 = f[i + 1] if i + 1 < len(f) else 0.0

    # Apply rotation to columns i, i+1, i+2 of rows i and i+1
    # Column i: [b[i], bulge]^T -> [new_b[i], 0]^T
    new_b_i = c * b_i - s * bulge
    # (the bulge is zeroed by design of the Givens rotation)

    # Column i+1: [f[i], b[i+1]]^T
    new_f_i = c * f_i - s * b_i1
    new_b_i1 = s * f_i + c * b_i1

    # Column i+2: [0, f[i+1]]^T -> [new_bulge, new_f[i+1]]^T
    new_bulge = -s * f_i1
    new_f_i1 = c * f_i1

    # Update arrays
    b[i] = new_b_i
    if i < len(f):
        f[i] = new_f_i
    b[i + 1] = new_b_i1
    if i + 1 < len(f):
        f[i + 1] = new_f_i1

    return new_bulge


def apply_right_givens_to_bf(
    b: list[float],
    f: list[float],
    bulge: float,
    j: int,
    c: float,
    s: float,
) -> float:
    """
    Apply a right Givens rotation to columns j and j+1 of a bidiagonal matrix,
    represented by diagonal b and superdiagonal f arrays.

    The bidiagonal structure before rotation (showing cols j and j+1):
        Row j-1:    [..., f[j-1], bulge,  ...]   (bulge at position (j-1, j+1))
        Row j:      [..., b[j],   f[j],   ...]
        Row j+1:    [..., 0,      b[j+1], ...]

    The 'bulge' is the element at position (j-1, j+1) introduced by a previous
    left rotation.

    After right rotation on columns j and j+1:
        - The bulge at (j-1, j+1) is zeroed
        - A new bulge appears at (j+1, j) below the diagonal

    Args:
        b: Diagonal elements
        f: Superdiagonal elements
        bulge: Current bulge value at position (j-1, j+1)
        j: Column index (rotation affects columns j and j+1)
        c: Cosine from givens()
        s: Sine from givens()

    Returns:
        New bulge value that appears at position (j+1, j).
    """
    n = len(b)

    # Current values
    f_jm1 = f[j - 1] if j > 0 else 0.0
    b_j = b[j]
    f_j = f[j] if j < len(f) else 0.0
    b_j1 = b[j + 1] if j + 1 < n else 0.0

    # Apply rotation to rows j-1, j, j+1 of columns j and j+1

    # Row j-1: [f[j-1], bulge] -> [new_f[j-1], 0]
    if j > 0:
        new_f_jm1 = c * f_jm1 + s * bulge
        f[j - 1] = new_f_jm1
    # (bulge is zeroed by design)

    # Row j: [b[j], f[j]]
    new_b_j = c * b_j + s * f_j
    new_f_j = -s * b_j + c * f_j

    # Row j+1: [0, b[j+1]] -> [new_bulge, new_b[j+1]]
    new_bulge = s * b_j1
    new_b_j1 = c * b_j1

    # Update arrays
    b[j] = new_b_j
    if j < len(f):
        f[j] = new_f_j
    if j + 1 < n:
        b[j + 1] = new_b_j1

    return new_bulge


# =============================================================================
# Shift and Deflation Functions
# =============================================================================


def rayleigh_shift(b: list[float], f: list[float], low: int, high: int) -> float:
    """
    Compute the Wilkinson (Rayleigh-quotient) shift for the implicit QR algorithm
    on a bidiagonal matrix.

    The shift is computed from the eigenvalues of the bottom-right 2x2 block of
    T = B^T * B, where B is the bidiagonal matrix. We choose the eigenvalue
    closer to t11 = b[high]^2, which typically gives faster convergence.

    For the 2x2 block of B^T B at indices [high-1, high]:
        t00 = b[high-1]^2 + f[high-2]^2   (if high-2 >= low, else just b[high-1]^2)
        t01 = b[high-1] * f[high-1]
        t11 = b[high]^2 + f[high-1]^2     (but f[high-1] appears in off-diag only)

    Actually, for T = B^T B:
        T[k,k] = b[k]^2 + (f[k-1]^2 if k > 0 else 0)
        T[k,k+1] = b[k] * f[k]

    So the 2x2 trailing block is:
        [ b[high-1]^2 + f[high-2]^2,  b[high-1]*f[high-1] ]
        [ b[high-1]*f[high-1],        b[high]^2 + f[high-1]^2 ]

    Args:
        b: Diagonal elements
        f: Superdiagonal elements
        low: Lower index of active block
        high: Upper index of active block (high > low required)

    Returns:
        The Wilkinson shift value (eigenvalue of 2x2 closer to t11).
    """
    if high <= low:
        # Single element block, no shift needed
        return 0.0

    # Build the 2x2 block of B^T B
    # T[high-1, high-1] = b[high-1]^2 + (f[high-2]^2 if high-2 >= low else 0)
    # T[high-1, high] = b[high-1] * f[high-1]
    # T[high, high] = b[high]^2 + f[high-1]^2

    bm = b[high - 1]  # b[high-1]
    bh = b[high]  # b[high]
    fm = f[high - 1] if high - 1 < len(f) else 0.0  # f[high-1]

    # Diagonal entries of 2x2
    t00 = bm * bm
    if high - 2 >= low and high - 2 < len(f):
        t00 += f[high - 2] ** 2
    t01 = bm * fm
    t11 = bh * bh + fm * fm

    # Eigenvalues of [[t00, t01], [t01, t11]]
    # Using the formula: lambda = (t00 + t11)/2 +/- sqrt(((t00-t11)/2)^2 + t01^2)

    d = (t00 - t11) / 2.0
    # Use hypot for numerical stability
    r = math.hypot(d, t01)

    # Two eigenvalues
    mid = (t00 + t11) / 2.0
    lambda1 = mid + r
    lambda2 = mid - r

    # Choose eigenvalue closer to t11 (Wilkinson strategy)
    if abs(lambda1 - t11) < abs(lambda2 - t11):
        return lambda1
    else:
        return lambda2


def deflate_index(
    b: list[float], f: list[float], low: int, high: int, tol: float
) -> int | None:
    """
    Find an index k in [low, high-1] where the superdiagonal f[k] is negligible,
    allowing the matrix to be split into two independent subproblems.

    Deflation criterion: |f[k]| <= tol * (|b[k]| + |b[k+1]|)

    Args:
        b: Diagonal elements
        f: Superdiagonal elements
        low: Lower index of active block
        high: Upper index of active block
        tol: Tolerance for deflation (typically 1e-12)

    Returns:
        Index k where deflation can occur, or None if no deflation is possible.
        Returns the largest such k (rightmost deflation point) to maximize
        progress per deflation.
    """
    # Search from high-1 down to low to find rightmost deflation point
    for k in range(high - 1, low - 1, -1):
        if k < len(f):
            threshold = tol * (abs(b[k]) + abs(b[k + 1]))
            if abs(f[k]) <= threshold:
                return k
    return None


# =============================================================================
# Orthogonality Check Utility
# =============================================================================


def check_orthogonality(M: Matrix) -> float:
    """
    Check how close a matrix M is to being orthogonal by computing ||I - M^T M||_F.

    Args:
        M: Matrix to check for orthogonality

    Returns:
        Frobenius norm of (I - M^T M). Should be close to 0 for orthogonal M.
    """
    n = M.cols
    I = Matrix.Identity(n)
    MTM = M.transpose() * M
    diff = I - MTM
    return diff.norm()


# =============================================================================
# Main Diagonalization Algorithm
# =============================================================================


def diagonalize_bidiagonal_matrix(
    B: Matrix,
    tol: float = 1e-12,
    maxiter: int = 1000,
    sort: bool = False,
) -> tuple[Matrix, Matrix, Matrix]:
    """
    Diagonalize an upper bidiagonal matrix using the implicit QR algorithm
    with Givens rotations and Wilkinson (Rayleigh-quotient) shifts.

    This implements the standard SVD algorithm for bidiagonal matrices:
    1. Extract diagonal b and superdiagonal f
    2. Use implicit QR steps with shifts to drive superdiagonal to zero
    3. Accumulate left and right Givens rotations into U_b and V_b

    The algorithm uses bulge-chasing: each QR step creates a "bulge" that
    is chased down the matrix via alternating left and right Givens rotations
    until it exits at the bottom-right corner.

    For rectangular m x n bidiagonal matrices (m != n), the matrix is internally
    padded to square max(m,n) x max(m,n) form before diagonalization.

    Complexity: O(n^2) per sweep, typically O(n^2) to O(n^3) total depending
    on convergence. With Wilkinson shifts, convergence is usually cubic.

    Integration with bidiagonalize():
        If A = U * B * V^T from bidiagonalize(), and B = U_b * D * V_b^T,
        then A = (U * U_b) * D * (V * V_b)^T gives the full SVD.

    Args:
        B: Upper bidiagonal matrix (m x n), can be rectangular
        tol: Tolerance for deflation criterion (default 1e-12)
        maxiter: Maximum iterations per active block (default 1000)
        sort: If True, sort singular values in descending order (default False)

    Returns:
        Tuple (U_b, D, V_b) where:
        - U_b: m x m orthogonal matrix (accumulated left Givens rotations)
        - D: m x n diagonal matrix (singular values on diagonal, non-negative)
        - V_b: n x n orthogonal matrix (accumulated right Givens rotations)

        Such that B = U_b * D * V_b^T (approximately, up to numerical tolerance)

    Raises:
        ValueError: If B is not bidiagonal
        RuntimeError: If algorithm fails to converge within maxiter iterations
    """
    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------
    if not is_bidiagonal_matrix(B):
        raise ValueError("B must be a bidiagonal matrix")

    m_orig, n_orig = B.rows, B.cols

    # Handle trivial cases
    if m_orig == 0 or n_orig == 0:
        return (Matrix([[]]), Matrix([[]]), Matrix([[]]))
    if m_orig == 1 and n_orig == 1:
        val = B.matrix[0][0]
        sign = 1.0 if val >= 0 else -1.0
        D = Matrix([[abs(val)]])
        U_b = Matrix([[sign]])
        V_b = Matrix([[1.0]])
        return (U_b, D, V_b)

    # -------------------------------------------------------------------------
    # Pad rectangular matrix to square if needed
    # -------------------------------------------------------------------------
    n = max(m_orig, n_orig)

    # Extract diagonal and superdiagonal from original B
    # Diagonal: min(m, n) elements at positions (i, i)
    # Superdiagonal: for m x n bidiagonal, elements at (i, i+1) for i < min(m, n-1)
    min_dim = min(m_orig, n_orig)
    b = [B.matrix[i][i] if i < m_orig and i < n_orig else 0.0 for i in range(n)]

    # Superdiagonal: f[i] = B[i, i+1]
    # For rectangular m < n, we have min(m, n-1) = m superdiagonal elements (one extra!)
    f = []
    for i in range(n - 1):
        if i < m_orig and i + 1 < n_orig:
            f.append(B.matrix[i][i + 1])
        else:
            f.append(0.0)

    # -------------------------------------------------------------------------
    # Initialize accumulators as identity matrices
    # -------------------------------------------------------------------------
    U_b = Matrix.Identity(n)
    V_b = Matrix.Identity(n)

    # -------------------------------------------------------------------------
    # Main iteration loop with deflation
    # -------------------------------------------------------------------------
    # We process the matrix from the bottom-right corner upward.
    # 'high' marks the current bottom of the active block.
    # When f[high-1] becomes negligible, we decrement high.

    high = n - 1
    total_iterations = 0

    while high > 0:
        low = 0

        # Check for deflation within the current block
        # Find the largest index where deflation occurs
        deflate_k = deflate_index(b, f, low, high, tol)

        if deflate_k is not None:
            # Set the negligible superdiagonal to exactly zero
            f[deflate_k] = 0.0

            if deflate_k == high - 1:
                # Bottom element is converged, move up
                high -= 1
                continue
            else:
                # Split: process the lower block [deflate_k+1, high]
                low = deflate_k + 1

        # Check if the entire active block has negligible values (zero singular values)
        # This handles the case where the original matrix was rank-deficient
        block_max = max(
            max(abs(b[k]) for k in range(low, high + 1)),
            max(abs(f[k]) for k in range(low, high) if k < len(f)) if low < high else 0,
        )
        if block_max < tol:
            # All values in this block are essentially zero - skip it
            for k in range(low, high):
                if k < len(f):
                    f[k] = 0.0
            high = low - 1 if low > 0 else 0
            continue

        # Check for zero diagonal elements (special case)
        # If b[k] ≈ 0 for some k in [low, high], we need special handling
        # to avoid division issues. Apply a rotation to zero f[k] directly.
        zero_diag = None
        for k in range(low, high):
            if abs(b[k]) <= tol * (abs(f[k]) if k < len(f) else 1.0):
                zero_diag = k
                break

        if zero_diag is not None:
            # Zero out f[zero_diag] using a sequence of left rotations
            # This handles the case where b[k] ≈ 0
            k = zero_diag
            if k < len(f) and abs(f[k]) > tol:
                # Rotate to eliminate f[k] using rows k and k+1
                c, s, _ = givens(b[k + 1], f[k])
                # For left rotation G_L on B: B <- G_L * B
                # G_L = [[c, s], [-s, c]], so G_L^T = [[c, -s], [s, c]]
                # Accumulate into U: U <- U * G_L^T
                # apply_givens_right_inplace with (c, s) applies [[c, -s], [s, c]] on the right
                apply_givens_right_inplace(U_b, k, k + 1, c, s)

                # Update b and f: new values after applying [[c, s], [-s, c]] to rows
                new_b_k1 = c * b[k + 1] + s * f[k]
                f[k] = 0.0
                b[k + 1] = new_b_k1
            continue

        # ---------------------------------------------------------------------
        # Perform one implicit QR step with Wilkinson shift
        # ---------------------------------------------------------------------
        total_iterations += 1
        if total_iterations > maxiter * n:
            raise RuntimeError(
                f"diagonalize_bidiagonal_matrix failed to converge after "
                f"{total_iterations} iterations. Active block: [{low}, {high}], "
                f"b = {b[low:high+1]}, f = {f[low:high]}"
            )

        # Compute the Wilkinson shift
        mu = rayleigh_shift(b, f, low, high)

        # ---------------------------------------------------------------------
        # Initialize the bulge chase
        # ---------------------------------------------------------------------
        # Compute the first Givens rotation from the implicit shift.
        # We work with the first column of (B^T B - mu I), which has entries:
        #   x = b[low]^2 - mu
        #   y = b[low] * f[low]

        x = b[low] * b[low] - mu
        y = b[low] * f[low]

        # First right Givens rotation on columns low and low+1
        c, s, _ = givens(x, y)

        # Apply right rotation to columns low and low+1
        apply_givens_right_inplace(V_b, low, low + 1, c, s)

        # Update b, f - this creates a bulge at position (low+1, low)
        # Right rotation G_R with G_R^T = [[c, -s], [s, c]] applied: B <- B * G_R^T
        # For row i: [b, f] * [[c, -s], [s, c]] = [c*b + s*f, -s*b + c*f]
        # B[low, low] = c*b[low] + s*f[low]
        # B[low, low+1] = -s*b[low] + c*f[low]
        # B[low+1, low] = s*b[low+1]  <- BULGE (from col low+1 mixing into col low)
        # B[low+1, low+1] = c*b[low+1]

        new_b_low = c * b[low] + s * f[low]
        new_f_low = -s * b[low] + c * f[low]
        bulge = s * b[low + 1]
        new_b_low1 = c * b[low + 1]

        b[low] = new_b_low
        f[low] = new_f_low
        b[low + 1] = new_b_low1

        # ---------------------------------------------------------------------
        # Chase the bulge down the matrix
        # ---------------------------------------------------------------------
        for k in range(low, high):
            # We have a bulge at position (k+1, k).
            # Apply left Givens rotation to rows k and k+1 to zero the bulge.

            c, s, _ = givens(b[k], bulge)
            # For left rotation G_L on B: B <- G_L * B
            # G_L = [[c, s], [-s, c]], so G_L^T = [[c, -s], [s, c]]
            # Accumulate into U: U <- U * G_L^T
            # apply_givens_right_inplace with (c, s) applies [[c, -s], [s, c]] on the right
            apply_givens_right_inplace(U_b, k, k + 1, c, s)

            # Update b, f arrays for left rotation G_L = [[c, s], [-s, c]]
            # Row k:   [..., b[k], f[k], 0, ...]
            # Row k+1: [..., bulge, b[k+1], f[k+1], ...]
            #
            # After G_L * B (left multiplication):
            # Row k:   [..., c*b[k]+s*bulge, c*f[k]+s*b[k+1], s*f[k+1], ...]
            # Row k+1: [..., -s*b[k]+c*bulge, -s*f[k]+c*b[k+1], c*f[k+1], ...]
            # Note: -s*b[k]+c*bulge = 0 by design of Givens (since c=b[k]/r, s=bulge/r)

            new_b_k = c * b[k] + s * bulge  # = r = hypot(b[k], bulge)
            new_f_k = c * f[k] + s * b[k + 1] if k < len(f) else 0.0
            new_b_k1 = -s * f[k] + c * b[k + 1] if k < len(f) else c * b[k + 1]

            # New bulge appears at position (k, k+2) from s*f[k+1]
            new_bulge_top = s * f[k + 1] if k + 1 < len(f) else 0.0
            new_f_k1 = c * f[k + 1] if k + 1 < len(f) else 0.0

            b[k] = new_b_k
            if k < len(f):
                f[k] = new_f_k
            b[k + 1] = new_b_k1
            if k + 1 < len(f):
                f[k + 1] = new_f_k1

            # If we're not at the last step, apply right rotation to zero the
            # bulge that appeared at (k, k+2)
            if k + 1 < high:
                # Apply right Givens rotation on columns k+1 and k+2
                c, s, _ = givens(f[k], new_bulge_top)
                apply_givens_right_inplace(V_b, k + 1, k + 2, c, s)

                # Update for right rotation
                # Col k+1: [f[k], b[k+1], 0]^T (in rows k, k+1, k+2)
                # Col k+2: [new_bulge_top, f[k+1], b[k+2]]^T
                #
                # After rotation:
                # Col k+1: [c*f[k]+s*new_bulge_top, c*b[k+1]+s*f[k+1], s*b[k+2]]
                # Col k+2: [-s*f[k]+c*new_bulge_top, -s*b[k+1]+c*f[k+1], c*b[k+2]]

                old_f_k = f[k]
                old_b_k1 = b[k + 1]
                old_f_k1 = f[k + 1] if k + 1 < len(f) else 0.0
                old_b_k2 = b[k + 2] if k + 2 < n else 0.0

                f[k] = c * old_f_k + s * new_bulge_top
                new_b_k1 = c * old_b_k1 + s * old_f_k1
                bulge = s * old_b_k2  # New bulge at (k+2, k+1)

                # Col k+2 updates
                # new_bulge_top position becomes 0 by design
                new_f_k1 = -s * old_b_k1 + c * old_f_k1
                new_b_k2 = c * old_b_k2

                b[k + 1] = new_b_k1
                if k + 1 < len(f):
                    f[k + 1] = new_f_k1
                if k + 2 < n:
                    b[k + 2] = new_b_k2
            else:
                # Bulge has exited the matrix
                bulge = 0.0

    # -------------------------------------------------------------------------
    # Post-processing: ensure non-negative diagonal
    # -------------------------------------------------------------------------
    for i in range(n):
        if b[i] < 0:
            b[i] = -b[i]
            # Flip sign of column i in V_b
            for k in range(n):
                V_b.matrix[k][i] = -V_b.matrix[k][i]

    # -------------------------------------------------------------------------
    # Optional: sort singular values in descending order
    # -------------------------------------------------------------------------
    if sort:
        # Create list of (value, index) pairs and sort
        indexed = [(b[i], i) for i in range(n)]
        indexed.sort(key=lambda x: -x[0])  # Descending order

        # Apply permutation to b and to columns of U_b and V_b
        perm = [idx for _, idx in indexed]
        b_sorted = [b[perm[i]] for i in range(n)]

        # Permute columns of U_b
        U_b_new = [[U_b.matrix[row][perm[col]] for col in range(n)] for row in range(n)]
        # Permute columns of V_b
        V_b_new = [[V_b.matrix[row][perm[col]] for col in range(n)] for row in range(n)]

        b = b_sorted
        U_b = Matrix(U_b_new)
        V_b = Matrix(V_b_new)

    # -------------------------------------------------------------------------
    # Build output matrices with correct dimensions for rectangular case
    # -------------------------------------------------------------------------
    # U_b should be m_orig x m_orig (extract from padded n x n)
    if m_orig < n:
        U_b_final = Matrix(
            [[U_b.matrix[i][j] for j in range(m_orig)] for i in range(m_orig)]
        )
    else:
        U_b_final = U_b

    # V_b should be n_orig x n_orig (extract from padded n x n)
    if n_orig < n:
        V_b_final = Matrix(
            [[V_b.matrix[i][j] for j in range(n_orig)] for i in range(n_orig)]
        )
    else:
        V_b_final = V_b

    # D should be m_orig x n_orig (rectangular diagonal matrix)
    D_data = [[b[i] if i == j else 0.0 for j in range(n_orig)] for i in range(m_orig)]
    D = Matrix(D_data)

    return (U_b_final, D, V_b_final)
