import math
from typing import Union


class Matrix:
    @property
    def rows(self) -> int:
        return len(self.matrix)

    @property
    def cols(self) -> int:
        return len(self.matrix[0])

    @property
    def size(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def __init__(self, matrix: list[list[float]], print_precision: int = 4):
        for i in range(len(matrix) - 1):
            if len(matrix[i]) != len(matrix[i + 1]):
                raise ValueError("All rows must have the same length")

        self.matrix = matrix
        self.print_precision = print_precision

    def __str__(self):
        # Convert all elements to strings
        str_matrix = [
            [f"{x:.{self.print_precision}f}" for x in row] for row in self.matrix
        ]
        # Find max width for each column
        col_widths = [
            max(len(str_matrix[row][col]) for row in range(len(str_matrix)))
            for col in range(len(str_matrix[0]))
        ]
        # Format each row with padding and brackets
        rows = []
        for i, row in enumerate(str_matrix):
            padded = [str_matrix[i][j].rjust(col_widths[j]) for j in range(len(row))]
            rows.append("[ " + "  ".join(padded) + " ]")
        return "\n".join(rows)

    def _multiply_matrix(self, other: "Matrix"):
        if self.cols != other.rows:
            raise ValueError(
                "Number of columns in the first matrix must match the number of rows in the second matrix"
            )

        return Matrix(
            [
                [sum(a * b for a, b in zip(row1, col2)) for col2 in zip(*other.matrix)]
                for row1 in self.matrix
            ]
        )

    def __add__(self, other: "Matrix"):
        # Check if the matrices have the same dimensions
        if self.size != other.size:
            raise ValueError("Matrices must have the same dimensions")

        return Matrix(
            [
                [a + b for a, b in zip(row1, row2)]
                for row1, row2 in zip(self.matrix, other.matrix)
            ]
        )

    def __sub__(self, other: "Matrix"):
        # Check if the matrices have the same dimensions
        if self.size != other.size:
            raise ValueError("Matrices must have the same dimensions")

        return Matrix(
            [
                [a - b for a, b in zip(row1, row2)]
                for row1, row2 in zip(self.matrix, other.matrix)
            ]
        )

    def __rmul__(self, other: float):
        return Matrix([[other * x for x in row] for row in self.matrix])

    def __mul__(self, other: Union[int, float, "Matrix"]):

        if isinstance(other, (int, float)):
            return other * self

        if isinstance(other, Matrix):
            return self._multiply_matrix(other)

        raise ValueError("Invalid operand type")

    def norm(self) -> float:
        """
        Compute the Frobenius norm of the matrix.

        For a 1*n matrix (vector), this is the Euclidean norm.
        """
        return math.sqrt(sum(x * x for row in self.matrix for x in row))

    @staticmethod
    def Identity(n: int) -> "Matrix":
        """Create an n*n identity matrix."""
        return Matrix([[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)])

    def transpose(self) -> "Matrix":
        """Return the transpose of this matrix."""
        return Matrix(
            [[self.matrix[i][j] for i in range(self.rows)] for j in range(self.cols)]
        )

    def submatrix(
        self, row_start: int, row_end: int, col_start: int, col_end: int
    ) -> "Matrix":
        """Extract a submatrix A[row_start:row_end, col_start:col_end]."""
        return Matrix(
            [
                [self.matrix[i][j] for j in range(col_start, col_end)]
                for i in range(row_start, row_end)
            ]
        )

    def set_submatrix(self, row_start: int, col_start: int, sub: "Matrix") -> None:
        """Set values in-place: A[row_start:, col_start:] = sub."""
        for i in range(sub.rows):
            for j in range(sub.cols):
                self.matrix[row_start + i][col_start + j] = sub.matrix[i][j]

    def get_column(self, col: int, row_start: int = 0) -> "Matrix":
        """Extract column as a column vector (n*1 Matrix)."""
        return Matrix([[self.matrix[i][col]] for i in range(row_start, self.rows)])

    def get_row(self, row: int, col_start: int = 0) -> "Matrix":
        """Extract row as a row vector (1*n Matrix)."""
        return Matrix([[self.matrix[row][j] for j in range(col_start, self.cols)]])
