from typing import Union


class Poly:
    def __init__(self, coeffs: list[float], tol: float = 1e-12):
        cleaned = []
        for c in coeffs:
            rounded = round(c)
            if abs(c - rounded) < tol:
                cleaned.append(float(rounded))
            else:
                cleaned.append(c)

        # Trim trailing zeros
        trimmed = cleaned[:]
        while trimmed and abs(trimmed[-1]) <= tol:
            trimmed.pop()
        self.coeffs = trimmed if trimmed else [0.0]

    def __str__(self):
        return " + ".join(reversed([f"{c}x^{i}" for i, c in enumerate(self.coeffs)]))

    def __add__(self, other: "Poly"):
        new_coeffs = []
        for i in range(max(len(self.coeffs), len(other.coeffs))):
            a = self.coeffs[i] if i < len(self.coeffs) else 0
            b = other.coeffs[i] if i < len(other.coeffs) else 0
            new_coeffs.append(a + b)
        return Poly(new_coeffs)

    def __sub__(self, other: "Poly"):
        return self + (other * -1)

    def __mul__(self, other: Union[int, float, "Poly"]):
        if isinstance(other, Poly):
            result_size = len(self.coeffs) + len(other.coeffs) - 1
            new_coeffs = [0.0] * result_size
            for i in range(len(self.coeffs)):
                for j in range(len(other.coeffs)):
                    new_coeffs[i + j] += self.coeffs[i] * other.coeffs[j]
            return Poly(new_coeffs)
        if isinstance(other, (int, float)):
            return Poly([c * other for c in self.coeffs])
        return NotImplemented

    def __rmul__(self, scalar: float):
        return Poly([c * scalar for c in self.coeffs])

    def __truediv__(self, scalar: Union[int, float]):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("division by zero")
        return Poly([c / scalar for c in self.coeffs])


def Lagrange_interpolation(xs: list[float], ys: list[float]):
    result = Poly([0.0])
    n = len(xs)

    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")

    if len(set(xs)) != len(xs):
        raise ValueError("Duplicate x-values detected; interpolation is not defined.")

    for i in range(n):
        numerator = Poly([1.0])
        denom = 1.0
        xi = xs[i]

        for j in range(n):
            if j == i:
                continue
            xj = xs[j]
            factor = Poly([-xj, 1.0])
            numerator = numerator * factor
            denom *= xi - xj

        if abs(denom) < 1e-14:
            raise ValueError(
                "Denominator too small (duplicate or very close x-values)."
            )

        # basis polynomial l_i(x) = numerator / denom
        li = numerator / denom

        # add f(x_i) * l_i(x) to result
        term = li * ys[i]
        result = result + term

    return result


def Divided_difference_with_values(xs: list[float], ys: list[float]):
    table = [[0.0] * len(xs) for _ in range(len(xs))]

    for i in range(len(xs)):
        table[i][i] = ys[i]

    for i in range(1, len(xs)):
        for j in range(len(xs) - i):
            table[j][j + i] = (table[j + 1][j + i] - table[j][j + i - 1]) / (
                xs[j + i] - xs[j]
            )
    return table


def Divided_difference(indices: list[int], xs: list[float], ys: list[float]):
    xs_subset = [xs[i] for i in indices]
    ys_subset = [ys[i] for i in indices]
    return Divided_difference_with_values(xs_subset, ys_subset)


def Newton_interpolation(xs: list[float], ys: list[float]) -> Poly:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")

    if len(set(xs)) != len(xs):
        raise ValueError("Duplicate x-values detected; interpolation is not defined.")

    divided_difference_table = Divided_difference_with_values(xs, ys)

    res = Poly([0.0])

    x = Poly([0, 1])
    e = Poly([1])

    n = len(xs)

    for i in range(n):
        term = Poly([1])
        coeff = divided_difference_table[0][i]

        # We want to build this term
        for j in range(i):
            term *= x - xs[j] * e

        res += term * coeff

    return res


def verify_equivalence(num_trials=100):
    import random

    for _ in range(num_trials):
        n = random.randint(2, 10)
        xs = [random.uniform(-10, 10) for _ in range(n)]
        ys = [random.uniform(-10, 10) for _ in range(n)]
        poly1 = Lagrange_interpolation(xs, ys)
        poly2 = Newton_interpolation(xs, ys)
        try:
            eps = 1e-6
            assert all(abs(a - b) <= eps for a, b in zip(poly1.coeffs, poly2.coeffs))
        except AssertionError:
            print(poly1)
            print()
            print(poly2)
            raise AssertionError

    print("Verification complete")


if __name__ == "__main__":
    # try:
    #     n = int(input("Enter number of data points n: ").strip())
    # except ValueError:
    #     raise ValueError("n must be an integer")

    # xs = []
    # ys = []
    # for i in range(n):
    #     raw = input(f"Enter x_{i} and f(x_{i}) separated by space: ").strip().split()
    #     if len(raw) != 2:
    #         raise ValueError("Please enter exactly two numbers per line: x_i f(x_i)")
    #     xi = float(raw[0])
    #     yi = float(raw[1])
    #     xs.append(xi)
    #     ys.append(yi)

    # if len(set(xs)) != len(xs):
    #     raise ValueError("Duplicate x-values detected; interpolation is not defined.")

    # poly1 = Lagrange_interpolation(xs, ys)

    # poly2 = Newton_interpolation(xs, ys)

    # print(poly1)
    # print(poly2)

    verify_equivalence()
