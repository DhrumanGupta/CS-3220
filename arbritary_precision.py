# Arbritary Precision Class, uses strings


class ArbitraryPrecision:
    def __init__(self, value: str):
        # If value is all 0s, set to 0
        if all(c == "0" for c in value):
            value = "0"

        if any((not c.isdigit() and c != "." for c in value)):
            raise ValueError("Invalid value")

        if len([i for i in value if i == "."]) > 1:
            raise ValueError("Invalid value")

        if "." in value:
            # After the . if all are 0s, remove the .
            if all(c == "0" for c in value.split(".")[1]):
                value = value.split(".")[0]
            # Remove all leading 0s if there is a "."
            else:
                left = value.split(".")[0]
                right = value.split(".")[1]
                for i in range(len(right) - 1, -1, -1):
                    if right[i] == "0":
                        right = right[:i]
                    else:
                        break
                value = left + "." + right

        self.value = value

    def __str__(self):
        return self.value

    def _get_num_dots(self, value: str) -> int:
        return len(value.split(".")[1]) if "." in value else 0

    def _left_shift(self, value: str, num_shifts: int) -> str:
        if num_shifts == 0:
            return value

        num_dots = self._get_num_dots(value)
        if num_dots == 0:
            return value + ("0" * num_shifts)
        else:
            left = value.split(".")[0]
            right = value.split(".")[1]
            if len(right) > num_shifts:
                return left + right[:-num_shifts] + "." + right[-num_shifts:]
            else:
                diff = num_shifts - len(right)
                return left + right + ("0" * diff)

    def _right_shift(self, value: str, num_shifts: int) -> str:
        if num_shifts == 0:
            return value
        num_dots = self._get_num_dots(value)
        print(f"Right shifting {value} by {num_shifts}")
        left = value.split(".")[0] if num_dots == 1 else value
        right = value.split(".")[1] if num_dots == 1 else "0"

        res = (
            left[: len(left) - num_shifts]
            + "."
            + left[len(left) - num_shifts :]
            + right
        )

        return res

    def __add__(self, other: "ArbitraryPrecision"):
        left = str(self.value)
        right = str(other.value)

        # Simplify this by padding 0s to the shorter string (according to dot)
        num_after_dot_left = self._get_num_dots(left)
        num_after_dot_right = self._get_num_dots(right)

        zeroes_to_add = max(num_after_dot_left, num_after_dot_right) - min(
            num_after_dot_left, num_after_dot_right
        )

        if zeroes_to_add > 0:
            if num_after_dot_left < num_after_dot_right:
                left = (
                    left
                    + ("." if num_after_dot_left == 0 else "")
                    + ("0" * zeroes_to_add)
                )
            else:
                right = (
                    right
                    + ("." if num_after_dot_right == 0 else "")
                    + ("0" * zeroes_to_add)
                )

        left = left[::-1]
        right = right[::-1]

        res = ""
        carry = 0

        len_left = len(left)
        len_right = len(right)

        for i in range(max(len_left, len_right)):

            a, b = left[i] if len_left > i else 0, (right[i] if len_right > i else 0)

            if a == ".":
                res = "." + res
                continue

            a, b = int(a), int(b)

            s = a + b + carry
            carry = s // 10
            res = str(s % 10) + res

        return ArbitraryPrecision(res)

    def __mul__(self, other: "ArbitraryPrecision"):
        left = str(self.value)
        right = str(other.value)

        num_dots_left = self._get_num_dots(left)
        num_dots_right = self._get_num_dots(right)

        left = self._left_shift(left, num_dots_left)
        right = self._left_shift(right, num_dots_right)

        res = ArbitraryPrecision("0")

        for i in range(len(right) - 1, -1, -1):
            if right[i] == ".":
                continue

            temp = ArbitraryPrecision("0")

            for j in range(int(right[i])):
                temp = temp + ArbitraryPrecision(left)

            val_str = self._left_shift(temp.value, len(right) - i - 1)
            val = ArbitraryPrecision(val_str)
            res = res + val

        str_val = self._right_shift(res.value, num_dots_left + num_dots_right)

        return ArbitraryPrecision(str_val)


if __name__ == "__main__":
    import random

    arb1 = ArbitraryPrecision("30.01")
    arb2 = ArbitraryPrecision("10.10")
    print(arb1 * arb2)
