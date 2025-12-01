def machine_epsilon():
    eps = 1
    while (1 + (eps / 2)) != 1:
        eps /= 2
    print(f"Machine epsilon (precision): {eps}")


def overflow_point():
    import sys

    x = 1.0
    res = 1.0
    while x != float("inf"):
        res = x
        x *= 2
    print(f"Overflow point: {res}")
    print(sys.float_info.max)


if __name__ == "__main__":
    machine_epsilon()
    overflow_point()
