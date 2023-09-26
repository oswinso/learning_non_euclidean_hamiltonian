def flat_join(arr, fmt="11.5e") -> str:
    return join(arr.flatten(), fmt)


def join(arr, fmt="11.5e") -> str:
    assert arr.ndim == 1

    fmt_string = "{{:{}}}".format(fmt)
    s = [fmt_string.format(n) for n in arr]
    s = " ".join(s)
    return "[{}]".format(s)
