def print_divider(label=None, length=40):
    divider = "=="
    if label:
        if not isinstance(label, str):
            raise ValueError("Parameter label expects str, got {}.".format(type(label)))
        divider += " " + label + " "
    if not isinstance(length, int) and length > 0:
        raise ValueError("Parameter length expects int (>0), got {}.".format(length))
    else:
        if len(divider) < length:
            divider += "=" * (length - len(divider))
    print()
    print(divider)
    print()
