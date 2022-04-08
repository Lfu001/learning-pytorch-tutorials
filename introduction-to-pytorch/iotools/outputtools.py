def print_divider(label=None, border="=", length=40):
    if not isinstance(border, str):
        raise TypeError("Parameter border expects str, got {}.".format(type(border)))
    elif len(border) != 1:
        raise ValueError("Parameter border expects str of length 1, got {}.".format(len(border)))
    divider = border * 2
    if label:
        if not isinstance(label, str):
            raise TypeError("Parameter label expects str, got {}.".format(type(label)))
        divider += " " + label + " "
    if not isinstance(length, int) and length > 0:
        raise ValueError("Parameter length expects int (>0), got {}.".format(length))
    else:
        if len(divider) < length:
            divider += border * (length - len(divider))
    print()
    print(divider)
    print()
