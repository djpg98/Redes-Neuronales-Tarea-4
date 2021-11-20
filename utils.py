def round_output(x):
    if x <= 0.1:
        return 0
    elif x < 0.9:
        return x
    else:
        return 1
