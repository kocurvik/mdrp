def depth_indices(depth):
    if depth == 1: # real
        return (8, 9)
    elif depth == 2: # midas
        return (10, 11)
    elif depth == 3:  # dpt
        return (12, 13)
    elif depth == 4: # zoe
        return (14, 15)
    elif depth == 5: # depth anyV1B
        return (16, 17)
    elif depth == 6:  # depth anyV2B
        return (18, 19)
    elif depth == 7: # apple depth pro
        return (20, 21)
    elif depth == 8:  # metric 3d
        return (22, 23)
    elif depth == 9:  # marigold e2e
        return (24, 25)
    elif depth == 10:  # moge
        return (26, 27)
    elif depth == 11:  # marigold
        return (28, 29)
    elif depth == 12:  # unidepth
        return (30, 31)
