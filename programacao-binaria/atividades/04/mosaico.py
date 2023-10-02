
def circle_packing_mosaico(radii=[]):
    return 0


# 1 - 7  - 1,2,. . . ,7
# 2 - 15 - 1, 2 e 3 (5× cada)
# 3 - 30 - 1, 2 e 3 (10× cada)
# 4 - 50 - 1, 2, 3, 4 e 5 (10× cada)
# 5 - 100 - 1,2,. . . ,10 (10× cada)
def __main__():
    radii = [1,1]
    awnser = circle_packing_mosaico(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"test")

    # radii = [1,2,3,4,5,6,7]
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_1")

    # radii = [1]*5 + [2]*5 + [3]*5
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_2")

    # radii = [1]*10 + [2]*10 + [3]*10
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_3")

    # radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_4")

    # radii = [1]*10 + [2]*10 + [3]*10 + [4]*10 + [5]*10 + [6]*10 + [7]*10 + [8]*10 + [9]*10 + [10]*10
    # awnser = circle_packing_gurobi(radii=radii)
    # print(awnser)
    # plot_solution(awnser,"instancia_5")
__main__()