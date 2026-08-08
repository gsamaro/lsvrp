import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProblemData:
    raw_map: dict
    p: int
    i: int
    k: int
    t: int
    v: int
    B: int
    b_p: list
    c_p: list
    s_p: list
    M: int
    U_p_i: list
    I_p_i_0: list
    h_p_i: list
    C: int
    f: float
    a_i_k: list
    d_p_i_t: list

    @classmethod
    def from_map(cls, map_data):
        return cls(
            raw_map=map_data,
            p=map_data["num_products"],
            i=map_data["num_customers"] + 1,
            k=map_data["num_customers"] + 1,
            t=map_data["num_periods"],
            v=map_data["num_vehicles"],
            B=map_data["B"],
            b_p=map_data["b_p"],
            c_p=map_data["c_p"],
            s_p=map_data["s_p"],
            M=map_data["M"],
            U_p_i=map_data["U_pi"],
            I_p_i_0=map_data["I_pi0"],
            h_p_i=map_data["h_pi"],
            C=map_data["C"],
            f=map_data["f"],
            a_i_k=map_data["a_ik"],
            d_p_i_t=map_data["d_pit"],
        )


def empty_solution(problem: ProblemData):
    return {
        "X": np.zeros((problem.p, problem.t), dtype=int),
        "Y": np.zeros((problem.p, problem.t), dtype=int),
        "I": np.zeros((problem.p, problem.i, problem.t), dtype=int),
        "Q": np.zeros((problem.p, problem.v, problem.i, problem.t), dtype=int),
        "R": np.zeros((problem.p, problem.v, problem.i, problem.k, problem.t), dtype=int),
        "Z": np.zeros((problem.v, problem.i, problem.k, problem.t), dtype=int),
    }


def clone_solution(solution):
    cloned = {}
    for key, value in solution.items():
        if isinstance(value, np.ndarray):
            cloned[key] = np.array(value, copy=True)
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def evaluate_solution_cost(problem: ProblemData, variables):
    total = 0

    for p in range(problem.p):
        for t in range(problem.t):
            total += problem.s_p[p] * int(variables["Y"][p, t])
            total += problem.c_p[p] * int(variables["X"][p, t])

    for p in range(problem.p):
        for i in range(problem.i):
            for t in range(problem.t):
                total += problem.h_p_i[p][i] * int(variables["I"][p, i, t])

    for v in range(problem.v):
        for k in range(1, problem.k):
            for t in range(problem.t):
                total += problem.f * int(variables["Z"][v, 0, k, t])

    for v in range(problem.v):
        for i in range(problem.i):
            for k in range(problem.k):
                if i == k:
                    continue
                for t in range(problem.t):
                    total += problem.a_i_k[i][k] * int(variables["Z"][v, i, k, t])

    return total


def build_results_from_variables(problem: ProblemData, variables, objective_value, elapsed_time):
    Z = []
    for t in range(problem.t):
        v_list = []
        for v in range(problem.v):
            i_list = []
            for i in range(problem.i):
                k_list = []
                for k in range(problem.k):
                    k_list.append(int(variables["Z"][v, i, k, t]))
                i_list.append(k_list)
            v_list.append(i_list)
        Z.append(v_list)

    Y = []
    for t in range(problem.t):
        p_list = []
        for p in range(problem.p):
            p_list.append(int(variables["Y"][p, t]))
        Y.append(p_list)

    X = []
    for t in range(problem.t):
        p_list = []
        for p in range(problem.p):
            p_list.append(int(variables["X"][p, t]))
        X.append(p_list)

    I = []
    for t in range(problem.t):
        i_period = []
        for i in range(problem.i):
            i_values = []
            for p in range(problem.p):
                i_values.append(int(variables["I"][p, i, t]))
            i_period.append(i_values)
        I.append(i_period)

    R = []
    for t in range(problem.t):
        t_list = []
        for v in range(problem.v):
            v_list = []
            for p in range(problem.p):
                p_list = []
                for i in range(problem.i):
                    i_list = []
                    for k in range(problem.k):
                        i_list.append(float(variables["R"][p, v, i, k, t]))
                    p_list.append(i_list)
                v_list.append(p_list)
            t_list.append(v_list)
        R.append(t_list)

    Q = []
    for t in range(problem.t):
        t_list = []
        for v in range(problem.v):
            v_list = []
            for p in range(problem.p):
                p_list = []
                for i in range(problem.i):
                    p_list.append(int(variables["Q"][p, v, i, t]))
                v_list.append(p_list)
            t_list.append(v_list)
        Q.append(t_list)

    return (
        Z,
        X,
        Y,
        I,
        R,
        Q,
        [],
        objective_value,
        0.0,
        elapsed_time,
        None,
        1,
        0,
        0,
        objective_value,
        None,
    )
