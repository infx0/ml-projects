"""Solve constrained maximization problems with a tableau-based simplex method.

The implementation adds slack variables, tracks basic and non-basic variables,
selects pivot columns and rows, updates the tableau until optimality, and returns the
optimal objective value and solution while exposing intermediate analysis in tables.
"""

import numpy as np
import pandas as pd


def simplex_algorithm(c, A, b):
    """
    Solve the Linear Programming problem using the Simplex Method.

    Maximize: c^T * x
    Subject to: A * x <= b, x >= 0

    Parameters:
    - c: Coefficients of the objective function (1D numpy array).
    - A: Constraint coefficients (2D numpy array).
    - b: Constraint bounds (1D numpy array).

    Returns:
    - Optimal value and solution.
    """
    m, n = A.shape  # O(1)

    # Add slack variables to convert inequalities to equalities
    slack_vars = np.eye(m)  # O(m^2)
    tableau = np.hstack([A, slack_vars, b.reshape(-1, 1)])  # O(mn + m^2)

    # Append the objective row
    obj_row = np.hstack([-c, np.zeros(m + 1)])  # O(m + n)
    tableau = np.vstack([tableau, obj_row])  # O(mn + m^2)

    # Variable tracking
    basic_vars = [n + i for i in range(m)]  # O(m)
    non_basic_vars = list(range(n))  # O(n)

    step = 0  # O(1)

    while True:
        print(f"Step {step}: Tableau")  # O(1)
        df = pd.DataFrame(
            tableau,
            columns=[f"x{i + 1}" for i in range(n + m)] + ["RHS"],
            index=[f"Constraint {i + 1}" for i in range(m)] + ["Objective"],
        )  # O(mn)
        print(df)  # O(1)
        print("\n")  # O(1)

        # Check if the current solution is optimal (no negative coefficients in the objective row)
        if np.all(tableau[-1, :-1] >= 0):  # O(n + m)
            print("Optimal solution found!\n")  # O(1)
            break  # O(1)

        # Choose entering variable (most negative coefficient in the objective row)
        entering = np.argmin(tableau[-1, :-1])  # O(n)

        # Choose leaving variable (minimum positive ratio of RHS to pivot column)
        ratios = []  # O(1)
        for i in range(m):  # O(m)
            if tableau[i, entering] > 0:  # O(1)
                ratios.append(tableau[i, -1] / tableau[i, entering])  # O(1)
            else:
                ratios.append(np.inf)  # O(1)
        leaving = np.argmin(ratios)  # O(m)

        if ratios[leaving] == np.inf:  # O(1)
            raise ValueError("Problem is unbounded!")  # O(1)

        # Pivot operation
        pivot = tableau[leaving, entering]  # O(1)
        tableau[leaving, :] /= pivot  # O(n)

        for i in range(m + 1):  # O(m)
            if i != leaving:  # O(1)
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]  # O(m + n)

        # Update basic and non-basic variables
        basic_vars[leaving] = entering  # O(1)

        step += 1  # O(1)

    # Extract solution
    solution = np.zeros(n + m)  # O(greater of n^2,m^2)
    for i, var in enumerate(basic_vars):  # O(m)
        if var < n:  # O(1)
            solution[var] = tableau[i, -1]  # O(1)

    optimal_value = tableau[-1, -1]  # O(1)

    print("Optimal Value:", optimal_value)  # O(1)
    print("Solution:", solution[:n])  # O(n)

    return optimal_value, solution[:n]  # O(n)


c = np.array([3, 5])  # Coefficients of the objective function # O(1)
A = np.array([[1, 2], [3, 2]])  # Coefficients of the constraints # O(1)
b = np.array([8, 12])  # RHS of the constraints # O(1)

simplex_algorithm(c, A, b)
