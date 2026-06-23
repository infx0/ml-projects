from typing import List, Tuple

prisoners_dilemma = [[(-5, -5), (-1, -10)], [(-10, -1), (-2, -2)]]


def successors(
    current_state: tuple[list, list],
    dominated_rows: list[int],
    dominated_cols: list[int],
) -> list[tuple]:
    """
        `successors` creates the child states in the state space search for SEDS using the current state and the dominated rows/columns. The state is defined by a vector of valid strategies for each player. A child is created for each removed row or column in the payoff matrix. **Used by**: [solve_game](#solve_game)

    * **current_state** tuple[list, list]: the current state, with the valid rows and columns remaining in the payoff matrix.
    * **dominated_rows** list[int]: the dominated rows for the current state, derived from either weak or strong dominance.
    * **dominated_cols** list[int]: the dominated columns for the current state, derived from either weak or strong dominance.

    **returns** list[tuple]: the child nodes given the current state and dominated rows/columns.
    """
    children = []
    for row in dominated_rows:
        tmp_rows = current_state[0].copy()
        if row in tmp_rows:
            tmp_rows.remove(row)
            children.append((tmp_rows, current_state[1].copy()))
    for col in dominated_cols:
        tmp_cols = current_state[1].copy()
        if col in tmp_cols:
            tmp_cols.remove(col)
            children.append((current_state[0].copy(), tmp_cols))
    return children


def terminal(current_state: tuple[list, list]) -> bool:
    """
        `terminal` creates a flag as to whether the current state is an ending point in the SEDS state space search. **Used by**: [solve_game](#solve_game)

    * **current_state** tuple[list, list]: the current state, with the valid rows and columns remaining in the payoff matrix.

    **returns** bool: the flag as to whether the current state is an ending point in the search.
    """
    valid_rows, valid_cols = current_state
    if len(valid_rows) == 1 and len(valid_cols) == 1:
        return True
    return False


def get_dominated_columns(
    game: list[list[tuple]], current_state: tuple[list, list], weak: bool
) -> list[int]:
    """
        `get_dominated_columns` find the dominated columns given the current state, the 2-player payoff matrix, and whether strong or weak dominance is being used, as part of the SEDS state space search. **Used by**: [solve_game](#solve_game)

    * **game** list[list[tuple]]: the payoff matrix for the two players.
    * **current_state** tuple[list, list]: the valid row and column strategies in the payoff matrix.
    * **weak** bool: whether to use the strong or weak dominance criteria in removing columns.

    **returns** list[int]: the list of valid columns after evaluating the payoff matrix for dominance criteria.
    """
    dominated_cols = []
    valid_rows, valid_cols = current_state
    for i in valid_cols:
        for j in valid_cols:
            if i == j:
                continue
            if weak:
                if all(game[k][i][1] <= game[k][j][1] for k in valid_rows) and any(
                    game[k][i][1] < game[k][j][1] for k in valid_rows
                ):
                    dominated_cols.append(i)
                    break
            else:
                if all(game[k][i][1] < game[k][j][1] for k in valid_rows):
                    dominated_cols.append(i)
                    break
    return dominated_cols


def get_dominated_rows(
    game: list[list[tuple]], current_state: tuple[list, list], weak: bool
) -> list[int]:
    """
        `get_dominated_rows` find the dominated rows given the current state, the 2-player payoff matrix, and whether strong or weak dominance is being used, as part of the SEDS state space search. **Used by**: [solve_game](#solve_game)

    * **game** list[list[tuple]]: the payoff matrix for the two players.
    * **current_state** tuple[list, list]: the valid row and column strategies in the payoff matrix.
    * **weak** bool: whether to use the strong or weak dominance criteria in removing rows.

    **returns** list[int]: the list of valid rows after evaluating the payoff matrix for dominance criteria.
    """
    dominated_rows = []
    valid_rows, valid_cols = current_state
    for i in valid_rows:
        for j in valid_rows:
            if i == j:
                continue
            if weak:
                if all(game[i][k][0] <= game[j][k][0] for k in valid_cols) and any(
                    game[i][k][0] < game[j][k][0] for k in valid_cols
                ):
                    dominated_rows.append(i)
                    break
            else:
                if all(game[i][k][0] < game[j][k][0] for k in valid_cols):
                    dominated_rows.append(i)
                    break
    return dominated_rows


def frontier_setup(game: list[list[tuple]]) -> list[tuple]:
    """
        `frontier_setup` initializes the frontier in the SEDS state space search by populating valid row and column strategies given the payoff matrix. **Used by**: [solve_game](#solve_game)

    * **game** list[list[tuple]]: the description of the game via the payoff matrix.

    **returns** list[tuple]: returns the valid row and column strategy indices.
    """
    num_rows = len(game)
    num_cols = len(game[0])
    valid_rows = [i for i in range(num_rows)]
    valid_cols = [j for j in range(num_cols)]
    frontier = [(valid_rows, valid_cols)]
    return frontier


def solve_game(game: List[List[Tuple]], weak: bool = False) -> List[Tuple]:
    """

    `solve_game` runs the SEDS 2-player DFS state space search for a given game/payoff matrix to find the Nash pure equilibriums. Use of either strong or weak dominance criteria for elimination can be specified using the weak flag. **Uses**: [frontier_setup](#frontier_setup), [get_dominated_rows](#get_dominated_rows), [get_dominated_columns](#get_dominated_columns), [terminal](#terminal), [successors](#successors).

    * **game** list[list[tuple]]: the description of the game via the payoff matrix for the two players.
    * **weak** bool: whether to use strong or weak dominance criteria.

    **returns** list[tuple]: returns the Nash pure equilibriums as strategy indices.
    """
    frontier = frontier_setup(game=game)
    explored = []
    strategy_pairs = []
    while frontier:
        current_state = frontier.pop()
        dominated_rows = get_dominated_rows(game, current_state, weak)
        dominated_cols = get_dominated_columns(game, current_state, weak)
        if terminal(current_state):
            strategies = (current_state[0][0], current_state[1][0])
            strategy_pairs.append(strategies)
        children = successors(current_state, dominated_rows, dominated_cols)
        for child in children:
            if child not in explored and child not in frontier:
                frontier.append(child)
        explored.append(current_state)
    return strategy_pairs


test_game_1 = [
    [(7, 6), (10, 10), (3, 2)],
    [(6, 6), (9, 8), (2, 2)],
    [(5, 4), (5, 5), (1, 2)],
]

solution = solve_game(game=test_game_1)
print(solution)
assert solution == [(0, 1)]  # insert your solution from above.

test_game_2 = [
    [(5, 4), (5, 5), (4, 4)],
    [(5, 4), (4, 4), (4, 4)],
    [(5, 4), (3, 4), (4, 4)],
]

strong_solution = solve_game(test_game_2)
weak_solution = solve_game(test_game_2, weak=True)
print(strong_solution)
print(weak_solution)

assert strong_solution == []
assert weak_solution == [(0, 1)]  # insert your solution from above.

test_game_3 = [
    [(1, 1, (1, 1), (1, 1))],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
]

strong_solution = solve_game(test_game_3)
weak_solution = solve_game(test_game_3, weak=True)

assert strong_solution == []
assert weak_solution == []

test_game_4 = [
    [(1, 0), (3, 1), (1, 1)],
    [(1, 1), (3, 0), (0, 1)],
    [(2, 2), (3, 3), (0, 2)],
]

strong_solution = solve_game(test_game_4)
weak_solution = solve_game(test_game_4, weak=True)
print(strong_solution)
print(weak_solution)

assert strong_solution == []
assert weak_solution == [(0, 2), (2, 1)]  # put solution here
