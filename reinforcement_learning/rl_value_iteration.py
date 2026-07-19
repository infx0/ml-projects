def read_world(filename):
    result = []
    with open(filename) as f:
        for line in f.readlines():
            if len(line) > 0:
                result.append(list(line.strip()))
    return result


def init_rewards(
    world: list[list[str]], costs: dict, goal_pos: tuple, goal_reward: float
) -> dict[tuple, float]:
    """
    Initializes a reward dict with an associated reward for each state in the applicable
    world, for use in RL value iteration.

    Args:
        world (list[list[str]]): The input world from which to initialize the rewards.
        costs (dict): The cost per specific terrain type.
        goal_pos (tuple): The position of the goal state in the world as a (row, col)
            tuple.
        goal_reward (float): The reward for the goal state.

    Returns:
        dict[tuple, float]: The dict with the reward in each state.
    """
    num_rows = len(world)
    num_cols = len(world[0])
    rewards = {}
    for row in range(num_rows):
        for col in range(num_cols):
            if world[row][col] == "x":
                continue
            rewards[(row, col)] = costs[world[row][col]]
    rewards[goal_pos] = goal_reward
    return rewards


def init_val_matrix(world: list[list[str]]) -> dict[tuple, float]:
    """
    Initializes the value matrix to zero, but as a dict, using a tuple key for the
    location, and float for the value.

    Args:
        world (list[list[str]]): The input world to use for reinforcement learning
            training.

    Returns:
        dict[tuple, float]: The value dict.
    """
    num_rows = len(world)
    num_cols = len(world[0])
    value = {}
    for row in range(num_rows):
        for col in range(num_cols):
            if world[row][col] == "x":
                continue
            value[(row, col)] = 0.0
    return value


def find_mountains(world: list[list[str]]) -> set[tuple]:
    """
    Compiles a set of mountain locations as tuples, based on the input world, for use in
    determining what the valid actions are in the value iteration algorithm.

    Args:
        world (list[list[str]]): The input world from which to create the mountain
            location set.

    Returns:
        set[tuple]: The set of mountain locations as tuples.
    """
    mountains = []
    num_rows = len(world)
    num_cols = len(world[0])
    for row in range(num_rows):
        for col in range(num_cols):
            if world[row][col] == "x":
                mountains.append((row, col))
    return set(mountains)


def valid_pos(world: list[list], mountains: list[tuple], pos: tuple) -> bool:
    """
    Tests whether a successor state is a valid position within the given world. It
    checks if the successor state is off the map, or on a mountain, which is not
    allowed. It returns a bool indicating if the position is valid, allowing the main
    value iteration algorithm to assess the action.

    Args:
        world (list[list]): The input world from which to assess valid positions.
        mountains (list[tuple]): A list of mountain locations within the world.
        pos (tuple): A tuple of the (row,col) coordinates for evaluation.

    Returns:
        list[tuple]: The list of mountain locations as tuples.
    """
    max_row = len(world)
    max_col = len(world[0])
    row_pos = pos[0]
    col_pos = pos[1]
    if (
        row_pos < 0
        or row_pos >= max_row
        or col_pos < 0
        or col_pos >= max_col
        or (pos in mountains)
    ):
        return False
    return True


def convergence(
    value1: dict[tuple, float], value2: dict[tuple, float], epsilon: float
) -> bool | None:
    """
    Tests to see if the value matrix has converged by comparing the max value of the
    absolute difference of V with V_last. It will return none if the matrix elements
    don't match (earlier in the code states with mountains are skipped, and these
    "holes" should end up as identical between V and V_last).

    Args:
        value1 (dict[tuple, float]): The first value matrix for comparison.
        value2 (dict[tuple, float]): The second value matrix for comparison.
        epsilon (float): The threshold value that controls the bool that's returned.

    Returns:
        bool | None: Returns None if there was an error, True if the algorithm has
            converged, or False if it hasn't.
    """
    keys1 = set(value1.keys())
    keys2 = set(value2.keys())
    diff_vals = []
    if keys1 != keys2:
        return None
    for key in keys1:
        val = abs(value1[key] - value2[key])
        diff_vals.append(val)
    if max(diff_vals) < epsilon:
        return True
    else:
        return False


def best_action_value(
    curr_pos: tuple,
    actions: list,
    world: list[list],
    mountains: list,
    rewards: dict,
    gamma: float,
    V_last: dict,
) -> tuple[float, tuple]:
    """
    Finds the max of Q summed across the possible actions for stochastic value
    iteration, and the corresponding argmax of Q.

    Args:
        curr_pos (tuple): The current position of the agent in the world as a tuple.
        actions (list): The possible actions the agent can take.
        world (list[list]): The input world for the agent to use.
        mountains (list): The mountain locations in the world.
        rewards (dict): The reward for each world state based on terrain type.
        gamma (float): The discount value.
        V_last (dict): The previous values for the value matrix.

    Returns:
        tuple[float, tuple]: Returns None if there was an error, True if the algorithm
            has converged, or False if it hasn't.
    """
    max_q = -1e10
    best_a = (0, 0)
    succeed_prob = 0.7
    fail_prob = 0.1
    for intended_action in actions:
        expected = 0.0
        for actual_action in actions:
            if actual_action == intended_action:
                prob = succeed_prob
            else:
                prob = fail_prob
            successor_pos = (
                curr_pos[0] + actual_action[0],
                curr_pos[1] + actual_action[1],
            )
            if not valid_pos(world, mountains, successor_pos):
                successor_pos = curr_pos
            expected += prob * V_last[successor_pos]
        q = rewards[curr_pos] + gamma * expected
        if q > max_q:
            max_q = q
            best_a = intended_action
    return max_q, best_a


def value_iteration(
    world: list[list],
    costs: dict,
    goal: tuple,
    reward: float,
    actions: list,
    gamma: float,
) -> dict[tuple, tuple]:
    """
    Runs the stochastic value iteration algorithm for reinforcement learning. It uses an
    input world, a dictionary of costs per terrain type, a goal position, the reward for
    that goal position, a list of actions, and a discount value to determine the optimal
    policy. It checks for convergence at the end of each iteration, and also bails after
    max_iters iterations if it hasn't converged by then, to avoid infinite loops.

    Args:
        world (list[list]): The world to run the value iteration algorithm on.
        costs (dict): The cost lookup per terrain type.
        goal (tuple): The goal position within the world.
        reward (float): The reward value for reaching the goal.
        actions (list): The possible actions that can be taken in each state.
        gamma (float): The discount value for the algorithm.

    Returns:
        dict[tuple, tuple]: Returns the optimal policy as a dict with the state/position
            as key and the cardinal move as value.
    """
    epsilon = 0.01
    policy = {}
    rewards = init_rewards(world=world, costs=costs, goal_pos=goal, goal_reward=reward)
    V, mountains = init_val_matrix(world), find_mountains(world)
    iteration = 0
    max_iters = 10000
    for _ in range(max_iters):
        V_last = V.copy()
        for row in range(len(world)):
            for col in range(len(world[0])):
                curr_pos = (row, col)
                if not valid_pos(world, mountains, curr_pos):
                    continue
                V[curr_pos], policy[curr_pos] = best_action_value(
                    curr_pos, actions, world, mountains, rewards, gamma, V_last
                )
        if convergence(V, V_last, epsilon):
            print(f"Converged after {iteration} iterations.")
            return policy
        iteration += 1
    print(f"Did not converage after {max_iters} iterations.")


def pretty_print_policy(
    cols: int, rows: int, policy: dict[tuple, tuple], goal: tuple
) -> None:
    """
    Print the policy derived from the stochastic value iteration algorithm as a
    sequence of string rows and columns.

    Args:
        cols (int): The number of columns in the world.
        rows (int): The number of rows in the world.
        policy (dict[tuple, tuple]): The policy derived from the value iteration
            algorithm, as a dict of states and actions.
        goal (tuple): The state/position of the goal.
    """
    actions = {(0, 1): ">", (0, -1): "<", (1, 0): "v", (-1, 0): "^"}
    grid = [["x" for _ in range(cols)] for _ in range(rows)]
    for k, v in policy.items():
        row = k[0]
        col = k[1]
        grid[row][col] = actions[v]
    grid[goal[0]][goal[1]] = "G"
    for row in grid:
        print("".join(row))


if __name__ == "__main__":
    costs = {".": -1, "*": -3, "^": -5, "~": -7}
    cardinal_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # small world
    small_world = read_world("small.txt")
    num_rows = len(small_world)
    num_cols = len(small_world[0])
    goal = (num_rows - 1, num_cols - 1)
    gamma = 0.9
    small_policy = value_iteration(
        world=small_world,
        costs=costs,
        goal=goal,
        reward=1000,
        actions=cardinal_moves,
        gamma=gamma,
    )
    rows = len(small_world)
    cols = len(small_world[0])
    pretty_print_policy(cols, rows, small_policy, goal)

    # large world
    large_world = read_world("large.txt")
    num_rows = len(large_world)
    num_cols = len(large_world[0])
    goal = (num_rows - 1, num_cols - 1)
    gamma = 0.9
    large_policy = value_iteration(
        world=large_world,
        costs=costs,
        goal=goal,
        reward=1e8,
        actions=cardinal_moves,
        gamma=gamma,
    )
    rows = len(large_world)
    cols = len(large_world[0])
    pretty_print_policy(cols, rows, large_policy, goal)
