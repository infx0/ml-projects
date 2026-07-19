from IPython.display import display_html
from worlds import small_world, full_world

MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]
COSTS = {"🌾": 1, "🌲": 3, "⛰": 5, "🐊": 7}


def display_emoji_grid(emoji_grid: list[list[str]]) -> None:
    """
    Display a list of Lists of emojis in a perfect grid (table) in a Jupyter
    Notebook.

    Args:
        emoji_grid (list[list[str]]): A 2D list containing emojis to display in a grid.
    """
    # Create HTML table
    html = '<table style="border-collapse: collapse;">'

    for row in emoji_grid:
        html += "<tr>"
        for emoji in row:
            html += f'<td style="border: none; padding: 0px; text-align: center; font-size: 1em;">{emoji}</td>'
        html += "</tr>"

    html += "</table>"

    # Display the HTML table
    display_html(html, raw=True)


def off_world(position: tuple[int, int], world: list[list[str]]) -> bool:
    """
    A helper function to determine if a child node would be off the edges of the
    provided world. Works for any rectangular world. Note the input position should be
    given as (y, x) (i.e. (vertical, horizontal)) in order to display properly.

    Args:
        position (tuple[int, int]): The current position in the world as (y, x).
        world (list[list[str]]): The reference world.

    Returns:
        bool: Whether the position is off-grid.
    """
    are_we_off = False
    vert_lim = len(world) - 1
    horiz_lim = len(world[1]) - 1
    if position[0] < 0 or position[1] < 0:
        are_we_off = True
    elif position[0] > vert_lim:
        are_we_off = True
    elif position[1] > horiz_lim:
        are_we_off = True
    return are_we_off


def get_terrain(position: tuple[int, int], world: list[list[str]]) -> str:
    """
    A helper function that retrieves the terrain from the given world and position. Note
    the input position should be given as (y, x) (i.e. (vertical, horizontal)) in order
    to display properly.

    Args:
        position (tuple[int, int]): The input position as (y, x).
        world (list[list[str]]): The world for evaluation.

    Returns:
        str: The ASCII string representation of the terrain.
    """
    return world[position[0]][position[1]]


def get_grid_squares(
    position: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """
    Retrieves the coordinate tuples between starting tuple and the goal tuple, using a
    Manhattan-type path, traversing horizontally first, then vertically.

    Note the positions are referenced as (y,x), i.e. (vertical, horizontal).

    Args:
        position (tuple[int, int]): The current position in the current reference
            world.
        goal (tuple[int, int]): The goal position in the current reference
            world.

    Returns:
        list[tuple[int, int]]: The tuple list that traverses a grid from the
            starting position to the goal position.
    """
    out = []
    y_start = min(position[0], goal[0])
    y_end = max(position[0], goal[0])
    x_start = min(position[1], goal[1])
    x_end = max(position[1], goal[1])
    for x in range(x_start, x_end):
        out.append((y_start, x))
    for y in range(y_start, y_end):
        out.append((y, x_end))
    if goal[1] >= position[1] and goal[0] >= position[0]:
        out.append(goal)
    else:
        out.append(position)
    return out


def heuristic(
    position: tuple[int, int],
    goal: tuple[int, int],
    world: list[list[str]],
    costs: dict[str, int],
) -> int:
    """
    The heuristic function, h(n), which is used as part of the total cost in choosing
    the next node in A* search. This particular heurisic uses Manhattan distance from
    the current position to the goal position, incorporating terrain costs along the
    way. It adds a heavy penalty if the estimated path intersects mountains.

    Args:
        position (tuple[int, int]): The current position in the world.
        goal (tuple[int, int]): The goal position in the world.
        world (list[list[str]]): The reference world.
        costs (dict[str, int]): The cost per each terrain type.

    Returns:
        int: The cost to the traverse the grid squares.
    """
    grid_squares = get_grid_squares(position, goal)
    path_cost = 0
    for square in grid_squares:
        if get_terrain(position=square, world=world) == "🌋":
            path_cost += 99
        else:
            path_cost += costs[get_terrain(position=square, world=world)]
    return path_cost


def update_path_cost(
    explored: list[tuple[int, int]],
    current_state: tuple[int, int],
    costs: dict[str, int],
    world: list[list[str]],
) -> int:
    """
    Calculates the cost to get from the starting position to the current node.

    Args:
        explored (list[tuple[int, int]]): The list of explored nodes.
        current_state (tuple[int, int]): The current node in the search.
        costs (dict[str, int]): The cost per terrain type.
        world (list[list[str]]): The reference world.

    Returns:
        int: The cost to get from the starting position to the current node.
    """
    path_cost = 0
    explored.pop(0)
    for node in explored:
        path_cost += costs[get_terrain(node, world)]
    path_cost += costs[get_terrain(current_state, world)]
    return path_cost


def get_moves(explored: list[tuple[int, tuple]]) -> list[tuple]:
    """
    Takes the explored list and modulates it into a list of moves, which are offsets
    from the previous node to the current node. Note the input explored list uses a
    (y,x) reference frame, but the output moves are in a (x,y) per the assignment
    specifications. The list of moves is returned from a_star_search.

    Args:
        explored (list[tuple[int, tuple]]): The list of explored nodes.

    Returns:
        list[tuple]: The list of coordinate offsets, in (x,y) reference frame.
    """
    moves = []
    explored_next = explored[1:]
    for node, node_next in zip(explored, explored_next):
        coord = node[1]
        coord_next = node_next[1]
        move = (coord_next[1] - coord[1], coord_next[0] - coord[0])
        moves.append(move)
    return moves


def get_path_coords(
    start: tuple[int, int], path: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Unpacks a list of moves and modulates them into a list of coordinates. Note this
    uses the (x,y) reference frame for both input and output.

    Args:
        start (tuple[int, int]): The starting position as a tuple.
        path (list[tuple[int ,int]]): The list of offsets to use in calculating world
            positions.

    Returns:
        list[tuple[int, int]]: The list of world positions in (x,y) reference frame.
    """
    x = start[0]
    y = start[1]
    coords = [(x, y)]
    for node in path:
        x += node[0]
        y += node[1]
        coords.append((x, y))
    return coords


def successors(
    current_state: tuple[int, tuple],
    world: list[list[str]],
    costs: dict[str, int],
    moves: list[tuple[int, int]],
    heuristic: callable,
    goal: tuple[int, int],
) -> list[tuple[int, tuple]]:
    """
    Creates the child nodes from the current node as well as associated total cost,
    f[n]. It also takes into account nodes that would be off-grid and non-traversable
    terrain.

    Args:
        current_state (tuple[int, tuple]): The current node.
        world (list[list[str]]): The reference world.
        costs (dict[str, int]): The cost per terrain type.
        moves (list[tuple[int, int]]): The possible moves for choosing the next
            node.
        heuristic (callable): Part of the evaluation function for used in
            selecting the next node.
        goal (tuple[int, int]): The goal position in the reference world.

    Returns:
        list[tuple[int, tuple]]: The list of child nodes with associated costs.
    """
    children: list[tuple[int, tuple]] = []
    for move in moves:
        child_position: tuple[int, int] = (
            current_state[1][0] + move[0],
            current_state[1][1] + move[1],
        )
        if off_world(position=child_position, world=world):
            continue
        child_terrain: str = get_terrain(position=child_position, world=world)
        if child_terrain == "🌋":
            continue
        child_terrain_cost: int = costs[child_terrain]
        g_n = current_state[0] + child_terrain_cost
        h_n = heuristic(child_position, goal, world, costs)
        f_n: int = g_n + h_n
        children.append((f_n, child_position))
    return children


def a_star_search(
    world: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    costs: dict[str, int],
    moves: list[tuple[int, int]],
    heuristic: callable,
) -> list[tuple[int, int]]:
    """
    Implements the A* search algorithm, which combines best-first search with a
    heuristic function that estimates the cost from a node to the goal.

    Args:
        world (list[list[str]]): The actual context for the navigation problem.
        start (tuple[int, int]): The starting location of the bot, `(x, y)`.
        goal (tuple[int, int]): The desired goal position for the bot, `(x, y)`.
        costs (dict[str, int]): Is a `dict` of costs for each type of terrain in
            **world**.
        moves (list[tuple[int, int]]): The legal movement model expressed in
            offsets in **world**.
        heuristic (callable): Is a heuristic function, $h(n)$.

    Returns:
        list[tuple[int, int]]: The offsets needed to get from start state to the
            goal as a `list`.
    """
    frontier: list[tuple[int, tuple]] = [(0, (start[1], start[0]))]
    explored: list[tuple[int, tuple]] = []
    goal = (goal[1], goal[0])
    while frontier:
        frontier = sorted(frontier)
        current_state = frontier.pop(0)
        explored_coords = [node[1] for node in explored]
        current_coord = current_state[1]
        if explored:
            state_path_cost = update_path_cost(
                explored_coords, current_coord, costs, world
            )
            current_state = (state_path_cost, current_state[1])
        if current_state[1] == goal:
            explored.append(current_state)
            print(explored)
            moves = get_moves(explored)
            return moves
        children = successors(current_state, world, costs, moves, heuristic, goal)
        frontier_positions = [state[1] for state in frontier]
        explored_positions = [state[1] for state in explored]
        for child in children:
            if (
                child[1] not in frontier_positions
                and child[1] not in explored_positions
            ):
                frontier.append(child)
        explored.append(current_state)
    return []


def pretty_print_path(
    world: list[list[str]],
    path: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    costs: dict[str, int],
) -> int:
    """
    Prints the world and associated path traversal through the world in a readable grid
    format. At each step, a direction marker is overlaid on top of the original world
    that indicates the direction of the greatest change. It also provides the total path
    cost.

    Args:
        world (list[list[str]]): The world (terrain map) for the path to be
            printed upon.
        path (list[tuple[int, int]]): The path from start to goal, in offsets.
        start (tuple[int, int]): The starting location for the path.
        goal (tuple[int, int]): The goal location for the path.
        costs (dict[str, int]): The costs for each action.

    Returns:
        int: The path cost.
    """
    coords = get_path_coords(start, path)
    path_cost = 0
    for coord in coords:  # (x,y) reference frame
        path_cost += costs[get_terrain(position=(coord[1], coord[0]), world=world)]
    for p, coord in zip(path, coords):
        max_val = max((abs(p[0]), abs(p[1])))
        if max_val == -p[0] or max_val == -p[1]:
            max_val = -1 * max_val
        max_idx = p.index(max_val)
        if max_idx == 0 and p[0] < 0:
            world[coord[1]][coord[0]] = "⏪"
        elif max_idx == 0 and p[0] > 0:
            world[coord[1]][coord[0]] = "⏩"
        if max_idx == 1 and p[1] < 0:
            world[coord[1]][coord[0]] = "⏫"
        elif max_idx == 1 and p[1] > 0:
            world[coord[1]][coord[0]] = "⏬"
        world[goal[0]][goal[1]] = "🎁"
    display_emoji_grid(world)
    return path_cost


if __name__ == "__main__":
    small_start = (0, 0)
    small_goal = (len(small_world[0]) - 1, len(small_world) - 1)
    small_path = a_star_search(
        small_world, small_start, small_goal, COSTS, MOVES, heuristic
    )
    small_path_cost = pretty_print_path(
        small_world, small_path, small_start, small_goal, COSTS
    )
    print(f"total path cost: {small_path_cost}")
    print(small_path)

    full_start = (0, 0)
    full_goal = (len(full_world[0]) - 1, len(full_world) - 1)
    full_path = a_star_search(
        full_world, full_start, full_goal, COSTS, MOVES, heuristic
    )
    full_path_cost = pretty_print_path(
        full_world, full_path, full_start, full_goal, COSTS
    )
    print(f"total path cost: {full_path_cost}")
    print(full_path)
