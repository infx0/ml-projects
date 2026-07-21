"""
This script solves map-coloring problems by treating each map as a constraint
satisfaction problem (CSP). A map is represented as a graph: nodes are regions,
edges connect adjacent regions, and each node's domain is the list of colors it
may use. The solver assigns colors so that no two adjacent regions share the
same color.

The main solving path starts with color_map(), which augments the graph with
color domains and then runs a recursive backtracking search. During that search,
the script chooses the next unassigned region with the degree heuristic, tries
colors in least-constraining-value order, checks assignments against already
colored neighbors, and uses forward checking to prune impossible colors from
neighboring domains before recursing. If a branch leaves any unassigned region
with no valid colors, the search backtracks and tries another assignment.

The draw_map() helper visualizes a graph with NetworkX and Matplotlib. When run
directly, the script defines sample Connecticut and Europe maps, colors them
with four and then three colors, verifies that adjacent regions do not share a
color, and draws the resulting colored maps.
"""

import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy


def draw_map(name, planar_map, size, color_assignments=None) -> None:
    """
    Draws a map represented as a graph, using the provided coordinates for node
    placement and the provided labels for each node. If color assignments are passed
    in, it colors the nodes with those assignments; otherwise, it colors all nodes red.

    Args:
        name (str): The title to display above the drawn map.
        planar_map (dict): The graph dictionary to draw. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]}.
        size (tuple): The figure size to use when drawing the map.
        color_assignments (list[tuple[str, str]] | None): The optional list of node
            color assignments to use when coloring the graph nodes.

    Returns:
        None
    """

    def as_dictionary(a_list):
        dct = {}
        for i, e in enumerate(a_list):
            dct[i] = e
        return dct

    G = nx.Graph()

    labels = as_dictionary(planar_map["nodes"])
    pos = as_dictionary(planar_map["coordinates"])

    # create a List of Nodes as indices to match the "edges" entry.
    nodes = [n for n in range(0, len(planar_map["nodes"]))]

    if color_assignments:
        colors = [c for n, c in color_assignments]
    else:
        colors = ["red" for c in range(0, len(planar_map["nodes"]))]

    G.add_nodes_from(nodes)
    G.add_edges_from(planar_map["edges"])

    plt.figure(figsize=size, dpi=100)
    plt.title(name)
    nx.draw(G, node_color=colors, with_labels=True, labels=labels, pos=pos)


def order_domain_values(
    csp: dict, var: str, assignment: list[tuple[str, str]]
) -> list[str]:
    """
    Orders the values (in this case color strings) for a given node, as part of solving
    a CSP problem. It uses the least constraining value heuristic, which orders the
    values by the number of unassigned neighbors with the same value. Ties are broken
    alphabetically.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]
            "domains": dict[str, list[str]]}.
        var (str): The node in the CSP to consider.
        assignment (list[tuple[str, str]]): The list of assignments made so far in the
            CSP.

    Returns:
        list[str]: The order of the colors to evaluate for a given node in the CSP.
    """
    assigned_nodes = [node for (node, _) in assignment]
    node_idx = csp["nodes"].index(var)
    edges_with_idx = [edge for edge in csp["edges"] if node_idx in edge]
    neighbors_with_color = {}
    for color in csp["domains"][var]:
        neighbors_with_color[color] = 0
        for edge in edges_with_idx:
            neighbor_idx = edge[1] if node_idx == edge[0] else edge[0]
            neighbor = csp["nodes"][neighbor_idx]
            if neighbor in assigned_nodes:
                continue
            if color in csp["domains"][neighbor]:
                neighbors_with_color[color] += 1
    sorted_neighbors_with_color = sorted(
        neighbors_with_color.items(), key=lambda kv: (kv[1], kv[0])
    )
    color_order = [color for color, _ in sorted_neighbors_with_color]
    return color_order


def inference(
    csp: dict, var: str, assignment: list[tuple[str, str]], trace: bool
) -> list[tuple[str, str]] | None:
    """
    Performs forward checking as part of the CSP algorithm we're using. It finds the
    neighbors of the current node, looks at the proposed assignment, and prunes that
    assignment from the neighbor domains. It returns None if any of the propagated
    prunings result in an empty neighbor domain.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]
            "domains": dict[str, list[str]]}.
        var (str): The node in the CSP to consider.
        assignment (list[tuple[str, str]]): The list of assignments made so far in the
            CSP.
        trace (bool): A debugging flag for whether to print verbose outputs.

    Returns:
        list[tuple[str,str]] | None: None if the inferences/prunings caused an empty
            domain, or the inferences performed so they can added back in later if
            needed during backtracking.
    """
    inferences = []
    assignment_dict = dict(assignment)
    var_color = assignment_dict[var]
    node_idx = csp["nodes"].index(var)
    edges_with_idx = [edge for edge in csp["edges"] if node_idx in edge]
    for edge in edges_with_idx:
        neighbor_idx = edge[1] if node_idx == edge[0] else edge[0]
        neighbor = csp["nodes"][neighbor_idx]
        if neighbor in assignment_dict:
            continue
        if var_color in csp["domains"][neighbor]:
            csp["domains"][neighbor].remove(var_color)
            inferences.append((neighbor, var_color))
            if not csp["domains"][neighbor]:
                if trace:
                    print(f"empty domain for node {neighbor}, backtracking")
                return None
    if trace:
        print(f"tree prunings from inference function {inferences}")
    return inferences


def select_unassigned_variable(
    csp: dict, assignment: list[tuple[str, str]], trace: bool
) -> str:
    """
    Uses the degree heuristic to select the next unassigned variable in solving a
    constraint satisfaction problem. It finds the unassigned variables, checks how many
    unassigned connections each of those variables has, and then selects the variable
    with the most connections. Any ties are broken by alphabetical order.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]
            "domains": dict[str, list[str]]}.
        assignment (list[tuple[str, str]]): The list of assignment made so far in the
            CSP.
        trace (bool): A debugging flag for whether to print verbose outputs.

    Returns:
        str: The next variable to be assigned in the CSP.
    """
    assigned_nodes = [var for (var, _) in assignment]
    unassigned_nodes = [node for node in csp["nodes"] if node not in assigned_nodes]
    num_neighbors = {}
    for node in unassigned_nodes:
        num_neighbors[node] = 0
        node_idx = csp["nodes"].index(node)
        edges_with_idx = [edge for edge in csp["edges"] if node_idx in edge]
        for edge in edges_with_idx:
            edge_idx = edge[1] if node_idx == edge[0] else edge[0]
            if csp["nodes"][edge_idx] not in assigned_nodes:
                num_neighbors[node] += 1
    sorted_neighbors = dict(
        sorted(num_neighbors.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    selected = next(iter(sorted_neighbors.keys()))
    if trace:
        print(
            f"selecting variable, number of unassigned neighbors per node {sorted_neighbors}"
        )
        print(f"selected {selected}")
    return selected


def is_consistent(
    csp: dict, var: str, val: str, assignment: list[tuple[str, str]]
) -> bool:
    """
    Checks that a given assignment for a variable in a CSP is consistent with
    its assigned neighbors.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]
            "domains": dict[str, list[str]]}.
        var (str): The proposed color for assignment.
        assignment (list[tuple[str, str]]): The list of assignments made so far in the
            CSP.

    Returns:
        bool: True if the proposed variable assignment does not violate any constraints
            with neighbors, otherwise returns false.
    """
    node_idx = csp["nodes"].index(var)
    edges_with_idx = [edge for edge in csp["edges"] if node_idx in edge]
    for edge in edges_with_idx:
        neighbor_idx = edge[1] if node_idx == edge[0] else edge[0]
        neighbor = csp["nodes"][neighbor_idx]
        neighbor_assignment = next((t for t in assignment if t[0] == neighbor), None)
        if not neighbor_assignment:
            continue
        neighbor_color = neighbor_assignment[1]
        if val == neighbor_color:
            return False
    return True


def backtrack(
    csp: dict, assignment: list[tuple[str, str]], trace: bool
) -> list[tuple[str, str]] | None:
    """
    Performs backtracking in the DFS as part of solving a CSP. It performs a recursive
    DFS search, assigning colors to nodes, and backtracks if any assignment violates
    constraints. It prunes the search tree prior to recursion, as well as restores the
    prunings when an assignment doesn't work.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]
            "domains": dict[str, list[str]]}.
        assignment (list[tuple[str, str]]): The list of assignments made so far in the
            CSP.
        trace (bool): A debugging flag for whether to print verbose outputs.

    Returns:
        list[tuple[str, str]] | None: Returns the pruned CSP search tree, or None if an
            assignment results in failure.
    """
    if trace:
        print(f"domains: {csp['domains']}")
    if len(assignment) == len(csp["nodes"]):
        return assignment
    var = select_unassigned_variable(csp, assignment, trace)
    values = order_domain_values(csp, var, assignment)
    for val in values:
        if trace:
            print(
                f"evaluating variable {var} with value {val} from ordered values {values}"
            )
        if is_consistent(csp, var, val, assignment):
            pruned_csp = deepcopy(csp)
            assignment.append((var, val))
            if trace:
                print("assignment ", assignment)
            pruned_csp["domains"][var] = [val]
            inferences = inference(pruned_csp, var, assignment, trace)
            if inferences is not None:
                result = backtrack(pruned_csp, assignment, trace)
                if result is not None:
                    return result
            assignment.remove((var, val))
    return None


def augment_csp(csp: dict, color_list: list[str]) -> dict:
    """
    Augments the original CSP data structure with the domain for each node (i.e. the
    color list). This allows us to implement forward checking in solving the problem.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]}.
        color_list (list[str]): The list of colors to be used in coloring the map, and
            also for the domain for each variable.

    Returns:
        dict: The augmented CSP, with the domains for each variable added. It now has
            the form {"coordinates": list[tuple], "edges": list[tuple], "nodes":
            list[str] "domains": dict[str, list[str]]}
    """
    augmented_csp = deepcopy(csp)
    augmented_csp["domains"] = {n: list(color_list) for n in augmented_csp["nodes"]}
    return augmented_csp


def backtracking_search(
    csp: dict[str, list[tuple] | list[str]], color_list: list[str], trace: bool
) -> list[tuple[str, str]] | None:
    """
    Is a wrapper around the CSP backtracking algorithm that performs some additional
    processing. It augments the original CSP problem given from color_map, by adding the
    domain for each variable. It also reorders the assignments to match the variable
    order in the original CSP before passing the resulting assignment back to color_map.

    Args:
        csp (dict): The constraint satisfaction problem to be solved. It has the form
            {"coordinates": list[tuple], "edges": list[tuple], "nodes": list[str]}.
        color_list (list[str]): The list of allowed colors from which to create the
            domain for each variable.
        trace (bool): A debugging flag for whether to print verbose outputs.

    Returns:
        list[tuple[str, str]] | None: Returns the assignments in the order given in the
            original CSP from color_map, or None if the algorithm could not find
            assignments that satisfy the constraints.
    """
    augmented_csp = augment_csp(csp, color_list)
    assignment = backtrack(augmented_csp, [], trace)
    if assignment:
        tmp = dict(assignment)
        assignment = [(node, tmp[node]) for node in csp["nodes"]]
    return assignment


def color_map(
    planar_map: dict, color_list: list[str], trace=False
) -> list[tuple[str, str]] | None:
    """
    Colors an arbitrary map modeled as a graph, using the Russel and Norvig
    backtracking_search algorithm, implemented with backtracking, forward checking,
    degree heuristic for variable selection, and least contraining value for value
    selection.

    Args:
        planar_map (dict): The graph dictionary that is solved using the CSP techniques.
        color_list (list[str]): The list of allowed colors from which to create the
            domain for each variable.
        trace (bool): A debugging flag for whether to print verbose outputs.

    Returns:
        list[tuple[str, str]] | None: The assignments for each graph node, or None if
            the constraints could not be met.
    """
    return backtracking_search(planar_map, color_list, trace)


if __name__ == "__main__":
    connecticut = {
        "coordinates": [
            (46, 52),
            (217, 146),
            (65, 142),
            (147, 85),
            (162, 140),
            (104, 77),
            (197, 94),
            (123, 142),
        ],
        "edges": [
            (0, 2),
            (0, 5),
            (2, 5),
            (2, 7),
            (5, 7),
            (5, 3),
            (7, 3),
            (7, 4),
            (7, 6),
            (3, 6),
            (4, 6),
            (4, 1),
            (6, 1),
        ],
        "nodes": [
            "Fairfield",
            "Windham",
            "Litchfield",
            "Middlesex",
            "Tolland",
            "New Haven",
            "New London",
            "Hartford",
        ],
    }
    print(connecticut)

    draw_map(
        "connecticut", connecticut, (5, 4), [(n, "red") for n in connecticut["nodes"]]
    )

    connecticut_colors = color_map(
        connecticut, ["red", "blue", "green", "yellow"], trace=True
    )

    edges = connecticut["edges"]
    nodes = connecticut["nodes"]
    colors = connecticut_colors
    COLOR = 1

    for start, end in edges:
        try:
            assert colors[start][COLOR] != colors[end][COLOR]
        except AssertionError:
            print(
                f"{nodes[start]} and {nodes[end]} are adjacent but have the same color."
            )

    draw_map("connecticut", connecticut, (5, 4), connecticut_colors)

    connecticut_colors = color_map(connecticut, ["red", "blue", "green"], trace=False)
    if connecticut_colors:
        draw_map("connecticut", connecticut, (5, 4), connecticut_colors)

    europe = {
        "coordinates": [
            (47, 28),
            (108, 14),
            (18, 147),
            (48, 83),
            (98, 59),
            (148, 57),
            (160, 13),
            (63, 34),
            (84, 80),
            (82, 69),
            (136, 95),
            (194, 32),
            (94, 97),
            (143, 149),
            (140, 111),
            (110, 67),
            (127, 40),
            (118, 47),
            (111, 54),
            (189, 39),
            (202, 33),
            (82, 74),
            (110, 162),
            (137, 102),
            (93, 55),
            (125, 32),
            (128, 37),
            (122, 42),
            (116, 53),
            (124, 65),
            (146, 87),
            (138, 54),
            (137, 41),
            (64, 90),
            (130, 22),
            (168, 29),
            (78, 55),
            (116, 144),
            (122, 57),
            (158, 65),
            (122, 78),
            (112, 60),
            (127, 48),
            (191, 118),
            (100, 78),
            (102, 35),
        ],
        "edges": [
            (2, 3),
            (2, 33),
            (3, 33),
            (33, 36),
            (33, 21),
            (33, 8),
            (33, 22),
            (0, 7),
            (7, 36),
            (7, 45),
            (36, 21),
            (36, 9),
            (36, 44),
            (36, 24),
            (36, 45),
            (21, 8),
            (21, 9),
            (21, 44),
            (8, 44),
            (9, 44),
            (44, 12),
            (44, 37),
            (44, 40),
            (44, 15),
            (44, 41),
            (44, 24),
            (12, 22),
            (12, 37),
            (12, 40),
            (22, 37),
            (22, 13),
            (22, 43),
            (37, 13),
            (37, 14),
            (37, 23),
            (37, 40),
            (13, 14),
            (13, 43),
            (14, 23),
            (14, 43),
            (23, 10),
            (23, 30),
            (23, 43),
            (10, 40),
            (10, 30),
            (40, 15),
            (40, 29),
            (40, 30),
            (40, 39),
            (15, 41),
            (15, 29),
            (41, 4),
            (41, 24),
            (41, 45),
            (41, 18),
            (41, 38),
            (41, 29),
            (4, 24),
            (24, 45),
            (45, 1),
            (45, 34),
            (45, 25),
            (45, 27),
            (45, 17),
            (45, 28),
            (45, 18),
            (34, 25),
            (34, 26),
            (34, 32),
            (34, 6),
            (34, 35),
            (25, 26),
            (25, 16),
            (25, 27),
            (26, 16),
            (26, 42),
            (26, 32),
            (16, 27),
            (16, 42),
            (27, 17),
            (27, 42),
            (17, 42),
            (17, 28),
            (42, 28),
            (42, 38),
            (42, 31),
            (42, 32),
            (28, 18),
            (28, 38),
            (18, 38),
            (38, 29),
            (38, 39),
            (38, 31),
            (29, 39),
            (30, 39),
            (30, 43),
            (39, 5),
            (39, 31),
            (39, 43),
            (5, 31),
            (31, 32),
            (32, 35),
            (6, 35),
            (35, 19),
            (35, 11),
            (35, 20),
            (19, 11),
            (19, 20),
            (19, 43),
            (11, 20),
            (20, 43),
        ],
        "nodes": [
            "Portugal",
            "Malta",
            "Iceland",
            "Ireland",
            "Liechtenstein",
            "Moldova",
            "Cyprus",
            "Spain",
            "Netherlands",
            "Luxembourg",
            "Lithuania",
            "Armenia",
            "Denmark",
            "Finland",
            "Estonia",
            "Czech Republic",
            "Kosovo",
            "Bosnia Herzegovina",
            "Slovenia",
            "Georgia",
            "Azerbaijan",
            "Belgium",
            "Norway",
            "Latvia",
            "Switzerland",
            "Albania",
            "Macedonia",
            "Montenegro",
            "Croatia",
            "Slovakia",
            "Belarus",
            "Romania",
            "Bulgaria",
            "United Kingdom",
            "Greece",
            "Turkey",
            "France",
            "Sweden",
            "Hungary",
            "Ukraine",
            "Poland",
            "Austria",
            "Serbia",
            "Russia",
            "Germany",
            "Italy",
        ],
    }
    print(europe)

    europe_colors = color_map(europe, ["red", "blue", "green", "yellow"], trace=True)

    edges = europe["edges"]
    nodes = europe["nodes"]
    colors = europe_colors
    COLOR = 1

    for start, end in edges:
        try:
            assert colors[start][COLOR] != colors[end][COLOR]
        except AssertionError:
            print(
                f"{nodes[start]} and {nodes[end]} are adjacent but have the same color."
            )

    draw_map("europe", europe, (10, 8), europe_colors)

    europe_colors = color_map(europe, ["red", "blue", "green"], trace=False)
    if europe_colors:
        draw_map("europe", europe, (10, 8), europe_colors)
