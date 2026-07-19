import tokenize
from io import StringIO


def is_variable(exp):
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    return isinstance(exp, str) and not is_variable(exp)


def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def occurs_check(exp1, exp2):
    return exp1 in flatten(exp2)


def inconsistent_assignment(exp1, exp2, frame):
    if not exp1 in frame:
        return False
    return not frame[exp1] == exp2


def unification(exp1, exp2, frame=None):
    if frame == None:
        frame = {}
    if is_constant(exp1) and is_constant(exp2) or len(exp1) == 0 and len(exp2) == 0:
        if exp1 == exp2:
            return frame
        else:
            return False
    if is_variable(exp1):
        if occurs_check(exp1, exp2) or inconsistent_assignment(exp1, exp2, frame):
            return False
        else:
            frame[exp1] = exp2
            return frame
    if is_variable(exp2):
        if occurs_check(exp2, exp1) or inconsistent_assignment(exp2, exp1, frame):
            return False
        else:
            frame[exp2] = exp1
            return frame
    head1 = exp1[0]
    head2 = exp2[0]
    frame = unification(head1, head2, frame)
    if frame == False:
        return False
    return unification(exp1[1:], exp2[1:], frame)


def atom(next, token):
    if token[1] == "(":
        out = []
        token = next()
        while token[1] != ")":
            out.append(atom(next, token))
            token = next()
            if token[1] == " ":
                token = next()
        return out
    elif token[1] == "?":
        token = next()
        return "?" + token[1]
    else:
        return token[1]


def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())


def unify(exp1, exp2):
    return unification(parse(exp1), parse(exp2))


def apply_result(result: dict, exp: list) -> None:
    """
    Is a helper function that recursively makes in-place variable substitutions in
    logic expressions.

    Args:
        result (dict): The resultant variable to constant mapping from recursive
            unification.
        exp (list): The logic expression. Modified in-place.

    Returns:
        None: Does not return anything and modifies in-place.
    """
    for i, e in enumerate(exp):
        if isinstance(e, list):
            apply_result(result, e)
        elif e in result:
            exp[i] = result[e]


def repeat_locations(frame: dict) -> bool:
    """
    Is a helper function used to guard against and prune the "?from" and "?to" locations
    being identical in the recursive DFS search for all possible unifications for an
    action's precondition.

    Args:
        frame (dict): The mapping of variable bindings.

    Returns:
        bool: Return True if there's an instance of "?to" and "?from" being the same
            location, otherwise returns False.
    """
    if "?from" in frame and "?to" in frame:
        if frame["?from"] == frame["?to"]:
            return True
    return False


def action_condition_search(
    current_facts: list[str], conditions: list[str], frame: dict | None = None
) -> list[dict]:
    """
    Does a recursive DFS and finds all the ways the facts can unify with the
    pre-conditions for an action, ensuring that ?to and ?from aren't counted as possible
    permutations of unifications.

    Args:
        current_facts (list[str]): The facts associated with the current state.
        conditions (list[str]): The conditions to unify with.
        frame (list[str]): The compiled unifications from the search.

    Returns:
        list[dict]: The list of unification dicts.
    """
    if frame is None:
        frame = {}
    if not conditions:
        return [frame]
    next_condition = conditions[0]
    rest_conditions = conditions[1:]
    parsed_condition = parse(next_condition)
    apply_result(frame, parsed_condition)
    results = []
    for fact in current_facts:
        parsed_fact = parse(fact)
        new_frame = unification(parsed_condition, parsed_fact, frame.copy())
        if new_frame is not False and not repeat_locations(new_frame):
            results.extend(
                action_condition_search(current_facts, rest_conditions, new_frame)
            )
    return results


def parenthify(expr: list[str]) -> str:
    """
    Is a helper function used in the process of updating the state when generating
    successor actions as part of planning. It takes a list of strings and converts it to
    our parenthetical fact/action format.

    Args:
        expr (list[str]): The list of strings to convert.

    Returns:
        str: Return a string of fact components enclosed by parenthesis.
    """
    return "(" + " ".join(expr) + ")"


def convert_to_fact_or_action(perm: dict, fact_or_action: str) -> str:
    """
    Is a helper function used in the process of updating the state when generating
    successor actions as part of planning. It uses a permutation of unification bindings
    and a fact from the add or delete list for a particular action to convert the
    permutation into a state fact that can either be added or deleted from the overall
    state.

    Args:
        perm (dict): The permutation of unifications from the recursive search.
        fact_or_action (str): The fact from the add/delete list for a particular action.

    Returns:
        str: Return a string enclosed by parenthesis, representing the fact to be added
            or deleted from the state.
    """
    parsed_fact_or_action = parse(fact_or_action)
    apply_result(perm, parsed_fact_or_action)
    fact_or_action = parenthify(parsed_fact_or_action)
    return fact_or_action


def update_state(perm: dict, successor_state: list[str], action: dict) -> list[str]:
    """
    Uses a permutation of unification bindings and the add/delete lists from an action
    to update the facts of a successor state.

    Args:
        perm (dict): The permutation of unifications from the recursive search.
        successor_state (list[str]): The updated state, based on the add/delete lists
            for the application action.
        action (dict): The action to use when updating the state.

    Returns:
        list[str]: Returns the updated facts for the successor state.
    """
    for del_fact in action["delete"]:
        factified = convert_to_fact_or_action(perm, del_fact)
        if factified in successor_state:
            successor_state.remove(factified)
    for add_fact in action["add"]:
        factified = convert_to_fact_or_action(perm, add_fact)
        if factified not in successor_state:
            successor_state.append(factified)
    return successor_state


def action_successors(
    current_facts: list[str], actions: dict
) -> list[tuple[list[str], str]]:
    """
    Finds the permissible actions given the current state facts and creates a successor
    for each permutation of state fact substitutions that make each action permissible.

    Args:
        current_facts (list[str]): The permutation of unifications from the recursive
            search.
        actions (dict): The action to use when updating the state.

    Returns:
        list[tuple[list[str], str]]: Returns the updated facts for the successor state
            as well as the action in a tuple.
    """
    successors = []
    explored = set()
    for action in actions.values():
        permissible_action_permutations = action_condition_search(
            current_facts, action["conditions"]
        )
        for perm in permissible_action_permutations:
            successor_state = current_facts.copy()
            successor = sorted(update_state(perm, successor_state, action))
            converted_action = convert_to_fact_or_action(perm, action["action"])
            if tuple(successor) not in explored:
                explored.add(tuple(successor))
                successors.append((successor, converted_action))
    return successors


def check_for_goal(current_facts: list[str], goal: list[str]) -> bool:
    """
    Check to see if the current state matches the goal.

    Args:
        current_facts (list[str]): The permutation of unifications from the recursive
            search.
        goal (list[str]): The list of facts that comprise the goal state.

    Returns:
        bool: Returns True if the goal matches the current state, otherwise returns
            False.
    """
    state_set = set(current_facts)
    goal_set = set(goal)
    return goal_set.issubset(state_set)


def interleave_states_and_actions(states: list[list[str]], actions: list[str]) -> list:
    """
    Is a helper function that interleaves the ordered actions, and the resulting state
    from each action, for use in the forward planner debug mode, to see how the state
    unfolds over time.

    Args:
        states (list[list[str]]): The list of states in the path toward the matched
            goal.
        actions (list[str]): The list of actions taken, in order.

    Returns:
        list: The interleaved states and actions, i.e.  [s0, a1, s1, a2, s2, a3, s3,
            ...].
    """
    result = []
    for i, state in enumerate(states):
        result.append(state)
        if i < len(actions):
            result.append(actions[i])
    return result


def forward_planner(
    start_state: list[str], goal: list[str], actions: dict, debug: bool = False
) -> list:
    """
    Implements forward planning using DFS. There's an outer search that checks if the
    goal has been met and generates child states based on actions, and an inner search
    in action_successors, that does a search on all the ways the state facts can unify
    with action conditions. If the goal is met, the ordered actions are returned if
    debug is False, and the interleaved actions and resulting states are returned if
    debug is True.

    Args:
        start_state (list[str]): The initial state facts for the planner.
        goal (list[str]): The list of facts that constitute the goal state.
        actions (dict): The action schema of possible actions, preconditions, and
            add/delete lists.
        debug (bool): Will add the state after each action to the returned list if True.

    Returns:
        list: The list of actions and state after each action if debug=True, or just the
            list of ordered actions of debug=False.
    """
    start_facts = sorted(start_state)
    explored = set()
    frontier = [(start_facts, [], [start_facts])]
    while frontier:
        facts, plan, accumulated_facts = frontier.pop()
        if check_for_goal(facts, goal):
            return (
                interleave_states_and_actions(accumulated_facts, plan)
                if debug
                else plan
            )
        children = action_successors(facts, actions)
        for child_facts, action in children:
            child_facts = sorted(child_facts)
            already_in_frontier = any(
                tuple(frontier_facts) == tuple(child_facts)
                for frontier_facts, _, _ in frontier
            )
            if tuple(child_facts) not in explored and not already_in_frontier:
                frontier.append(
                    (child_facts, plan + [action], accumulated_facts + [child_facts])
                )
        explored.add(tuple(facts))
    return []


if __name__ == "__main__":
    start_state = [
        "(item Saw)",
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(place Bank)",
        "(agent Me)",
        "(at Me Home)",
        "(at Saw Store)",
        "(at Drill Store)",
    ]

    goal = [
        "(item Saw)",
        "(item Drill)",
        "(place Home)",
        "(place Store)",
        "(place Bank)",
        "(agent Me)",
        "(at Me Home)",
        "(at Drill Me)",
        "(at Saw Store)",
    ]

    actions = {
        "drive": {
            "action": "(drive ?agent ?from ?to)",
            "conditions": [
                "(agent ?agent)",
                "(place ?from)",
                "(place ?to)",
                "(at ?agent ?from)",
            ],
            "add": ["(at ?agent ?to)"],
            "delete": ["(at ?agent ?from)"],
        },
        "buy": {
            "action": "(buy ?purchaser ?seller ?item)",
            "conditions": [
                "(item ?item)",
                "(place ?seller)",
                "(agent ?purchaser)",
                "(at ?item ?seller)",
                "(at ?purchaser ?seller)",
            ],
            "add": ["(at ?item ?purchaser)"],
            "delete": ["(at ?item ?seller)"],
        },
    }
    print(
        forward_planner(
            start_state=start_state, goal=goal, actions=actions, debug=False
        )
    )

    print(
        forward_planner(start_state=start_state, goal=goal, actions=actions, debug=True)
    )
