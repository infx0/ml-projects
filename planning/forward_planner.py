import tokenize
from io import StringIO

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
        `apply_result` is a helper function that recursively makes in-place variable substitutions in logic expressions. **Used by**: [action_condition_search](#action_condition_search), [convert_to_fact_or_action](#convert_to_fact_or_action)

    * **result** dict: the resultant variable to constant mapping from recursive unification.
    * **exp** list: the logic expression. Modified in-place.

    **returns**: does not return anything and modifies in-place.
    """
    for i, e in enumerate(exp):
        if isinstance(e, list):
            apply_result(result, e)
        elif e in result:
            exp[i] = result[e]


def repeat_locations(frame: dict) -> bool:
    """
        `repeat_locations` is a helper function used to guard against and prune the "?from" and "?to" locations being identical in the recursive DFS search for all possible unifications for an action's precondition. **Used by**: [action_condition_search](#action_condition_search)

    * **frame** dict: the mapping of variable bindings.

    **returns** bool: return True if there's an instance of "?to" and "?from" being the same location, otherwise returns False.
    """
    if "?from" in frame and "?to" in frame:
        if frame["?from"] == frame["?to"]:
            return True
    return False


def action_condition_search(
    current_facts: list[str], conditions: list[str], frame: dict | None = None
) -> list[dict]:
    """
        `action_condition_search` does a recursive DFS and finds all the ways the facts can unify with the pre-conditions for an action, ensuring that ?to and ?from aren't counted as possible permutations of unifications. **Uses**: [action_condition_search](#action_condition_search), [apply_result](#apply_result), [repeat_locations](#repeat_locations), parse, unification **Used by**: [action_successors](#action_successors)

    * **current_facts** list[str]: the facts associated with the current state.
    * **conditions** list[str]: the conditions to unify with.
    * **frame** list[str]: the compiled unifications from the search.

    **returns** list[dict]: the list of unification dicts.
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
        `parenthify` is a helper function used in the process of updating the state when generating successor actions as part of planning. It takes a list of strings and converts it to our parenthetical fact/action format. **Used by**: [convert_to_fact_or_action](#convert_to_fact_or_action)

    * **expr**  list[str]: the list of strings to convert.

    **returns** str: return a string of fact components enclosed by parenthesis.
    """
    return "(" + " ".join(expr) + ")"


def convert_to_fact_or_action(perm: dict, fact_or_action: str) -> str:
    """
        `convert_to_fact_or_action` is a helper function used in the process of updating the state when generating successor actions as part of planning. It uses a permutation of unification bindings and a fact from the add or delete list for a particular action to convert the permutation into a state fact that can either be added or deleted from the overall state. **Uses**: [parenthify](#parenthify), [apply_result](#apply_result) **Used by**: [update_state](#update_state)

    * **perm** dict: the permutation of unifications from the recursive search.
    * **fact** str: the fact from the add/delete list for a particular action.

    **returns** str: return a string enclosed by parenthesis, representing the fact to be added or deleted from the state.
    """
    parsed_fact_or_action = parse(fact_or_action)
    apply_result(perm, parsed_fact_or_action)
    fact_or_action = parenthify(parsed_fact_or_action)
    return fact_or_action


def update_state(perm: dict, successor_state: list[str], action: dict) -> list[str]:
    """
        `update_state` uses a permutation of unification bindings and the add/delete lists from an action to update the facts of a successor state. **Uses**: [convert_to_fact_or_action](#convert_to_fact_or_action), **Used by**: [action_successors](#action_successors)

    * **perm** dict: the permutation of unifications from the recursive search.
    * **successor_state** list[str]: the updated state, based on the add/delete lists for the application action.
    * **action** dict: the action to use when updating the state.

    **returns** list[str]: returns the updated facts for the successor state.
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
        `action_successors` finds the permissible actions given the current state facts and creates a successor for each permutation of state fact substitutions that make each action permissible. **Uses**: [action_condition_search](#action_condition_search), [update_state](#update_state), **Used by**: [forward_planner](#forward_planner)

    * **current_facts** list[str]: the permutation of unifications from the recursive search.
    * **action** dict: the action to use when updating the state.

    **returns** list[tuple[list[str], str]]: returns the updated facts for the successor state as well as the action in a tuple.
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
        `check_for_goal` check to see if the current state matches the goal. **Used by**: [forward_planner](#forward_planner)

    * **current_state** list[str]: the permutation of unifications from the recursive search.
    * **goal** list[str]: the list of facts that comprise the goal state.

    **returns** bool: returns True if the goal matches the current state, otherwise returns False.
    """
    state_set = set(current_facts)
    goal_set = set(goal)
    return goal_set.issubset(state_set)


def interleave_states_and_actions(states: list[list[str]], actions: list[str]) -> list:
    """
        `interleave_states_and_actions` is a helper function that interleaves the ordered actions, and the resulting state from each action, for use in the forward planner debug mode, to see how the state unfolds over time. **Used by**: [forward_planner](#forward_planner)

    * **states** list[list[str]]: the list of states in the path toward the matched goal.
    * **actions** list[str]: the list of actions taken, in order.

    **returns** list: the interleaved states and actions, i.e.  [s0, a1, s1, a2, s2, a3, s3, ...].
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
        `forward_planner`implements forward planning using the DFS psuedocode from Module 2. There's an outer search that checks if the goal has been met and generates child states based on actions, and an inner search in action_successors, that does a search on all the ways the state facts can unify with action conditions. If the goal is met, the ordered actions are returned if debug is False, and the interleaved actions and resulting states are returned if debug is True. **Uses**: [check_for_goal](#check_for_goal), [action_sucessors](#action_successors), [interleave_states_and_actions](#interleave_states_and_actions)

    * **start_state** list[str]: the initial state facts for the planner.
    * **goal** list[str]: the list of facts that constitute the goal state.
    * **actions** dict: the action schema of possible actions, preconditions, and add/delete lists.
    * **debug** bool: will add the state after each action to the returned list if True.

    **returns** list: the list of actions and state after each action if debug=True, or just the list of ordered actions of debug=False.
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
    print(forward_planner(start_state=start_state, goal=goal, actions=actions, debug=False))

    print(forward_planner(start_state=start_state, goal=goal, actions=actions, debug=True))
