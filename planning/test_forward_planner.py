from planning.forward_planner import (
    apply_result,
    repeat_locations,
    action_condition_search,
    parenthify,
    convert_to_fact_or_action,
    update_state,
    action_successors,
    check_for_goal,
    interleave_states_and_actions,
)


def test_apply_result():
    result = {}
    exp = ["A"]
    apply_result(result, exp)
    assert exp == ["A"]

    result = {"?x": "A"}
    exp = ["?x", "B"]
    apply_result(result, exp)
    assert exp == ["A", "B"]

    exp = ["?x", "?x"]
    apply_result(result, exp)
    assert exp == ["A", "A"]

    result = {"?x": "A", "?y": "B"}
    exp = ["?x", ["?y", "C"]]
    apply_result(result, exp)
    assert exp == ["A", ["B", "C"]]


def test_repeat_locations():
    frame = {"?agent": "me"}
    assert not repeat_locations(frame)
    frame = {"?from": "north_pole", "?to": "south_pole"}
    assert not repeat_locations(frame)
    frame = {"?agent": "me", "?from": "north_pole", "?to": "north_pole"}
    assert repeat_locations(frame)


def test_action_condition_search():
    conditions = ["(agent ?agent)"]
    current_facts = ["(agent santa)"]
    assert action_condition_search(
        current_facts=current_facts, conditions=conditions
    ) == [{"?agent": "santa"}]
    conditions = ["(agent ?agent)", "(place ?from)"]
    current_facts = ["(agent santa)", "(place north_pole)"]
    assert action_condition_search(
        current_facts=current_facts, conditions=conditions
    ) == [{"?agent": "santa", "?from": "north_pole"}]
    conditions = ["(agent ?agent)", "(place ?from)", "(place ?to)"]
    current_facts = ["(agent santa)", "(place north_pole)", "(place south_pole)"]
    assert action_condition_search(
        current_facts=current_facts, conditions=conditions
    ) == [
        {"?agent": "santa", "?from": "north_pole", "?to": "south_pole"},
        {"?agent": "santa", "?from": "south_pole", "?to": "north_pole"},
    ]


def test_parenthify():
    assert parenthify(["agent", "me"]) == "(agent me)"
    assert parenthify(["at", "me", "los_angeles"]) == "(at me los_angeles)"
    assert (
        parenthify(["from", "los_angeles", "to", "new_york"])
        == "(from los_angeles to new_york)"
    )


def test_convert_to_fact_or_action():

    fact = "(?agent)"
    perm = {"?agent": "me"}
    assert convert_to_fact_or_action(perm, fact) == "(me)"
    fact = "(at ?agent ?to)"
    perm = {"?agent": "me", "?to": "los_angeles"}
    assert convert_to_fact_or_action(perm, fact) == "(at me los_angeles)"
    fact = "(at ?item ?place)"
    perm = {"?item": "tool", "?place": "shed"}
    assert convert_to_fact_or_action(perm, fact) == "(at tool shed)"


def test_update_state():
    test_action = {
        "action": "(drive ?agent ?from ?to)",
        "conditions": ["(agent ?agent)", "(place ?from)", "(place ?to)"],
        "add": ["(at ?agent ?to)"],
        "delete": ["(at ?agent ?from)"],
    }
    perm = {"?agent": "me", "?from": "los_angeles", "?to": "new_york"}
    successor_state = ["(agent me)", "(at me los_angeles)"]
    assert update_state(perm, successor_state, test_action) == [
        "(agent me)",
        "(at me new_york)",
    ]
    successor_state = ["(agent me)", "(at me new_york)"]
    assert update_state(perm, successor_state, test_action) == [
        "(agent me)",
        "(at me new_york)",
    ]
    perm = {"?agent": "me", "?from": "los_angeles", "?to": "new_york"}
    successor_state = []
    assert update_state(perm, successor_state, test_action) == ["(at me new_york)"]


def test_action_succesors():
    current_facts = [
        "(agent me)",
        "(place north_pole)",
        "(place south_pole)",
        "(at me north_pole)",
    ]
    action = {
        "go": {
            "action": "(move ?agent ?from ?to)",
            "conditions": [
                "(agent ?agent)",
                "(place ?from)",
                "(place ?to)",
                "(at ?agent ?from)",
            ],
            "add": ["(at ?agent ?to)"],
            "delete": ["(at ?agent ?from)"],
        }
    }
    result = action_successors(current_facts, action)
    assert len(result) == 1
    assert (
        [
            "(agent me)",
            "(at me south_pole)",
            "(place north_pole)",
            "(place south_pole)",
        ],
        "(move me north_pole south_pole)",
    ) in result
    current_facts = [
        "(agent me)",
        "(place north_pole)",
        "(place south_pole)",
        "(place equator)",
        "(at me north_pole)",
    ]
    result = action_successors(current_facts, action)
    assert len(result) == 2
    assert (
        [
            "(agent me)",
            "(at me south_pole)",
            "(place equator)",
            "(place north_pole)",
            "(place south_pole)",
        ],
        "(move me north_pole south_pole)",
    ) in result
    assert (
        [
            "(agent me)",
            "(at me equator)",
            "(place equator)",
            "(place north_pole)",
            "(place south_pole)",
        ],
        "(move me north_pole equator)",
    ) in result


def test_check_for_goal():
    current_facts = ["(agent smith)", "(at matrix)"]
    test_goal = ["(agent smith)", "(at matrix)"]
    assert check_for_goal(current_facts, test_goal)
    test_goal = ["(agent neo)", "(at matrix)"]
    assert not check_for_goal(current_facts, test_goal)
    assert not check_for_goal(current_facts, test_goal)


def test_interleave_states_and_actions():
    test_states = [["(agent me)", "(at north_pole)"]]
    test_actions = ["(move me north_pole)"]
    assert interleave_states_and_actions(test_states, test_actions) == [
        ["(agent me)", "(at north_pole)"],
        "(move me north_pole)",
    ]
    test_states = [
        ["(agent me)", "(at me north_pole)"],
        ["(agent me)", "(at me south_pole)"],
    ]
    test_actions = ["(move me south_pole)"]
    assert interleave_states_and_actions(test_states, test_actions) == [
        ["(agent me)", "(at me north_pole)"],
        "(move me south_pole)",
        ["(agent me)", "(at me south_pole)"],
    ]
    test_states = [
        ["(agent me)", "(at me north_pole)"],
        ["(agent me)", "(at me south_pole)"],
        ["(agent me)", "(at me north_pole)"],
    ]
    test_actions = ["(move me south_pole)", "(move me north_pole)"]
    assert interleave_states_and_actions(test_states, test_actions) == [
        ["(agent me)", "(at me north_pole)"],
        "(move me south_pole)",
        ["(agent me)", "(at me south_pole)"],
        "(move me north_pole)",
        ["(agent me)", "(at me north_pole)"],
    ]
