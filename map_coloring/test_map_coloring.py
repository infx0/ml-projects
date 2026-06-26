from map_coloring import (
    order_domain_values,
    inference,
    select_unassigned_variable,
    is_consistent,
    backtrack,
    augment_csp,
    backtracking_search,
)


def test_order_domain_values():
    csp = {
        "edges": [(0, 1)],
        "nodes": ["a", "b"],
        "domains": {"a": ["c1"], "b": ["c1"]},
    }
    var = "a"
    assignment = []
    assert order_domain_values(csp, var, assignment) == ["c1"]
    csp = {
        "edges": [(0, 1), (2, 1)],
        "nodes": ["a", "b", "c"],
        "domains": {"a": ["c1", "c2"], "b": ["c1", "c2"], "c": ["c1", "c2"]},
    }
    assert order_domain_values(csp, var, assignment) == ["c1", "c2"]
    csp = {
        "edges": [(0, 1), (0, 2)],
        "nodes": ["a", "b", "c"],
        "domains": {"a": ["c1", "c2"], "b": ["c1", "c2"], "c": ["c1", "c2"]},
    }
    assignment = [("b", "c1")]
    assert order_domain_values(csp, var, assignment) == ["c1", "c2"]
    csp = {
        "edges": [(0, 1), (0, 2)],
        "nodes": ["a", "b", "c"],
        "domains": {"a": ["c1", "c2"], "b": ["c1"], "c": ["c1", "c2"]},
    }
    assignment = []
    assert order_domain_values(csp, var, assignment) == ["c2", "c1"]


def test_inference():
    csp = {
        "edges": [(0, 1)],
        "nodes": ["a", "b"],
        "domains": {"a": ["c1", "c2"], "b": ["c1", "c2"]},
    }
    assignment = [("a", "c1")]
    var = "a"
    assert inference(csp, var, assignment, trace=False) == [("b", "c1")]
    assert csp["domains"]["b"] == ["c2"]
    csp = {
        "edges": [(0, 1)],
        "nodes": ["a", "b"],
        "domains": {"a": ["c1", "c2"], "b": ["c1", "c2"]},
    }
    assignment = [("a", "c2")]
    var = "a"
    assert inference(csp, var, assignment, trace=False) == [("b", "c2")]
    assert csp["domains"]["b"] == ["c1"]


def test_select_unassigned_variable():
    # test easy case, should choose a
    csp = {"edges": [(0, 1), (0, 2), (0, 3)], "nodes": ["a", "b", "c", "d"]}
    assignment = []
    assert select_unassigned_variable(csp, assignment, trace=False) == "a"
    # test some already assigned, b should be chosen since a already assigned
    assignment = [("a", "color")]
    assert select_unassigned_variable(csp, assignment, trace=False) == "b"
    # test alphabetical sorting for tie-breakers, a should be chosen
    csp = {"edges": [(0, 1)], "nodes": ["b", "a"]}
    assignment = []
    assert select_unassigned_variable(csp, assignment, trace=False) == "a"
    # test a more complex network, b has most connections
    csp = {"edges": [(0, 1), (0, 2), (1, 2), (1, 3)], "nodes": ["c", "b", "a", "d"]}
    assignment = []
    assert select_unassigned_variable(csp, assignment, trace=False) == "b"
    # test complex network with existing assignment
    csp = {"edges": [(0, 1), (0, 2), (1, 2), (1, 3)], "nodes": ["a", "b", "c", "d"]}
    assignment = [("a", "color")]
    assert select_unassigned_variable(csp, assignment, trace=False) == "b"


def test_is_consistent():
    csp = {"edges": [(0, 1)], "nodes": ["a", "b"]}
    var = "a"
    val = "c1"
    assignment = []
    assert is_consistent(csp, var, val, assignment) == True
    var = "a"
    val = "c1"
    assignment = [("b", "c2")]
    assert is_consistent(csp, var, val, assignment) == True
    var = "a"
    val = "c1"
    assignment = [("b", "c1")]
    assert is_consistent(csp, var, val, assignment) == False


def test_backtrack():
    csp = {
        "edges": [(0, 1)],
        "nodes": ["a", "b"],
        "domains": {"a": ["c1", "c2"], "b": ["c1", "c2"]},
    }
    assignment = []
    assert backtrack(csp, assignment, trace=False) == [("a", "c1"), ("b", "c2")]
    csp = {
        "edges": [(0, 1)],
        "nodes": ["a", "b"],
        "domains": {"a": ["c1"], "b": ["c1"]},
    }
    assignment = []
    assert backtrack(csp, assignment, trace=False) == None
    csp = {
        "edges": [(0, 1), (0, 2)],
        "nodes": ["a", "b", "c"],
        "domains": {
            "a": ["c1", "c2", "c3"],
            "b": ["c1", "c2", "c3"],
            "c": ["c1", "c2", "c3"],
        },
    }
    assignment = []
    assert backtrack(csp, assignment, trace=False) == [
        ("a", "c1"),
        ("b", "c2"),
        ("c", "c2"),
    ]


def test_augment_csp():
    csp = {"edges": [], "nodes": ["a"]}
    color_list = []
    assert augment_csp(csp, color_list) == {
        "edges": [],
        "nodes": ["a"],
        "domains": {"a": []},
    }
    color_list = ["color"]
    assert augment_csp(csp, color_list) == {
        "edges": [],
        "nodes": ["a"],
        "domains": {"a": ["color"]},
    }
    csp = {"edges": [], "nodes": ["a", "b"]}
    color_list = ["color1", "color2"]
    assert augment_csp(csp, color_list) == {
        "edges": [],
        "nodes": ["a", "b"],
        "domains": {"a": ["color1", "color2"], "b": ["color1", "color2"]},
    }


def test_backtracking_search():
    csp = {"edges": [(0, 1)], "nodes": ["a", "b"]}
    color_list = ["c1", "c2"]
    assert backtracking_search(csp, color_list, trace=False) == [
        ("a", "c1"),
        ("b", "c2"),
    ]
    csp = {"edges": [(0, 1)], "nodes": ["a", "b"]}
    color_list = ["c1"]
    assert backtracking_search(csp, color_list, trace=False) == None
    csp = {"edges": [(0, 1)], "nodes": ["a", "b"]}
    color_list = ["c1", "c2", "c3"]
    assert [("a", "c1"), ("b", "c2"), ("c", "c2")]
