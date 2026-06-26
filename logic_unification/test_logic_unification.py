from logic_unification import format_substitution_list, apply_result


def test_format_substitution_list():
    subst = {}
    assert format_substitution_list(subst) == {}
    subst = {"?x": ["A", "B"]}
    assert format_substitution_list(subst) == {"?x": "A B"}
    subst = {"?x": ["A", "B"], "?y": ["C", "D"]}
    assert format_substitution_list(subst) == {"?x": "A B", "?y": "C D"}


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
