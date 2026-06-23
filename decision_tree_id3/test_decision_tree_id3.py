from decision_tree_id3 import (
    is_homogenous,
    majority_label,
    pick_best_attribute,
    train,
    classify_row,
    classify,
    evaluate,
    cross_validate,
    pretty_print_tree,
    clean_data,
)


def test_is_homogenous():
    data = [{"eat": "edible"}]
    assert is_homogenous(data)
    data = [{"eat": "edible"}, {"eat": "poisonous"}]
    assert not is_homogenous(data)
    data = []
    assert not is_homogenous(data)


def test_majority_label():
    data = [{"eat": "edible", "key": "val"}]
    assert majority_label(data) == "edible"
    data = [
        {"eat": "edible", "key": "val1"},
        {"eat": "poisonous", "key": "val2"},
        {"eat": "poisonous", "key": "val3"},
    ]
    assert majority_label(data) == "poisonous"
    data = [{"eat": "edible", "key": "val1"}, {"eat": "poisonous", "key": "val2"}]
    assert majority_label(data) == "edible"


def test_pick_best_attribute():
    data = []
    attr = {}
    assert pick_best_attribute(data, attr) == (None, 99999)
    data = [{"eat": "edible", "test_key": "test_val"}]
    attr = {"test_key"}
    assert pick_best_attribute(data, attr) == ("test_key", 0.0)
    data = [
        {"eat": "edible", "test_key1": "test_val2", "test_key2": "test_val3"},
        {"eat": "poisonous", "test_key1": "test_val2", "test_key2": "test_val3"},
    ]
    attr = {"test_key1", "test_key2"}
    assert pick_best_attribute(data, attr) == ("test_key1", 1.0)


def test_train():
    training_data = []
    assert train(training_data, attributes={}) is None
    training_data = [{"color": "blue", "eat": "p"}, {"color": "white", "eat": "e"}]
    tree = train(training_data, attributes={"color"})
    assert train(training_data, attributes={"color"}) == {
        "color": {"white": "e", "blue": "p"}
    }
    training_data = [
        {"color": "blue", "eat": "p"},
        {"color": "white", "eat": "e"},
        {"color": "green", "eat": "p"},
    ]
    tree = train(training_data, attributes={"color"})
    assert tree == {"color": {"white": "e", "green": "p", "blue": "p"}}


def test_classify_row():
    tree = {
        "odor": {
            "a": "e",
            "m": "p",
            "l": "e",
            "p": "p",
            "n": {
                "spore-print-color": {
                    "r": "p",
                    "n": "e",
                    "w": {
                        "cap-color": {
                            "g": "e",
                            "p": "e",
                            "y": "p",
                            "n": "e",
                            "c": "e",
                            "w": "p",
                        }
                    },
                    "k": "e",
                }
            },
            "c": "p",
            "f": "p",
        }
    }
    row = {
        "eat": "e",
        "cap-shape": "x",
        "cap-surface": "f",
        "cap-color": "e",
        "bruises": "t",
        "odor": "n",
        "gill-attachment": "f",
        "gill-spacing": "c",
        "gill-size": "b",
        "gill-color": "p",
        "stalk-shape": "t",
        "stalk-root": "b",
        "stalk-surface-above-ring": "s",
        "stalk-surface-below-ring": "s",
        "stalk-color-above-ring": "g",
        "stalk-color-below-ring": "w",
        "veil-type": "p",
        "veil-color": "w",
        "ring-number": "o",
        "ring-type": "p",
        "spore-print-color": "k",
        "population": "y",
        "habitat": "d",
    }
    pred = classify_row(tree, row)
    assert pred == "e"
    row = {
        "eat": "e",
        "cap-shape": "k",
        "cap-surface": "f",
        "cap-color": "c",
        "bruises": "f",
        "odor": "n",
        "gill-attachment": "f",
        "gill-spacing": "w",
        "gill-size": "n",
        "gill-color": "w",
        "stalk-shape": "e",
        "stalk-root": "b",
        "stalk-surface-above-ring": "s",
        "stalk-surface-below-ring": "f",
        "stalk-color-above-ring": "w",
        "stalk-color-below-ring": "n",
        "veil-type": "p",
        "veil-color": "w",
        "ring-number": "o",
        "ring-type": "e",
        "spore-print-color": "w",
        "population": "v",
        "habitat": "l",
    }
    pred = classify_row(tree, row)
    assert pred == "e"
    row = {
        "eat": "p",
        "cap-shape": "x",
        "cap-surface": "s",
        "cap-color": "p",
        "bruises": "f",
        "odor": "c",
        "gill-attachment": "f",
        "gill-spacing": "c",
        "gill-size": "n",
        "gill-color": "n",
        "stalk-shape": "e",
        "stalk-root": "b",
        "stalk-surface-above-ring": "s",
        "stalk-surface-below-ring": "s",
        "stalk-color-above-ring": "w",
        "stalk-color-below-ring": "w",
        "veil-type": "p",
        "veil-color": "w",
        "ring-number": "o",
        "ring-type": "p",
        "spore-print-color": "n",
        "population": "v",
        "habitat": "d",
    }
    pred = classify_row(tree, row)
    assert pred == "p"


def test_classify():
    tree = {
        "odor": {
            "a": "e",
            "m": "p",
            "l": "e",
            "p": "p",
            "n": {
                "spore-print-color": {
                    "r": "p",
                    "n": "e",
                    "w": {
                        "cap-color": {
                            "g": "e",
                            "p": "e",
                            "y": "p",
                            "n": "e",
                            "c": "e",
                            "w": "p",
                        }
                    },
                    "k": "e",
                }
            },
            "c": "p",
            "f": "p",
        }
    }
    observations = [
        {
            "eat": "e",
            "cap-shape": "x",
            "cap-surface": "f",
            "cap-color": "e",
            "bruises": "t",
            "odor": "n",
            "gill-attachment": "f",
            "gill-spacing": "c",
            "gill-size": "b",
            "gill-color": "p",
            "stalk-shape": "t",
            "stalk-root": "b",
            "stalk-surface-above-ring": "s",
            "stalk-surface-below-ring": "s",
            "stalk-color-above-ring": "g",
            "stalk-color-below-ring": "w",
            "veil-type": "p",
            "veil-color": "w",
            "ring-number": "o",
            "ring-type": "p",
            "spore-print-color": "k",
            "population": "y",
            "habitat": "d",
        }
    ]
    classify(tree, observations)
    assert observations[0]["pred"] == "e"
    observations = [
        {
            "eat": "e",
            "cap-shape": "x",
            "cap-surface": "f",
            "cap-color": "e",
            "bruises": "t",
            "odor": "n",
            "gill-attachment": "f",
            "gill-spacing": "c",
            "gill-size": "b",
            "gill-color": "p",
            "stalk-shape": "t",
            "stalk-root": "b",
            "stalk-surface-above-ring": "s",
            "stalk-surface-below-ring": "s",
            "stalk-color-above-ring": "g",
            "stalk-color-below-ring": "w",
            "veil-type": "p",
            "veil-color": "w",
            "ring-number": "o",
            "ring-type": "p",
            "spore-print-color": "k",
            "population": "y",
            "habitat": "d",
        },
        {
            "eat": "e",
            "cap-shape": "k",
            "cap-surface": "f",
            "cap-color": "c",
            "bruises": "f",
            "odor": "n",
            "gill-attachment": "f",
            "gill-spacing": "w",
            "gill-size": "n",
            "gill-color": "w",
            "stalk-shape": "e",
            "stalk-root": "b",
            "stalk-surface-above-ring": "s",
            "stalk-surface-below-ring": "f",
            "stalk-color-above-ring": "w",
            "stalk-color-below-ring": "n",
            "veil-type": "p",
            "veil-color": "w",
            "ring-number": "o",
            "ring-type": "e",
            "spore-print-color": "w",
            "population": "v",
            "habitat": "l",
        },
    ]
    classify(tree, observations)
    assert len(observations) == 2
    assert observations[1]["pred"] == "e"
    observations = [
        {
            "eat": "p",
            "cap-shape": "x",
            "cap-surface": "s",
            "cap-color": "p",
            "bruises": "f",
            "odor": "c",
            "gill-attachment": "f",
            "gill-spacing": "c",
            "gill-size": "n",
            "gill-color": "n",
            "stalk-shape": "e",
            "stalk-root": "b",
            "stalk-surface-above-ring": "s",
            "stalk-surface-below-ring": "s",
            "stalk-color-above-ring": "w",
            "stalk-color-below-ring": "w",
            "veil-type": "p",
            "veil-color": "w",
            "ring-number": "o",
            "ring-type": "p",
            "spore-print-color": "n",
            "population": "v",
            "habitat": "d",
        }
    ]
    classify(tree, observations)
    assert observations[0]["pred"] == "p"


def test_evaluate():
    observations = [{"eat": "p", "pred": "p"}]
    assert evaluate(observations) == 0.0
    observations = [{"eat": "e", "pred": "p"}]
    assert evaluate(observations) == 1.0
    observations = [{"eat": "e", "pred": "p"}, {"eat": "p", "pred": "p"}]
    assert evaluate(observations) == 0.0


def test_cross_validate():
    data = [
        {"color": "red", "eat": "e"},
        {"color": "red", "eat": "e"},
        {"color": "green", "eat": "p"},
        {"color": "green", "eat": "p"},
    ]
    train_err, test_err = cross_validate(data, {"color"}, 2)
    assert train_err == [0.0, 0.0]
    assert test_err == [0.0, 0.0]
    data = [
        {"color": "green", "eat": "p"},
        {"color": "yellow", "eat": "p"},
        {"color": "orange", "eat": "e"},
        {"color": "blue", "eat": "e"},
    ]
    train_err, test_err = cross_validate(data, {"color"}, 2)
    assert train_err == [0.0, 0.0]
    assert test_err == [0.5, 0.5]


def test_pretty_print_tree():
    tree = {"color": "blue"}
    assert pretty_print_tree(tree)
    tree = {"color": "blue", "eat": "p"}
    assert pretty_print_tree(tree)
    tree = {
        "odor": {
            "m": "p",
            "c": "p",
            "l": "e",
            "p": "p",
            "n": {
                "spore-print-color": {
                    "r": "p",
                    "n": "e",
                    "w": {
                        "cap-color": {
                            "g": "e",
                            "p": "e",
                            "y": "p",
                            "n": "e",
                            "c": "e",
                            "w": "p",
                        }
                    },
                    "k": "e",
                }
            },
            "a": "e",
            "f": "p",
        }
    }
    assert pretty_print_tree(tree)


def test_clean_data():
    data = [["A", "B", "C"]]
    assert clean_data(data) == [["A", "B", "C"]]
    data = [["A", "B", "C"], ["?"]]
    assert clean_data(data) == [["A", "B", "C"]]
    data = [["?"]]
    assert clean_data(data) == []
