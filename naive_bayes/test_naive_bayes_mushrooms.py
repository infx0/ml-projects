
from naive_bayes_mushrooms import clean_data, probability_of, normalize, find_best, get_attr_types, get_class_totals, get_counts, ATTRIBUTES, train, classify, classify_all, evaluate, cross_validate
import math

def test_clean_data():
    data = [["A", "B", "C"]]
    assert clean_data(data) == [["A", "B", "C"]]
    data = [["A", "B", "C"], ["?"]]
    assert clean_data(data) == [["A", "B", "C"]]
    data = [["?"]]
    assert clean_data(data) == []

def test_probability_of():
    mock_nbc = {'priors':{'e': 0.6666666666666666, 'p': 0.3333333333333333}, 
            'conditionals': {'e':
                             {'cap-shape': {'b': 0.4, 'x': 0.2, 'f': 0.4}, 'cap-surface': {'s': 0.4, 'y': 0.4, 'f': 0.2}, 'cap-color': {'e': 0.5, 'y': 0.5}},
                             'p':
                             {'cap-shape': {'b': 0.25, 'x': 0.5, 'f': 0.25}, 'cap-surface': {'s': 0.25, 'y': 0.25, 'f': 0.5}, 'cap-color': {'e': 0.3333333333333333, 'y': 0.6666666666666666}}
                              }
            }
    mock_observation = {'cap-shape': 'b', 'cap-surface': 's', 'cap-color': 'y'}
    e_prob = probability_of(mock_nbc,mock_observation,label="e")
    p_prob = probability_of(mock_nbc,mock_observation,label="p")
    expected_e_prob = 0.66666666666*0.4*0.4*0.5
    expected_p_prob = 0.33333333333*0.25*0.25*0.666666666
    assert e_prob > 0, p_prob > 0
    assert math.isclose(e_prob, expected_e_prob, rel_tol=1e-6)
    assert math.isclose(p_prob, expected_p_prob, rel_tol=1e-6)

def test_normalize():
    mock_results = {"e": 0.5, "p": 0.5}
    assert list(normalize(mock_results).items()) == [("e", 0.5), ("p", 0.5)]
    mock_results = {"e": 0.1, "p": 0.4}
    assert list(normalize(mock_results).items()) == [("p", 0.8), ("e", 0.2)]
    mock_results = {"e": 0.05, "p": 0.4, "x": 0.05}
    assert list(normalize(mock_results).items()) == [("p", 0.8), ("e", 0.1), ("x", 0.1)]

def test_find_best():
    mock_results = {"e": 0.5, "p": 0.5}
    assert find_best(mock_results) == "e"
    mock_results = {"p": 0.8, "e": 0.2}
    assert find_best(mock_results) == "p"
    mock_results = {"p": 0.8, "e": 0.1, "x": 0.1}
    assert find_best(mock_results) == "p"

def test_get_attr_types():
    mock_training_data = [{'eat': 'e', 'cap-shape': 'b', 'cap-surface': 's', 'cap-color': 'y', 'bruises': 't', 'odor': 'l', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'k', 'stalk-shape': 'e', 'stalk-root': 'c', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'w', 'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 's', 'habitat': 'm'},
                      {'eat': 'p', 'cap-shape': 'x', 'cap-surface': 'f', 'cap-color': 'y', 'bruises': 'f', 'odor': 'f', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'p', 'stalk-shape': 'e', 'stalk-root': 'b', 'stalk-surface-above-ring': 'k', 'stalk-surface-below-ring': 'k', 'stalk-color-above-ring': 'b', 'stalk-color-below-ring': 'n', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'l', 'spore-print-color': 'h', 'population': 'v', 'habitat': 'g'},
                      {'eat': 'e', 'cap-shape': 'f', 'cap-surface': 'y', 'cap-color': 'e', 'bruises': 't', 'odor': 'n', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'n', 'stalk-shape': 't', 'stalk-root': 'b', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'p', 'stalk-color-below-ring': 'g', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 'v', 'habitat': 'd'}]
    attr_types_dict = get_attr_types(mock_training_data, ATTRIBUTES[1:])
    assert attr_types_dict["cap-shape"] == {"b", "x", "f"}
    assert attr_types_dict["cap-color"] == {"e", "y"}
    assert attr_types_dict["gill-spacing"] == {"c"}

def test_get_class_totals():
    mock_training_data = [{'eat': 'e'}, {'eat': 'p'},{'eat': 'x', },{'eat': 'x', },{'eat': 'x', }]
    class_totals_dict = get_class_totals(mock_training_data, "eat")
    assert class_totals_dict["e"] == 1
    assert class_totals_dict["p"] == 1
    assert class_totals_dict["x"] == 3

def test_get_counts():
    mock_training_data = [{'eat': 'e', 'cap-shape': 'b', 'cap-surface': 's', 'cap-color': 'y', 'bruises': 't', 'odor': 'l', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'k', 'stalk-shape': 'e', 'stalk-root': 'c', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'w', 'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 's', 'habitat': 'm'},
                      {'eat': 'p', 'cap-shape': 'x', 'cap-surface': 'f', 'cap-color': 'y', 'bruises': 'f', 'odor': 'f', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'p', 'stalk-shape': 'e', 'stalk-root': 'b', 'stalk-surface-above-ring': 'k', 'stalk-surface-below-ring': 'k', 'stalk-color-above-ring': 'b', 'stalk-color-below-ring': 'n', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'l', 'spore-print-color': 'h', 'population': 'v', 'habitat': 'g'},
                      {'eat': 'e', 'cap-shape': 'f', 'cap-surface': 'y', 'cap-color': 'e', 'bruises': 't', 'odor': 'n', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'n', 'stalk-shape': 't', 'stalk-root': 'b', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'p', 'stalk-color-below-ring': 'g', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 'v', 'habitat': 'd'}]
    count_dict = get_counts(mock_training_data, ATTRIBUTES[1:])
    assert count_dict["e"]["cap-shape"]["b"] == 1
    assert count_dict["e"]["bruises"]["t"] == 2
    assert count_dict["p"]["cap-color"]["y"] == 1

def test_train():
    mock_training_data = [{'eat': 'e', 'cap-shape': 'b', 'cap-surface': 's', 'cap-color': 'y', 'bruises': 't', 'odor': 'l', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'k', 'stalk-shape': 'e', 'stalk-root': 'c', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'w', 'stalk-color-below-ring': 'w', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 's', 'habitat': 'm'},
                      {'eat': 'p', 'cap-shape': 'x', 'cap-surface': 'f', 'cap-color': 'y', 'bruises': 'f', 'odor': 'f', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'p', 'stalk-shape': 'e', 'stalk-root': 'b', 'stalk-surface-above-ring': 'k', 'stalk-surface-below-ring': 'k', 'stalk-color-above-ring': 'b', 'stalk-color-below-ring': 'n', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'l', 'spore-print-color': 'h', 'population': 'v', 'habitat': 'g'},
                      {'eat': 'e', 'cap-shape': 'f', 'cap-surface': 'y', 'cap-color': 'e', 'bruises': 't', 'odor': 'n', 'gill-attachment': 'f', 'gill-spacing': 'c', 'gill-size': 'b', 'gill-color': 'n', 'stalk-shape': 't', 'stalk-root': 'b', 'stalk-surface-above-ring': 's', 'stalk-surface-below-ring': 's', 'stalk-color-above-ring': 'p', 'stalk-color-below-ring': 'g', 'veil-type': 'p', 'veil-color': 'w', 'ring-number': 'o', 'ring-type': 'p', 'spore-print-color': 'n', 'population': 'v', 'habitat': 'd'}]
    nbc = train(mock_training_data, ATTRIBUTES[1:], smoothing=False)
    assert nbc["priors"] == {'e': 0.6666666666666666, 'p': 0.3333333333333333}
    assert nbc["conditionals"]["e"]["cap-shape"] == {"b": 0.5, "x": 0.0, "f": 0.5}
    nbc = train(mock_training_data, ATTRIBUTES[1:], smoothing=True)
    assert nbc["conditionals"]["e"]["cap-shape"] == {"b": 0.4, "x": 0.2, "f": 0.4}

def test_classify():
    mock_training_data = [{'eat': 'e', 'cap-shape': 'x', 'cap-surface': 's', 'cap-color': 'y'},
                      {'eat': 'p', 'cap-shape': 'x', 'cap-surface': 'f', 'cap-color': 'e'},
                      {'eat': 'e', 'cap-shape': 'f', 'cap-surface': 'f', 'cap-color': 'y'}]
    mock_test_instance = {'cap-shape': 'f', 'cap-surface': 's', 'cap-color': 'e'}

    nbc = train(mock_training_data, ["cap-shape", "cap-surface", "cap-color"], smoothing=True)
    result = classify(nbc,mock_test_instance)
    expected_nbc = {'priors': {'e': 0.6666666666666666, 'p': 0.3333333333333333},
                    'conditionals': {'e': {'cap-shape': {'x': 0.5, 'f': 0.5},
                                        'cap-surface': {'s': 0.5, 'f': 0.5},
                                        'cap-color': {'e': 0.25, 'y': 0.75}},
                                    'p': {'cap-shape': {'x': 0.6666666666666666, 'f': 0.3333333333333333},
                                        'cap-surface': {'s': 0.3333333333333333, 'f': 0.6666666666666666},
                                        'cap-color': {'e': 0.6666666666666666, 'y': 0.3333333333333333}}}}
    expected_e_prob = 0.666666*0.5*0.5*0.25
    expected_p_prob = 0.333333*0.333333*0.333333*0.666666
    expected_norm_e_prob = expected_e_prob/(expected_e_prob+expected_p_prob)
    expected_norm_p_prob = expected_p_prob/(expected_e_prob+expected_p_prob)
    expected_result = ("e", {"e": 0.6279076, "p": 0.372092})
    assert result[0] == expected_result[0]
    assert math.isclose(result[1]["e"], expected_result[1]["e"], rel_tol=1e-4)
    assert math.isclose(result[1]["p"], expected_result[1]["p"], rel_tol=1e-4)

def test_classify_all():
    mock_training_data = [{'eat': 'e', 'cap-shape': 'x', 'cap-surface': 's', 'cap-color': 'y'},
                      {'eat': 'p', 'cap-shape': 'x', 'cap-surface': 'f', 'cap-color': 'e'},
                        {'eat': 'e', 'cap-shape': 'f', 'cap-surface': 'f', 'cap-color': 'y'}]
    mock_test_instance_labeled = [{'eat': 'e', 'cap-shape': 'f', 'cap-surface': 's', 'cap-color': 'e'}]
    mock_test_instance_unlabeled = [{'cap-shape': 'f', 'cap-surface': 's', 'cap-color': 'e'}]

    nbc = train(mock_training_data, ["cap-shape", "cap-surface", "cap-color"], smoothing=True)
    results_labeled = classify_all(nbc=nbc,observations=mock_test_instance_labeled, labeled=True)
    results_unlabeled = classify_all(nbc=nbc,observations=mock_test_instance_unlabeled, labeled=False)
    expected_nbc = {'priors': {'e': 0.6666666666666666, 'p': 0.3333333333333333},
                    'conditionals': {'e': {'cap-shape': {'x': 0.5, 'f': 0.5},
                                        'cap-surface': {'s': 0.5, 'f': 0.5},
                                        'cap-color': {'e': 0.25, 'y': 0.75}},
                                    'p': {'cap-shape': {'x': 0.6666666666666666, 'f': 0.3333333333333333},
                                        'cap-surface': {'s': 0.3333333333333333, 'f': 0.6666666666666666},
                                        'cap-color': {'e': 0.6666666666666666, 'y': 0.3333333333333333}}}}
    expected_e_prob = 0.666666*0.5*0.5*0.25
    expected_p_prob = 0.333333*0.333333*0.333333*0.666666
    expected_norm_e_prob = expected_e_prob/(expected_e_prob+expected_p_prob)
    expected_norm_p_prob = expected_p_prob/(expected_e_prob+expected_p_prob)
    expected_result = ("e", {"e": 0.6279076, "p": 0.372092})

    assert results_labeled == results_unlabeled
    assert len(results_labeled) == 1

def test_evaluate():
    observations = [{"eat": "p", "foo": "bar"}]
    inferences = [("p", {})]
    assert evaluate(observations, inferences) == 0.0
    observations = [{"eat": "p", "foo": "bar"}]
    inferences = [("e", {})]
    assert evaluate(observations, inferences) == 1.0
    observations = [{"eat": "p", "foo": "bar"}, {"eat": "p"}]
    inferences = [("p", {}), ("e", {})]
    assert evaluate(observations, inferences) == 0.5

def test_cross_validate():
    data = [{"eat": "e", "color": "red"},{"eat": "e", "color": "red"},{"eat": "p", "color": "green"},{"eat": "p", "color": "green"}]
    train_err, test_err = cross_validate(data, {"color"}, 2, smoothing=True)
    assert train_err == [0.0, 0.0]
    assert test_err == [0.0, 0.0]
    data = [{"eat": "p", "color": "green"},{"eat": "p", "color": "yellow"},{"eat": "e", "color": "green"},{"eat": "e", "color": "yellow"}]
    train_err, test_err = cross_validate(data, {"color"}, 2, smoothing=True)
    print(train_err, test_err)
    assert train_err == [0.0, 0.0]
    assert test_err == [1.0, 1.0]