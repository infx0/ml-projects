
from naive_bayes_mushrooms import clean_data, probability_of, normalize, find_best
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