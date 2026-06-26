from logistic_regression import prepend_bias, sigmoid, calculate_error, derivative
import numpy as np


def test_prepend_bias():
    data = [([], 1)]
    assert prepend_bias(data) == [([1], 1)]
    data = [([1, 2, 3], 1)]
    assert prepend_bias(data) == [([1, 1, 2, 3], 1)]
    data = [([1, 2, 3], 0), ([4, 5, 6], 1)]
    assert prepend_bias(data) == [([1, 1, 2, 3], 0), ([1, 4, 5, 6], 1)]


def test_sigmoid():
    thetas = [0, 0, 0]
    vals = [0, 0, 0]
    assert sigmoid(thetas=thetas, vals=vals) == 0.5
    thetas = [1]
    vals = [1]
    assert np.isclose(sigmoid(thetas=thetas, vals=vals), 0.7310585)
    thetas = [9999]
    vals = [1]
    assert sigmoid(thetas=thetas, vals=vals) == 1.0


def test_calculate_error():
    thetas = [0.8, 1.1]
    data = [([1.0, 1.1], 0), ([1.0, 2.7], 1)]
    J = calculate_error(thetas=thetas, data=data)
    assert np.isclose(J, 1.07927)
    thetas = [0.0]
    data = [([0.0], 0), ([0.0], 0)]
    J = calculate_error(thetas=thetas, data=data)
    assert np.isclose(J, 0.69315)
    thetas = [1.0]
    data = [([1.0], 1), ([0.0], 0)]
    J = calculate_error(thetas=thetas, data=data)
    assert np.isclose(J, 0.5032)


def test_derivative():
    j = 0
    thetas = [0.8, 1.1]
    data = [([1, 1.1], 0), ([1, 2.7], 1)]
    assert np.isclose(derivative(j=j, thetas=thetas, data=data), 0.429655)
    j = 0
    thetas = [0.8, 1.1]
    data = [([0, 0.0], 0), ([1, 2.7], 1)]
    assert np.isclose(derivative(j=j, thetas=thetas, data=data), -0.0112663)
    j = 0
    thetas = [0.8, 1.1]
    data = [([0, 0.0], 0), ([0, 0.0], 1)]
    assert np.isclose(derivative(j=j, thetas=thetas, data=data), 0.0)
