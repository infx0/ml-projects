from knn import processing, distance, knn, eval_knn_mse, cross_validate


def test_processing():
    nearest = [(0, [0, 0, 0, 0, 0])]
    assert processing(nearest) == 0.0
    nearest = [(0, [0, 0, 0, 0, 0]), (1, [1, 1, 1, 1, 1])]
    assert processing(nearest) == 0.5
    nearest = [(0, [0, 0, 0, 0, 0]), (1, [1, 1, 1, 1, 1]), (2, [2, 2, 2, 2, 2])]
    assert processing(nearest) == 1.0


def test_distance():
    example = [0, 0, 0, 0, 0]
    query = [0, 0, 0, 0, 0]
    assert distance(example, query) == 0.0
    query = [1, 1, 1, 1, 1]
    assert distance(example, query) == 5.0
    query = [2, 2, 2, 2, 2]
    assert distance(example, query) == 20.0


def test_knn():
    test_data = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 5], [2, 2, 2, 2, 10]]
    query = [0, 0, 0, 0]
    assert knn(test_data, query, k=1) == 0.0
    assert knn(test_data, query, k=2) == 2.5
    assert knn(test_data, query, k=3) == 5.0


def test_eval_knn_mse():
    train_data = [[0, 0, 0, 0, 0.0], [1, 1, 1, 1, 5.0], [2, 2, 2, 2, 10.0]]
    assert eval_knn_mse(train_data, train_data, k=1) == 0.0
    assert eval_knn_mse(train_data, train_data, k=2) == 6.25
    assert abs(16.6666667 - eval_knn_mse(train_data, train_data, k=3)) < 0.01
    test_data = [[0, 0, 0, 0, 0]]
    assert eval_knn_mse(train_data, test_data, k=1) == 0.0
    assert eval_knn_mse(train_data, test_data, k=2) == 6.25
    assert eval_knn_mse(train_data, test_data, k=3) == 25.0
    train_data = [[0, 0, 0, 0, 1.0], [1, 1, 1, 1, 2.0], [2, 2, 2, 2, 3.0]]
    assert eval_knn_mse(train_data, test_data, k=1, null=True) == 4.0


def test_cross_validate():
    mock_data = [[0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0.0], [0, 0, 0, 0, 0.0]]
    train_err, test_err = cross_validate(
        observations=mock_data, k=1, num_folds=3, debug=True
    )
    assert train_err == [0.0, 0.0, 0.0]
    assert test_err == [0.0, 0.0, 0.0]
    mock_data = [[0, 0, 0, 0, 0.0], [1, 1, 1, 1, 5.0], [2, 2, 2, 2, 10.0]]
    train_err, test_err = cross_validate(
        observations=mock_data, k=1, num_folds=3, debug=True
    )
    assert train_err == [0.0, 0.0, 0.0]
    assert test_err == [25.0, 25.0, 25.0]
