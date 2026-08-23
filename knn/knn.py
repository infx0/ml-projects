"""Implement and evaluate k-nearest-neighbor regression for numeric datasets.

The module loads and shuffles observations, creates training and validation folds,
computes Euclidean distances, averages the nearest targets, and compares candidate
values of k using mean squared error and validation plots.
"""

import random
import matplotlib.pyplot as plt


def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = [float(value) for value in line.rstrip().split(",")]
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: list, n: int) -> list[list[list]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def create_train_test(
    folds: list[list[list]], index: int
) -> tuple[list[list], list[list]]:
    training = []
    test = []
    for i, fold in enumerate(folds):
        if i == index:
            test = fold
        else:
            training = training + fold
    return training, test


def processing(nearest: list[tuple[float, list]]) -> float:
    """
    Implements the regression portion of our kNN algorithm, and returns the mean target
    value of the k-nearest observations.

    Args:
        nearest (list[tuple[float, list]]): The nearest neighbor observations from the
            kNN algorithm.

    Returns:
        float: Returns mean target value from the nearest observations.
    """
    return sum([i[1][-1] for i in nearest]) / len(nearest)


def distance(example: list[float], query: list[float]) -> float:
    """
    Calculates the distance metric used in kNN. In this version, the squared Euclidean
    distance is used to save on computation cost.

    Args:
        example (list[float]): The neighboring observations.
        query (list[float]): The reference observation.

    Returns:
        float: Returns the square of the Euclidean distance.
    """
    return sum((example[i] - query[i]) ** 2 for i in range(len(query)))


def knn(dataset: list[list[float]], query: list[float], k: int = 9) -> float:
    """
    Runs regression k-Nearest Neighbors. It calculates the distance between the query
    and all other observations, sorts by ascending distance, and using the processing
    function returns the mean value of the target variable from the k-nearest distance
    observations.

    Args:
        dataset (list[list[float]]): The input dataset for the kNN algorithm.
        query (list[float]): Observation to be used for prediction.
        k (int): The number of nearest neighbors to retrieve for inference in kNN.

    Returns:
        float: Returns average target value from the k-nearest observations.
    """
    distances = []
    for example in dataset:
        distances.append((distance(example, query), example))
    nearest = sorted(distances, key=lambda x: x[0])
    nearest = nearest[:k]
    out = processing(nearest)
    return out


def eval_knn_mse(
    train_dataset: list[list[float]],
    test_dataset: list[list[float]],
    k: int,
    null: bool = False,
) -> float:
    """
    Evaluates the MSE for kNN regression, using the input training and test sets. It
    uses the actual target value from each test observation and predicted target value
    from the training observations to sum the squared error before finally dividing by
    number of test instances to get MSE.

    Args:
        train_dataset (list[list[float]]): The training dataset.
        test_dataset (list[list[float]]): The test dataset (this could just be the
            training set again if we're looking at training error).
        k (int): The number of nearest neighbors to retrieve for inference in kNN.
        null (bool): A flag for whether to evaluate MSE against the null model, which
            sets the predicted value to the mean of the input training set.

    Returns:
        float: Returns MSE from the input training and test set.
    """
    err = 0.0
    for obs in test_dataset:
        actual_target = obs[-1]
        if null:
            pred_target = sum([i[-1] for i in train_dataset]) / len(train_dataset)
        else:
            pred_target = knn(train_dataset, obs[:-1], k)
        err += (actual_target - pred_target) ** 2
    err = err / len(test_dataset)
    return err


def cross_validate(
    observations: list[list[float]],
    k: int,
    num_folds: int,
    debug: bool = False,
    null: bool = False,
) -> tuple[list, list]:
    """
    Runs cross validation for kNN, using a dataset and specified number of folds. It
    trains, and then evaluates error on both the training set and test set. The MSE is
    returned for each fold, and optionally can be printed to console.

    Args:
        observations (list[list[float]]): The input dataset for validation that will
            create folds and then train and evaluate for each split.
        k (int): The number of nearest neighbors to retrieve for inference in kNN.
        num_folds (int): The number of splits for cross validation in the dataset.
        debug (bool): A flag to control whether the function prints to console.
        null (bool): A flag to control whether to use the null model for fold
            validation.

    Returns:
        tuple[list, list]: Returns the list of training and test errors.
    """
    folded_data = create_folds(observations, num_folds)
    train_error = []
    test_error = []
    for i in range(len(folded_data)):
        train_folds = folded_data[:i] + folded_data[i + 1 :]
        training_set = sum(train_folds, [])
        test_set = folded_data[i]
        train_err = eval_knn_mse(training_set, training_set, k, null)
        test_err = eval_knn_mse(training_set, test_set, k, null)
        if debug:
            print(f"Fold {i + 1} MSE, train: {train_err:.4f} test: {test_err:.4f}")
        train_error.append(train_err)
        test_error.append(test_err)
    return train_error, test_error


if __name__ == "__main__":
    data = parse_data("concrete_compressive_strength.csv")
    train_err, test_err = cross_validate(
        observations=data, k=9, num_folds=10, debug=True, null=False
    )
    null_train_err, null_test_err = cross_validate(
        observations=data, k=9, num_folds=10, debug=True, null=True
    )

    mean_train_errs = []
    mean_test_errs = []
    start_k = 1
    end_k = 20
    for k in range(start_k, end_k + 1):
        train_err, test_err = cross_validate(
            observations=data, k=k, num_folds=10, debug=False, null=False
        )
        mean_train = sum(train_err) / len(train_err)
        mean_test = sum(test_err) / len(test_err)
        print(f"{k=}, {mean_train=}, {mean_test=}")
        mean_train_errs.append(mean_train)
        mean_test_errs.append(mean_test)

    plt.figure()
    plt.plot(
        range(start_k, end_k + 1), mean_train_errs, label="Mean 10-Fold Training Error"
    )
    plt.plot(range(start_k, end_k + 1), mean_test_errs, label="Mean 10-Fold Test Error")
    plt.xlabel("Value of k")
    plt.ylabel("Mean Squared Error")
    plt.title("kNN Cross-Validated Error vs k")
    plt.legend()
    plt.grid()
    plt.show()

    data = parse_data("concrete_compressive_strength.csv")
    data_size = []
    train_err = []
    test_err = []
    for i in range(100, 1100, 100):
        dataset = data[:i]
        n = len(dataset)
        train, test = cross_validate(
            observations=dataset, k=3, num_folds=10, debug=False, null=False
        )
        mean_train = sum(train) / len(train)
        mean_test = sum(test) / len(test)
        print(f"{n=}, {mean_train=}, {mean_test=}")
        data_size.append(n)
        train_err.append(mean_train)
        test_err.append(mean_test)

    plt.figure()
    plt.plot(data_size, train_err, label="Mean 10-Fold Training Error")
    plt.plot(data_size, test_err, label="Mean 10-Fold Test Error")
    plt.xlabel("Dataset size")
    plt.ylabel("Mean Squared Error")
    plt.title("kNN Cross-Validated Error vs Dataset Size")
    plt.legend()
    plt.grid()
    plt.figure()
    plt.plot(data_size, train_err, label="Mean 10-Fold Training Error")
    plt.plot(data_size, test_err, label="Mean 10-Fold Test Error")
    plt.xlabel("Dataset size")
    plt.ylabel("Mean Squared Error")
    plt.title("kNN Cross-Validated Error vs Dataset Size")
    plt.legend()
    plt.grid()
    plt.show()
    plt.show()
