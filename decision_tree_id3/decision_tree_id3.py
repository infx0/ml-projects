"""
This script trains and evaluates an ID3 decision tree classifier on the
agaricus-lepiota mushroom dataset. It reads the comma-separated data file, removes
rows with missing values, maps each row to named mushroom attributes, and builds a
nested-dictionary decision tree that predicts whether a mushroom is edible or
poisonous. The tree is trained recursively by choosing the remaining attribute with
the lowest weighted entropy at each split, with majority-label fallbacks for empty
or ambiguous branches.

When run directly, the script trains a decision tree on the cleaned dataset, performs
10-fold cross validation to print training and test error rates, and then prints the
learned tree in a readable indented form.
"""

from copy import deepcopy
import random
import math


def parse_data(file_name: str) -> list[list]:
    """
    Reads comma-separated mushroom data from a file into a list of rows, then randomly
    shuffles the rows before returning them.

    Args:
        file_name (str): The name of the data file to read.

    Returns:
        list[list]: Returns the parsed and shuffled dataset as a list of row values.
    """
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: list, n: int) -> list[list[dict]]:
    """
    Splits a dataset into a specified number of folds for cross validation. The folds
    are kept as evenly sized as possible, with any extra rows distributed across the
    earliest folds.

    Args:
        xs (list): The dataset to split into folds.
        n (int): The number of folds to create.

    Returns:
        list[list[dict]]: Returns a list containing the folded subsets of the dataset.
    """
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def is_homogenous(data: list[dict], attr: str = "eat") -> bool:
    """
    Looks at a attribute from the dataset and determines if all targets have the same
    value.

    Args:
        data (list[dict]): The dataset to evaluate.
        attr (str): The attribute name to evaluate in the target. Defaults to
            "eat".

    Returns:
        bool: Returns True if all targets have the same value (e.g. "edible",
            "poisonous") or False if they don't.
    """
    var = []
    for row in data:
        var.append(row[attr])
    if len(set(var)) == 1:
        return True
    else:
        return False


def majority_label(data: list[dict], attr: str = "eat") -> str:
    """
    Evaluates the target attributes (defaults to "eat") for the given dataset, and
    returns the value with the greatest number of occurrences. This is needed in the ID3
    algorithm for cases where there are no more attributes to evaluate but the remaining
    data contains heterogenous target attributes.

    Args:
        data (list[dict]): The input dataset to evaluate the target attribute
            from.
        attr (str): The target attribute, defaults to "eat".

    Returns:
        str: The target value with the greatest number of occurrences (i.e. the majority
            label).
    """
    counts = {}
    for row in data:
        counts[row[attr]] = counts.get(row[attr], 0) + 1

    label = None
    max_count = -999
    for k, v in counts.items():
        if v > max_count:
            max_count = v
            label = k
    return label


def pick_best_attribute(
    data: list[dict], attr: set[str], target_attr: str = "eat"
) -> tuple[str, float]:
    """
    Chooses the attribute as determined by information gain, i.e. the attribute with the
    lowest entropy. The best attribute will then be used to further split the data in
    the ID3 algorithm.

    Args:
        data (list[dict]): The remaining dataset used to form the next decision
            split.
        attr (set[str]): The group of attributes/features in the data.
        target_attr (str): The attribute that forms the label for evaluation,
            e.g. "edible", "poisonous".

    Returns:
        tuple[str, float]: Returns the best attribute to use for the next split, as well
            as the minimum calculated entropy, mainly used for debugging and unit test
            purposes. It will return None and the default min_entropy value if the
            function fails to find an attribute.
    """
    min_entropy = 99999.0
    best_attr = None
    for a in attr:
        attr_entropy = 0.0
        unique_vals = set(row[a] for row in data)
        for val in unique_vals:
            counts = {}
            row_count = 0
            for row in data:
                if row[a] == val:
                    row_count += 1
                    counts[row[target_attr]] = counts.get(row[target_attr], 0) + 1
            subset_entropy = 0.0
            for count in counts.values():
                subset_entropy -= math.log2(count / row_count) * count / row_count
            attr_entropy += (row_count / len(data)) * subset_entropy
        if attr_entropy < min_entropy:
            min_entropy = attr_entropy
            best_attr = a
    return best_attr, min_entropy


def train(
    training_data: list[dict],
    attributes: set[str],
    target_attr: str = "eat",
    default: str | None = None,
) -> dict | None:
    """
    Recursively trains a decision tree using the ID3 algorithm. It checks if any
    training data is remaining in the subtree, and if so, then checks to see if all the
    labels are homogenous. It returns the label if so. If they're not, it will then find
    the majority label if no attributes are left to split on. Otherwise, it will find
    the best remaining attribute according to information gain and recurse the tree.

    Args:
        training_data (list[dict]): The remaining dataset to recursively train the
            decision tree on.
        attributes (set[dict]): The remaining attributes to use for splitting the tree.
        target_attr (str): The attribute that represents the target label, e.g.
            "edible", "poisonous". Defaults to "eat".
        default (str | None): The default label. Defaults to None.

    Returns:
        dict | None: Returns the sub-tree represented as a nested dict. If there is no
            data left the function will return None.
    """
    if not training_data:
        return default
    if is_homogenous(training_data):
        return training_data[0][target_attr]
    if not attributes:
        return majority_label(training_data, target_attr)
    best_attr, _ = pick_best_attribute(training_data, attributes)
    node = {best_attr: {}}
    default_label = majority_label(training_data, target_attr)
    for val in {row[best_attr] for row in training_data}:
        subset = [row for row in training_data if row[best_attr] == val]
        node[best_attr][val] = train(
            subset, attributes - {best_attr}, target_attr, default_label
        )
    return node


def classify_row(tree: dict, row: dict, default: str | None = None) -> str:
    """
    Moves through a decision tree trained by the ID3 algorithm to classify a single
    observation. It contains a fallback label of "e" if an attribute choice is observed
    during test that wasn't seen in the training set.

    Args:
        tree (dict): The pre-trainined decision tree as a nested dictionary.
        row (dict): The dictionary that corresponds to all the attributes for a
            single observation.
        default (str | None): The default label to assign if an attribute choice is
            observed during test that wasn't seen during training.

    Returns:
        str: Returns the predicted label for the given observation.
    """
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = row[attr]
    if value not in tree[attr]:
        return default if default is not None else "e"
    branch = tree[attr][value]

    return classify_row(branch, row, default)


def classify(tree: dict, observations: list[dict], pred_key="pred") -> list[dict]:
    """
    Retrieves predicted classifications for a given dataset, as part of a pipeline to
    evaluate the efficacy of an ID3 decision tree. It wraps classify_row to form a
    prediction for each observation.

    Args:
        tree (dict): The pre-trained decision tree use for classification.
        observations (list[dict]): The dataset to use for classification.
        pred_key (str): The string to use for the predictions key. Defaults to "pred".

    Returns:
        list[dict]: Returns the dataset with an additional attribute for the prediction
            per observation.
    """
    for row in observations:
        row[pred_key] = classify_row(tree, row)
    return observations


def evaluate(
    observations: list[dict], pred_key: str = "pred", target_attr: str = "eat"
) -> float:
    """
    Determines the error rate of a dataset, by comparing the predicted labels to the
     true labels.

    Args:
        observations (list[dict]): The dataset for evaluation.
        pred_key (str): The key to use for the prediction attribute. Defaults to "pred".
        target_attr (str): The attribute name to use for the truth label. Default to
            "eat".

    Returns:
        type: Returns the error rate of the given dataset.
    """
    n = len(observations)
    errors = 0
    for row in observations:
        if row[pred_key] != row[target_attr]:
            errors += 1
    error_rate = errors / n
    return error_rate


def cross_validate(
    observations: list[dict], attributes: set[str], num_folds: int
) -> tuple[list, list]:
    """
    Runs cross validation using the observations, specified set of attributes and the
    number of folds. It trains, and then classifies and evaluates on both the training
    set and test set. The error rate is then printed out for both and added to lists
    that are returned for debugging purposes.

    Args:
        observations (list[dict]): The input dataset for validation that will create
            folds and then train and evaluate for each split.
        attributes (set[str]): The attribute set containing all the features of the
            dataset.
        num_folds (int): The number of splits for cross validation in the dataset.

    Returns:
        tuple[list, list]: Returns the list of training and test errors.
    """
    random.seed(42)
    random.shuffle(observations)
    folded_data = create_folds(observations, num_folds)
    train_error = []
    test_error = []
    for i in range(len(folded_data)):
        train_folds = folded_data[:i] + folded_data[i + 1 :]
        training_set = sum(train_folds, [])
        test_set = folded_data[i]
        tree = train(training_set, attributes)
        train_classified = classify(tree, deepcopy(training_set))
        test_classified = classify(tree, deepcopy(test_set))
        train_error_rate = evaluate(train_classified)
        test_error_rate = evaluate(test_classified)
        print(
            f"Fold {i + 1} error rates, "
            f"train: {train_error_rate:.4f} "
            f"test: {test_error_rate:.4f}"
        )
        train_error.append(train_error_rate)
        test_error.append(test_error_rate)
    return train_error, test_error


def pretty_print_tree(tree: dict, spaces: str = "") -> bool:
    """
    Prints out the ID3 trained decision tree in a human-friendly readable format. The
    function uses logic similar to classify_row to walk through the tree.

    Args:
        tree (dict): The trained decision tree.

    Returns:
        bool: Returns True if the function executed without error, mainly used for unit
            test purposes.
    """
    if not isinstance(tree, dict):
        print(f"{spaces}->{tree}")
        return

    attr = next(iter(tree))
    branches = tree[attr]

    if isinstance(branches, str):
        print(f"{attr}: {branches}")
        return True

    for value, subtree in branches.items():
        print(f"{spaces}{attr}: {value}")
        pretty_print_tree(subtree, spaces=spaces + "  ")

    return True


def clean_data(data: list[list[str]], missing_str: str = "?") -> list[list[str]]:
    """
    Removes observations from the dataset that contain missing observations. The symbol
    for a missing observation defaults to "?".

    Args:
        data (list[list[str]]): The input data set. Assumed to be a list of lists of
            strings.
        missing_str (str): The symbol indicating missing data. Lists containing the
            symbol will be removed. Defaults to "?".

    Returns:
        list[list[str]]: The cleaned dataset.
    """
    clean = []
    for row in data:
        if missing_str in row:
            continue
        else:
            clean.append(row)
    print(
        f"{len(data) - len(clean)} rows removed due to missing values. "
        f"Original dataset contained {len(data)} rows. "
        f"Cleaned dataset contains {len(clean)} rows."
    )
    return clean


if __name__ == "__main__":
    attributes = [
        "eat",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ]

    data = parse_data(
        file_name="agaricus-lepiota.data"
    )  # this file should be in the same directory as the script
    cleaned_data = clean_data(data)
    formatted_data = [dict(zip(attributes, row)) for row in cleaned_data]
    root_attributes = attributes[1:]
    decision_tree = train(training_data=formatted_data, attributes=set(root_attributes))
    train_err, test_err = cross_validate(formatted_data, set(root_attributes), 10)
    pretty_print_tree(decision_tree)
