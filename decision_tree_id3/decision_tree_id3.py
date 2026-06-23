from copy import deepcopy
import random
import math


def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: list, n: int) -> list[list[dict]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def is_homogenous(data: list[dict], attr: str = "eat") -> bool:
    """
        `is_homogenous` looks at a attribute from the dataset and determines if all targets have the same value. **Used by**: [train](#train)
    * **data** list[dict]: the dataset to evaluate.
    * **attr** str: the attribute name to evaluate in the target. Defaults to "eat".

    **returns** bool: returns True if all targets have the same value (e.g. "edible", "poisonous") or False if they don't.
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
        `majority_label` evaluates the target attributes (defaults to "eat") for the given dataset, and returns the value with the greatest number of occurrences. This is needed in the ID3 algorithm for cases where there are no more attributes to evaluate but the remaining data contains heterogenous target attributes. **Used by**: [train](#train)
    * **data** list[dict]: the input dataset to evaluate the target attribute from.
    * **attr** str: the target attribute, defaults to "eat".

    **returns** str: the target value with the greatest number of occurrences (i.e. the majority label).
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
        `pick_best_attribute` chooses the attribute as determined by information gain, i.e. the attribute with the lowest entropy. The best attribute will then be used to further split the data in the ID3 algorithm. **Used by**: [train](#train)

    * **data** list[dict]: the remaining dataset used to form the next decision split.
    * **attr** set[str]: the group of attributes/features in the data.
    * **target_attr** str: the attribute that forms the label for evaluation, e.g. "edible", "poisonous".

    **returns** tuple[str, float]: returns the best attribute to use for the next split, as well as the minimum calculated entropy, mainly used for debugging and unit test purposes. It will return None and the default min_entropy value if the function fails to find an attribute.
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
       `train` recursively trains a decision tree using the ID3 algorithm. It checks if any training data is remaining in the subtree, and if so, then checks to see if all the labels are homogenous. It returns the label if so. If they're not, it will then find the majority label if no attributes are left to split on. Otherwise, it will find the best remaining attribute according to information gain and recurse the tree. **Uses**: [is_homogenous](#is_homogenous), [majority_label](#majority_label), [pick_best_attribute](#pick_best_attribute), [train](#train)

    * **training_data** list[dict]: the remaining dataset to recursively train the decision tree on.
    * **attributes** set[dict]: the remaining attributes to use for splitting the tree.
    * **target_attr** str: the attribute that represents the target label, e.g. "edible", "poisonous". Defaults to "eat".
    * **default** str | None: the default label. Defaults to None.

    **returns** dict | None: returns the sub-tree represented as a nested dict. If there is no data left the function will return None.
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
        `classify_row` moves through a decision tree trained by the ID3 algorithm to classify a single observation. It contains a fallback label of "e" if an attribute choice is observed during test that wasn't seen in the training set. **Used by**: [classify](#classify) **Uses**: [classify_row](#classify_row)
    * **tree** dict: the pre-trainined decision tree as a nested dictionary.
    * **row** dict: the dictionary that corresponds to all the attributes for a single observation.
    * **default** str | None: the default label to assign if an attribute choice is observed during test that wasn't seen during training.

    **returns** str: returns the predicted label for the given observation.
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
        `classify` retrieves predicted classifications for a given dataset, as part of a pipeline to evaluate the efficacy of an ID3 decision tree. It wraps classify_row to form a prediction for each observation. **Uses**: [classify_row](#classify_row)

    * **tree** dict: the pre-trained decision tree use for classification.
    * **observations** list[dict]: the dataset to use for classification.
    * **pred_key** str: the string to use for the predictions key. Defaults to "pred".

    **returns** list[dict]: returns the dataset with an additional attribute for the prediction per observation.
    """
    for row in observations:
        row[pred_key] = classify_row(tree, row)
    return observations


def evaluate(
    observations: list[dict], pred_key: str = "pred", target_attr: str = "eat"
) -> float:
    """
        `evaluate` determines the error rate of a dataset, by comparing the predicted labels to the true labels.

    * **observations** list[dict]: the dataset for evaluation.
    * **pred_key** str: the key to use for the prediction attribute. Defaults to "pred".
    * **target_attr**: str: the attribute name to use for the truth label. Default to "eat".

    **returns** type: returns the error rate of the given dataset.
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
        `cross_validate` runs cross validation using the observations, specified set of attributes and the number of folds. It trains, and then classifies and evaluates on both the training set and test set. The error rate is then printed out for both and added to lists that are returned for debugging purposes. **Uses**: [classify](#classify), [evaluate](#evaluate), [create_folds](#create_folds)

    * **observations** list[dict]: the input dataset for validation that will create folds and then train and evaluate for each split.
    * **attributes** set[str]: the attribute set containing all the features of the dataset.
    * **num_folds** int: the number of splits for cross validation in the dataset.

    **returns** tuple[list, list]: returns the list of training and test errors
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
            f"Fold {i + 1} error rates, train: {train_error_rate:.4f} test: {test_error_rate:.4f}"
        )
        train_error.append(train_error_rate)
        test_error.append(test_error_rate)
    return train_error, test_error


def pretty_print_tree(tree: dict, spaces: str = "") -> bool:
    """
        `pretty_print_tree` prints out the ID3 trained decision tree in a human-friendly readable format. The function uses logic similar to classify_row to walk through the tree.
    * **tree** dict: the trained decision tree.

    **returns** bool: Returns True if the function executed without error, mainly used for unit test purposes.
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
        `clean_data` removes observations from the dataset that contain missing observations. The symbol for a missing observation defaults to "?".
    * **data** list[list[str]]: the input data set. Assumed to be a list of lists of strings.
    * **missing_str** str: the symbol indicating missing data. Lists containing the symbol will be removed. Defaults to "?".

    **returns** list[list[str]]: the cleaned dataset.
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
