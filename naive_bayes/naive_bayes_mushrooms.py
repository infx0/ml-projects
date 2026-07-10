from copy import deepcopy
import random
import math

ATTRIBUTES = [
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


def parse_data(file_name: str) -> list[list]:
    data = []
    file = open(file_name, "r")
    for line in file:
        datum = line.rstrip().split(",")
        data.append(datum)
    random.shuffle(data)
    return data


def create_folds(xs: list, n: int) -> list[list[list]]:
    k, m = divmod(len(xs), n)
    # be careful of generators...
    return list(xs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


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


def probability_of(nbc: dict, instance: dict, label: str) -> float:
    """
        `probability_of` calculates the probability for the label that's input into the function, given the trained Naive Bayes Classifier and observation instance. **Used By**: [classify](#classify)

    * **nbc** dict: the trained Naive Bayes Classifier, with prior and conditional probabilities, in a nested dict structure.
    * **instance** dict: the specific observation to use for calculating the probability for the given label.
    * **label** str:  the class label to evaluate.

    **returns** float: returns the probability of the class label given the evidence and NBC nested dict.
    """
    prob = nbc["priors"][label]
    for k, v in instance.items():
        prob *= nbc["conditionals"][label][k][v]
    return prob


def normalize(results: dict) -> dict:
    """
        `normalize` takes the raw probabilities for each class label and normalizes them so that they sum to 1. It also sorts the class probabilities in descending order. **Used By**: [classify](#classify)

    * **results** results: the raw probabilities from an inference of the NBC.

    **returns** dict: returns the same results dict but with the values normalized and sorted by descending probability.
    """
    total = sum(results.values())
    for k, v in results.items():
        results[k] = v / total
    sorted_results = dict(
        sorted(results.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_results


def find_best(results: dict) -> str:
    """

    `find_best` finds the class label with the highest priority and returns the label string, and higher up the call stack returns the string from the NBC train function. **Used By**: [train](#train)

    * **results** dict: a dict containing each class label and its probability.

    **returns** str: returns the class label with highest probability.
    """
    best = max(results, key=results.get)
    return best


def get_attr_types(
    training_data: list[dict], attributes: set[str]
) -> dict[str, set[str]]:
    """
        `get_attr_types` creates a dictionary of sets that contain all the known values for each feature based on the training data. This is required to correctly count feature value combinations for training the NBC. **Used By**: [train](#train)

    * **training_data** list[dict]: The training data as a list of dicts.
    * **attributes** set[str]: The feature attributes from which to compile the possible values for each feature.

    **returns** dict[str, set[str]]: returns a dictionary that contains all the known possible values that each feature can take on.
    """
    attr_types_dict = {attr: set() for attr in attributes}
    for row in training_data:
        for attr in attributes:
            attr_types_dict[attr].add(row[attr])
    return attr_types_dict


def get_class_totals(training_data: list[dict], target_attr: str) -> dict:
    """

    `get_class_totals` calculates the overall occurrences of each class label as part of the Naive Bayes Classifier function. **Used By**: [train](#train)

    * **training_data** list[dict]: The training data as a list of dicts.
    * **target_attr** str: The target attribute.

    **returns** dict: returns a dict with the total counts for each target class.
    """
    class_totals_dict = {}
    for row in training_data:
        if row[target_attr] not in class_totals_dict:
            class_totals_dict[row[target_attr]] = 0
        class_totals_dict[row[target_attr]] += 1
    return class_totals_dict


def get_counts(
    training_data: list[dict], attributes: set[str], target_attr: str = "eat"
) -> dict:
    """
        `get_counts` totals the counts for each possible feature value, over all attributes as a helper function in the main Naive Bayes Classifier training function. **Used By**: [train](#train)

    * **training_data** list[dict]: The training data as a list of dicts.
    * **attributes** set[str]: The feature attributes from which to total the counts.
    * **target_attr** str: The target attribute.

    **returns** dict: returns a dictionary containing the counts for each value that each attribute takes on.
    """
    target_dict = {}
    for row in training_data:
        tgt_val = row[target_attr]
        if tgt_val not in target_dict:
            target_dict[tgt_val] = {}
        for attr in attributes:
            attr_val = row[attr]
            if attr not in target_dict[tgt_val]:
                target_dict[tgt_val][attr] = {}
            if attr_val not in target_dict[tgt_val][attr]:
                target_dict[tgt_val][attr][attr_val] = 0
            target_dict[tgt_val][attr][attr_val] += 1
    return target_dict


def train(
    training_data: list[dict],
    attributes: set[str],
    target_attr: str = "eat",
    smoothing=True,
) -> dict:
    """

       `train` trains a Naive Bayesian Classifier by totaling the overall class probabilities, and then each conditioanl probability for every combination of feature value and class value. It also optionally implements +1 smooths during calculations. **Uses**: [get_counts](#get_counts), [get_attr_types](#get_attr_types), [get_class_totals](#get_class_totals) **Used By**: [cross_validate](#cross_validate)

    * **training_data** list[dict]: The training data as a list of dicts.
    * **attributes** set[str]: The feature attributes from which to calculate the prior and conditional probabilities.
    * **target_attr** str: The target attribute.
    * **smoothing** bool: A flag for whether to implement +1 smoothing when calculating the conditional probabilities.

    **returns** dict: returns a dictionary with "prior" and "conditionals" keys that contain all the necessary probabilities to do inference."""
    count_dict = get_counts(training_data, attributes, target_attr)
    attr_types_dict = get_attr_types(training_data, attributes)
    class_totals_dict = get_class_totals(training_data, target_attr)
    class_probs_dict = {
        key: class_totals_dict[key] / sum(class_totals_dict.values())
        for key in class_totals_dict.keys()
    }
    prob_dict = {}
    for tgt_val, attr_dict in count_dict.items():
        prob_dict[tgt_val] = {}
        denominator = class_totals_dict[tgt_val]
        for attr, attr_type_dict in attr_dict.items():
            prob_dict[tgt_val][attr] = {}
            num_unique_attrs = len(attr_types_dict[attr])
            for attr_val in attr_types_dict[attr]:
                attr_count = attr_type_dict.get(attr_val, 0)
                if smoothing:
                    prob_dict[tgt_val][attr][attr_val] = (attr_count + 1) / (
                        denominator + num_unique_attrs
                    )
                else:
                    prob_dict[tgt_val][attr][attr_val] = attr_count / denominator
    return {"priors": class_probs_dict, "conditionals": prob_dict}


def classify(nbc: dict, instance: dict) -> tuple[str, dict]:
    """

        `classify` calculates the probability of each class label, given a single observation, using the Naive Bayes Classify psuedocode. Part of the larger Naive Bayes Classifier pipeline. **Uses**: [probability_of](#probability_of), [normalize](#normalize), [find_best](#find_best) **Used By**: [classify_all](#classify_all)

    * **nbc** dict: the trained Naive Bayes Classifier, with prior and conditional probabilities, in a nested dict structure.
    * **instance** dict: the specific observation to use for calculating the probability for the given label.

    **returns** tuple[str, dict]: returns a tuple of the most likely class label, and a dict of all of the class probabilities.
    """
    results = {}
    for label in nbc["conditionals"]:
        results[label] = probability_of(nbc, instance, label)
    results = normalize(results)
    best = find_best(results)
    return best, results


def classify_all(
    nbc: dict, observations: list[dict], labeled: bool = True, tgt_attr: str = "eat"
) -> list[tuple[str, dict]]:
    """
        `classify_all` finds the most likely class label and set of probabilities for each label, for a series of observations. It is essentially a wrapper around the classify function. If an observation already has a truth label, it's removed before running the classifier. **Uses**: [classify](#classify) **Used By**: [cross_validate](#cross_validate)

    * **nbc** dict: the trained Naive Bayes Classifier, with prior and conditional probabilities, in a nested dict structure.
    * **observations** list[dict]: the list of observations to use for inference.
    * **labeled** bool: a flag that indicates whether to expect the truth label as part of each observation or not.
    * **tgt_attr** str: the target label that may be removed from the observation depending on the value of the labeled flag.

    **returns** type: returns a list of inferences based on the observations, where each result is a tuple with the most likely class label and associated dict with the probability of all class labels.
    """
    results = []
    for row in observations:
        instance = row.copy()
        if labeled:
            del instance[tgt_attr]
        result = classify(nbc, instance)
        results.append(result)
    return results


def evaluate(
    observations: list[dict], inferences: list[tuple[str, dict]], tgt_attr: str = "eat"
) -> float:
    """

    `evaluate` determines the error rate of a dataset, by comparing the predicted labels to the true labels. The ordered list of observations is assumed to correspond to the ordered list of inferences used as inputs.

    * **observations** list[dict]: the dataset for evaluation.
    * **inferences** list[tuple[str, dict]]: the Naive Bayes Classifier inferences, provided earlier up the call stack by the classify_all function.
    * **tgt_attr** str: the target attribute name to use for the truth label.

    **returns** type: returns the error rate of the given dataset.
    """
    n = len(observations)
    errors = 0
    for obs, inference in zip(observations, inferences):
        if obs[tgt_attr] != inference[0]:
            errors += 1
    error_rate = errors / n
    return error_rate


def cross_validate(
    observations: list[dict], attributes: set[str], num_folds: int, smoothing: bool
) -> tuple[list, list]:
    """
        `cross_validate` runs cross validation using the observations, specified set of attributes and the number of folds. It trains, and then classifies and evaluates on both the training set and test set. The error rate is then printed out for both and added to lists that are returned for debugging purposes. **Uses**: [classify_all](#classify_all), [evaluate](#evaluate), [create_folds](#create_folds), [train](#train)

    * **observations** list[dict]: the input dataset for validation that will create folds and then train and evaluate for each split.
    * **attributes** set[str]: the attribute set containing all the features of the dataset.
    * **num_folds** int: the number of splits for cross validation in the dataset.
    * **smoothing** bool: whether to use +1 smoothing or not.

    **returns** tuple[list, list]: returns the list of training and test errors.
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
        nbc = train(
            training_data=training_set, attributes=attributes, smoothing=smoothing
        )
        train_classified = classify_all(nbc, deepcopy(training_set))
        test_classified = classify_all(nbc, deepcopy(test_set))
        train_error_rate = evaluate(training_set, train_classified)
        test_error_rate = evaluate(test_set, test_classified)
        print(
            f"Fold {i + 1} error rates, train: {train_error_rate:.4f} test: {test_error_rate:.4f}"
        )
        train_error.append(train_error_rate)
        test_error.append(test_error_rate)
    return train_error, test_error


if __name__ == "__main__":
    data = parse_data(
        file_name="agaricus-lepiota.data"
    )  # this file should be in the same directory as the script
    cleaned_data = clean_data(data)
    formatted_data = [dict(zip(ATTRIBUTES, row)) for row in cleaned_data]
    root_attributes = ATTRIBUTES[1:]
    print("CROSS VALIDATION SMOOTHING ON")
    train_err, test_err = cross_validate(
        formatted_data, set(root_attributes), 10, smoothing=True
    )
    print("CROSS VALIDATION SMOOTHING OFF")
    train_err, test_err = cross_validate(
        formatted_data, set(root_attributes), 10, smoothing=False
    )
