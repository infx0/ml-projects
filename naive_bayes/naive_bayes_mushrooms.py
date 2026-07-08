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
    "habitat"
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
    return list(xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

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
    print(f"{len(data) - len(clean)} rows removed due to missing values. "
          f"Original dataset contained {len(data)} rows. "
          f"Cleaned dataset contains {len(clean)} rows.")
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
        results[k] = v/total
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    return sorted_results

def find_best(results: dict) -> str:
    """
    
`find_best` finds the class label with the highest priority and returns the label string, and higher up the call stack returns the string from the NBC train function. **Used By**: [train](#train)

* **results** dict: a dict containing each class label and its probability.

**returns** str: returns the class label with highest probability.
    """
    best = max(results, key=results.get)
    return best