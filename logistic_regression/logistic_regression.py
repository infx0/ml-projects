import numpy as np
import matplotlib.pyplot as plt
import random


def view_sensor_image(data):
    figure = plt.figure(figsize=(4, 4))
    axes = figure.add_subplot(1, 1, 1)
    pixels = np.array([255 - p * 255 for p in data[:-1]], dtype="uint8")
    pixels = pixels.reshape((4, 4))
    axes.set_title("Left Camera:" + data[-1])
    axes.imshow(pixels, cmap="gray")
    plt.show()
    plt.close()


def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.30, 0.07)  # (0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.70, 0.07)  # (0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v

    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


plain = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
forest = [
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    0.0,
    0.0,
]
hills = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
swamp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]

figure = plt.figure(figsize=(20, 6))

axes = figure.add_subplot(1, 3, 1)
pixels = np.array([255 - p * 255 for p in plain], dtype="uint8")
pixels = pixels.reshape((4, 4))
axes.set_title("Left Camera")
axes.imshow(pixels, cmap="gray")

axes = figure.add_subplot(1, 3, 2)
pixels = np.array([255 - p * 255 for p in forest], dtype="uint8")
pixels = pixels.reshape((4, 4))
axes.set_title("Front Camera")
axes.imshow(pixels, cmap="gray")

axes = figure.add_subplot(1, 3, 3)
pixels = np.array([255 - p * 255 for p in hills], dtype="uint8")
pixels = pixels.reshape((4, 4))
axes.set_title("Right Camera")
axes.imshow(pixels, cmap="gray")

plt.show()
plt.close()

clean_data = {
    "plains": [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "plains",
        ]
    ],
    "forest": [
        [
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            "forest",
        ],
        [
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            "forest",
        ],
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            "forest",
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            "forest",
        ],
    ],
    "hills": [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "hills",
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "hills",
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            "hills",
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            "hills",
        ],
    ],
    "swamp": [
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "swamp",
        ],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            "swamp",
        ],
    ],
}

view_sensor_image(clean_data["forest"][0])
view_sensor_image(clean_data["swamp"][0])
view_sensor_image(blur(clean_data["swamp"][0]))


def prepend_bias(data: list[tuple[list, int]]) -> list[tuple[list, int]]:
    """
        `prepend_bias` prepends each data row in the dataset with a "1" which corresponds to the 'imaginary' observation associated with theta_0, the bias term.  **Used by**: [learn_model](#learn_model), [apply_model](#apply_model)

    * **data** list[tuple[list, int]]: the input data values that will get the prepended 1's.

    **returns** list[tuple[list, int]]: the updates observations.
    """
    for row in data:
        row[0].insert(0, 1.0)
    return


def sigmoid(thetas: list[float], vals: list[float]) -> float:
    """
        `sigmoid` calculates the result of the sigmoid function, 1/(1+exp(-(theta*vals))) for a vector of thetas and values. **Used by**: [calculate_error](#calculate_error), [derivative](#derivative), [apply_model](#apply_model)

    * **thetas** list[float]: the theta values to use in the function, i.e. logistic regression parameter estimates.
    * **vals** list[float]: the values to use in the function, i.e. the real-valued data for calculations.


    **returns** float: the result of the function given the input vectors thetas and vals.
    """
    x = 0.0
    for theta, val in zip(thetas, vals):
        x += theta * val
    result = 1 / (1 + np.exp(-x))
    return result


def calculate_error(thetas: list[float], data: list[tuple[list, int]]) -> float:
    """
        `calculate_error` calculates the error as defined with cross entropy loss, for use in gradient descent. **Uses:** [sigmoid](#sigmoid) **Used by**: [learn_model](#learn_model)

    * **thetas** list[float]: the theta values to use in the function, i.e. logistic regression parameter estimates.
    * **data** list[tuple[list, int]]: the values to use in the function, i.e. the real-valued data for calculations.

    **returns** float: the error as calculated by using cross entropy loss.
    """
    J = 0.0
    for row in data:
        yi = row[1]
        yi_hat = sigmoid(thetas=thetas, vals=row[0])
        J += yi * np.log(yi_hat) + (1 - yi) * np.log(1 - yi_hat)
    J *= -1 / len(data)
    return J


def derivative(j: int, thetas: list[float], data: list[tuple[list, int]]) -> float:
    """
        `derivative` calculates the partial derivate of the loss function with respect to a particular theta_j, used in updating each theta. **Uses:** [sigmoid](#sigmoid) **Used by**: [learn_model](#learn_model)

    * **j** list[float]: the particular j used to index into the observation, corresponding to the theta that will be updated.
    * **thetas** list[float]: the theta values to use in the function, i.e. logistic regression parameter estimates.
    * **data** list[tuple[list, int]]: the values to use in the function, i.e. the real-valued data for calculations.

    **returns** float: the partial derivative value.
    """
    dJ = 0.0
    for row in data:
        xij = row[0][j]
        yi = row[1]
        yi_hat = sigmoid(thetas=thetas, vals=row[0])
        dJ += (yi_hat - yi) * xij
    dJ *= 1 / len(data)
    return dJ


def generate_data(
    data: dict[str, list[list[float]]], n: int, key_label: str
) -> list[tuple[list, int]]:
    labels = list(data.keys())
    labels.remove(key_label)

    total_labels = len(labels)
    result = []
    # create n "not label" and code as y=0
    count = 1
    while count <= n:
        label = labels[count % total_labels]
        datum = blur(random.choice(data[label]))
        xs = datum[0:-1]
        result.append((xs, 0))
        count += 1

    # create n "label" and code as y=1
    for _ in range(n):
        datum = blur(random.choice(data[key_label]))
        xs = datum[0:-1]
        result.append((xs, 1))
    random.shuffle(result)
    return result


def learn_model(data: list[tuple[list, int]], verbose: bool = False) -> list[float]:
    """
        `learn_model` trains a one-vs-rest logistic regression classifier for hills vs not hills, using gradient descent. It also uses an adaptive learning rate that decreases as the iterations increase. **Uses:** [prepend_bias](#prepend_bias), [calculate_error](#calculate_error), [derivative](#derivative)

    * **data** list[tuple[list, int]]: the pixel values from our simulated cameras, along with the truth labels.
    * **verbose** bool: whether to print the error at every iteration or only every 1000 iterations.

    **returns** list[float]: the theta values after running gradient descent until convergence (linear model parameters).
    """
    epsilon = 1e-5
    alpha = 0.1
    iter = 0
    data = prepend_bias(data=data)
    num_thetas = len(data[0][0])
    thetas = [random.uniform(-1, 1) for _ in range(num_thetas)]
    previous_error = 0.0
    current_error = calculate_error(thetas=thetas, data=data)
    while abs(current_error - previous_error) > epsilon:
        new_thetas = []
        for j in range(len(thetas)):
            new_thetas.append(thetas[j] - alpha * derivative(j, thetas, data))
        thetas = new_thetas
        previous_error = current_error
        current_error = calculate_error(thetas=thetas, data=data)
        iter += 1
        if verbose or iter % 1000 == 0:
            print("iter ", iter, "current error ", current_error)
        if iter % 1000 == 0:
            alpha *= 0.9
    return thetas


def apply_model(model: list[float], test_data: list[tuple[list, int]]) -> list[tuple]:
    """
        `apply_model` uses previously learned linear model parameters (our thetas from learn_model) to predict labels using a logit link function. **Uses:** [prepend_bias](#prepend_bias), [sigmoid](#sigmoid)

    * **model** list[float]: the list of model parameters (thetas) from our gradient descent algorithm, for use in predicting class labels.
    * **test_data** list[tuple[list, int]]: the pixel values from our simulated cameras, along with the truth labels. Expected to be previously partitioned off as a test set from the original dataset.

    **returns** list[tuple]: a list of tuples with the first element in each tuple representing the probability of a positive predicted label and second element representing the truth label.
    """
    data = prepend_bias(data=test_data)
    results = []
    for row in data:
        truth = row[1]
        vals = row[0]
        estimate = sigmoid(thetas=model, vals=vals)
        results.append((estimate, truth))
    return results


def evaluate(results: list[tuple]) -> None:
    """
        `evaluate` evalutes the results from a logistic regression classifier, and prints out the error rate as well as confusion matrix parameters. It uses maximum likelihood thresholding to assign a class for the predicted label. It does not return anything.

    * **results** list[tuple]: a list of tuples with the first element in each tuple representing the probability of a positive predicted label and second element representing the truth label.
    """
    conf_matrix = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for r in results:
        if round(r[0]) == 0 and r[1] == 0:
            conf_matrix["TN"] += 1
        elif round(r[0]) == 1 and r[1] == 1:
            conf_matrix["TP"] += 1
        elif round(r[0]) == 0 and r[1] == 1:
            conf_matrix["FN"] += 1
        elif round(r[0]) == 1 and r[1] == 0:
            conf_matrix["FP"] += 1
    error_rate = (conf_matrix["FN"] + conf_matrix["FP"]) / (len(results))
    print("error_rate: ", error_rate)
    print(conf_matrix)


results = generate_data(clean_data, 5, "hills")
for result in results:
    print(result)

train_data = generate_data(clean_data, 100, "hills")
for i in range(10):
    print(train_data[i])

test_data = generate_data(clean_data, 100, "hills")

model = learn_model(train_data, False)

results = apply_model(model, test_data)
print(results)

evaluate(results)
