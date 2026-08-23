"""Train support vector machines by solving their dual optimization problem.

The module remaps binary targets, supplies linear and radial-basis kernels, constructs
and solves the constrained quadratic program with CVXOPT, extracts support vectors and
the intercept, and uses the learned parameters to predict new observations.
"""

from cvxopt import matrix, solvers
import numpy as np
import numpy.typing as npt

# first we need to re-assign the targets to {-1, 1}. Let's remap the 0 targets to -1
X_train_svm = np.array(X_train)
X_test_svm = np.array(X_test)
Y_train_svm = Y_train.copy()
Y_test_svm = Y_test.copy()
Y_train_svm = np.array(Y_train_svm.replace(0, -1))
Y_test_svm = np.array(Y_test_svm.replace(0, -1))


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def rbf_kernel(x1: npt.NDArray, x2: npt.NDArray, gamma=0.001):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def train_svm(
    X: npt.NDArray, y: npt.NDArray, K: callable, C: float = 1.0, eps: float = 1e-6
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
    """
    This function assumes the target variables have already been mapped to {-1, 1}.

    For the dual form cvxopt needs min 1/2 * (X.T * P * X) - (q.T * X).
    Since our SVM is stated in terms of a maximization, we flip the sign to make it a minimization problem:

    max_a [sum_i(alpha) - 1/2*sum_i(sum_j(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j)))]
    turns into min_a [1/2 1/2*sum_i(sum_j(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))) - sum_i(alpha)]

    which is the format cvxopt expects.

    The dual summation term is equivalent in matrix form to alpha.T * P * alpha, with P=outer_product(y, y) * K.
    The single summation is equivalent in matrix form to ones.T * alpha,  which gives us the format cvxopt requires.

    For the constraints, cvxopt needs G * x <= h and A * x = b.
    Our SVM constraints are 0 <= alpha <= C for the soft margin formulation and sum_i(alpha_i * y_i) = 0.
    We can map this to the cvxopt format by letting A be a row vector of our target values, b=0,
    and then the constraints for alpha to be between 0 and C be a vertically stacked matrix for G and
    one row vector for h.

    Args:
        X (npt.NDArray): Training data of shape (n_samples, n_features).
        y (npt.NDArray): Target labels of shape (n_samples,), assumed to be mapped to {-1, 1}.
        K (callable): Kernel function, should accept two inputs x1, x2 and return a scalar.
        C (float, optional): Regularization parameter for soft-margin SVM. Default is 1.0.
        eps (float, optional): Threshold for identifying support vectors. Default is 1e-6.

    Returns:
        sv_X (npt.NDArray): Support vectors of shape (n_support_vectors, n_features).
        sv_y (npt.NDArray): Labels corresponding to the support vectors.
        sv_a (npt.NDArray): Lagrange multipliers (alphas) corresponding to support vectors.
        b (float): Bias term of the decision function.
    """

    # number of samples
    n = len(y)

    # define the kernel function across i, j
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k[i, j] = K(X[i], X[j])

    # cast all components as cvxopt matrix object with double-precision floats
    P = matrix(np.outer(y, y) * k, tc="d")
    q = matrix(-np.ones(n), tc="d")

    G = matrix(np.vstack((-np.eye(n), np.eye(n))), tc="d")
    h = matrix(np.hstack((np.zeros(n), np.ones(n) * C)), tc="d")
    A = matrix(y.astype(float), (1, n), tc="d")
    b = matrix(0.0, tc="d")

    # run the quadratic optimization
    sol = solvers.qp(P, q, G, h, A, b)

    # pull the solution from the solver output
    alphas = np.ravel(sol["x"])

    # the support vectors
    sv = (alphas > eps) & (alphas < C)
    sv_X = X[sv]
    sv_y = y[sv]
    sv_a = alphas[sv]
    num_sv = np.sum(sv)

    # use the support vectors to get the bias term, Bishop Eq. 7.18
    bias_sum = 0
    for i in range(num_sv):
        tmp = 0
        for j in range(num_sv):
            tmp += sv_a[j] * sv_y[j] * K(sv_X[i], sv_X[j])
        bias_sum += sv_y[i] - tmp
    b = bias_sum / num_sv

    alphas = np.ravel(sol["x"])
    print("DIAGNOSTICS")
    print("Alpha stats:")
    print("  min:", np.min(alphas))
    print("  max:", np.max(alphas))
    print("  mean:", np.mean(alphas))
    print("  nonzero count:", np.sum(alphas > 1e-6))
    print("Bias:", b)
    print("Support vector to Sample Size Ratio: ", num_sv / len(sv))

    return sv_X, sv_y, sv_a, b


def predict_svm(
    X: npt.NDArray,
    sv_X: npt.NDArray,
    sv_y: npt.NDArray,
    sv_a: npt.NDArray,
    K: callable,
    b: float,
    threshold_preds: bool,
) -> npt.NDArray:
    """
    Predict labels {-1, 1} given a test sample, and previously calculated support vectors and bias term.
    (Bishop Eq. 7.13)


    Args:
        X: Test data of shape (n_samples, n_features)
        sv_X: Support vectors from training
        sv_y: Labels of support vectors
        sv_a: Lagrange multipliers of support vectors
        K: Kernel function used during training
        b: Bias term from the trained SVM
        threshold_preds: If true, returns {-1, 1}, otherwise returns the raw scores

    Returns:
        Predicted labels {-1, 1} as a numpy array of shape (n_samples,)
    """
    out = []

    for x in X:
        tmp = 0
        for i in range(len(sv_X)):
            tmp += sv_a[i] * sv_y[i] * K(x, sv_X[i])
        y = tmp + b
        out.append(y)

    if threshold_preds:
        return np.array(np.sign(out))
    else:
        return np.array(out)


sv_X, sv_y, sv_a, b = train_svm(X_train, Y_train_svm, K=rbf_kernel, C=100.0)

# get the thresholded decisions
y_pred = predict_svm(X_test, sv_X, sv_y, sv_a, rbf_kernel, b, threshold_preds=True)

# also get the raw decision scores for the ROC curve
y_pred_svm_raw = predict_svm(
    X_test, sv_X, sv_y, sv_a, rbf_kernel, b, threshold_preds=False
)

# convert back to {0, 1} for metrics calculations
y_pred_svm = (y_pred == 1).astype(int)
y_true_svm = (Y_test_svm == 1).astype(int)
