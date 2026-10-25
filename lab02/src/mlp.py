"""Two-hidden-layer MLP in pure NumPy: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax.

Notation (batch of B rows):
    Z1 = X  @ W1 + b1   (B, W)     A1 = relu(Z1)
    Z2 = A1 @ W2 + b2   (B, W)     A2 = relu(Z2)
    Z3 = A2 @ W3 + b3   (B, 10)    P  = softmax(Z3)
    L  = -(1/B) * sum(Y * log P)
"""
import numpy as np

N_IN, N_OUT = 64, 10
INIT_STD = 0.01
PARAM_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


def init_params(width, rng, std=INIT_STD, scheme="normal"):
    """Weights ~ N(0, std) (std = 0.01), biases = 0.

    scheme="he" (control experiment only) uses std = sqrt(2 / fan_in) per layer instead.
    """
    def w(fan_in, fan_out):
        s = np.sqrt(2.0 / fan_in) if scheme == "he" else std
        return rng.normal(0.0, s, size=(fan_in, fan_out))

    return {
        "W1": w(N_IN, width),
        "b1": np.zeros(width),
        "W2": w(width, width),
        "b2": np.zeros(width),
        "W3": w(width, N_OUT),
        "b3": np.zeros(N_OUT),
    }


def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # shift for numerical stability; result unchanged
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    """Returns the probabilities P (B, 10) and a cache of intermediates for backward."""
    Z1 = X @ params["W1"] + params["b1"]
    A1 = relu(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = relu(Z2)
    Z3 = A2 @ params["W3"] + params["b3"]
    P = softmax(Z3)
    return P, (X, Z1, A1, Z2, A2, P)


def cross_entropy(P, Y, eps=1e-12):
    """Mean cross-entropy over the batch; Y is one-hot."""
    return float(-np.sum(Y * np.log(P + eps)) / Y.shape[0])


def backward(params, cache, Y):
    """Gradients of the mean cross-entropy w.r.t. every parameter (chain rule, see report §2)."""
    X, Z1, A1, Z2, A2, P = cache
    B = Y.shape[0]
    dZ3 = (P - Y) / B                     # softmax + cross-entropy combined
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(axis=0)
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (Z2 > 0)                  # ReLU derivative
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(params, grads, lr):
    for k in PARAM_NAMES:
        params[k] -= lr * grads[k]


def predict(params, X):
    return forward(params, X)[0].argmax(axis=1)
