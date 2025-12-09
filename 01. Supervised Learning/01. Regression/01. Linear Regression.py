import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import generate_linear_regression_data, split_data


# =========================
# Mean Squared Error Loss
# =========================
class MSE:
    def __call__(self, y_pred, y_true):
        # MSE = (1/n) * Σ(y_true - y_pred)^2
        return np.sum((y_true - y_pred) ** 2) / y_true.size
    
    def grad(self, y_pred, y_true):
        # Gradient of MSE w.r.t y_pred
        # d/dy_pred = -2 * (y_true - y_pred) / n
        return -2 * (y_true - y_pred) / y_true.size


# =========================
# Stochastic Gradient Descent Regressor
# =========================
class SGDRegressor:
    def __init__(self, n_iterations=100, lr=0.0001):
        # Number of epochs
        self.n_iterations = n_iterations
        # Learning rate
        self.lr = lr
        # Model parameters
        self.weight = None
        self.bias = None
        # Loss function
        self.loss_fn = MSE()

    def init_weights(self, n_features):
        # Initialize weights only once
        if self.weight is None or self.bias is None:
            # self.weight = np.random.uniform(-1, 1, (1, n_features)) if self.weight is None else self.weight
            # self.weight = np.random.normal(0, pow(n_features, -0.5), (1, n_features)) if self.weight is None else self.weight
            # self.weight = np.random.normal(0, 1, (1, n_features)) if self.weight is None else self.weight

            # Xavier initialization:
            # Keeps variance stable across layers
            stdv = 1 / np.sqrt(n_features)
            self.weight = np.random.uniform(-stdv, stdv, (1, n_features)) if self.weight is None else self.weight
            
            # Bias initialized to zero
            self.bias = np.zeros((1, 1)) if self.bias is None else self.bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.init_weights(n_features)

        losses = []

        # Progress bar for epochs
        tqdm_range = tqdm(range(self.n_iterations), total=self.n_iterations)

        for i in range(self.n_iterations):
            tqdm_range.update(1)

            # Loop over samples (SGD: one sample at a time)
            for x_true, y_true in zip(X, y):

                # Ensure column vector shapes
                y_true = y_true[:, np.newaxis]
                x_true = x_true[None, ...] if x_true.ndim == 1 else x_true

                # Forward pass: ŷ = XW + b
                y_pred = np.matmul(x_true, self.weight) + self.bias

                # Compute loss
                loss = self.loss_fn(y_pred, y_true)

                # Compute gradient of loss w.r.t prediction
                grad = self.loss_fn.grad(y_pred, y_true)

                # Weight gradient: ∂L/∂W
                self.weight -= self.lr * np.matmul(grad.T, x_true)

                # Bias gradient: ∂L/∂b
                self.bias -= self.lr * np.sum(grad)

                # Store loss value
                losses.append(loss)

                tqdm_range.set_description(
                    f'epoch: {i + 1}/{self.n_iterations}, loss: {loss:.7f}'
                )

        return losses
    
    def predict(self, X):
        # Vectorized prediction
        return np.matmul(X, self.weight) + self.bias


# =========================
# Ordinary Least Squares (Closed-form Solution)
# =========================
class OrdinaryLeastSquares:
    def __init__(self):
        self.b = None  # coefficient vector

    def add_bias(self, X):
        # Add bias column (x₀ = 1)
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    def fit(self, X, y):
        X = self.add_bias(X)

        # Normal Equation:
        # b* = (XᵀX)^(-1) Xᵀ y
        self.b = np.linalg.inv(X.T @ X) @ X.T @ y
        return self.b
    
    def predict(self, X):
        X = self.add_bias(X)
        return X @ self.b


# =========================
# Main Execution
# =========================
if __name__ == '__main__':

    # Generate synthetic linear regression data
    X_train, y_train, true_coefs = generate_linear_regression_data(300)

    # Split into train and test
    X_train, X_test, y_train, y_test = split_data(
        X_train, y_train, ratio=0.25
    )

    # Visualization setup
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Ground truth points
    plt.scatter(X_test, y_test, color='g', s=10, label='Ground truth')

    # ==== SGD Model ====
    model = SGDRegressor(n_iterations=1000)
    losses = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.plot(X_test, y_pred, 'red', label='Gradient descent')

    # ==== OLS Model ====
    model = OrdinaryLeastSquares()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.plot(X_test, y_pred, 'orange', label='Ordinary least squares')

    # ==== True Function ====
    y_true = np.dot(X_test, true_coefs)
    plt.plot(X_test, y_true, 'blue', label='True coefficients')

    # Final plot adjustments
    plt.legend(loc=2)
    plt.grid(True, linestyle='-', color='0.75')
    plt.show()
