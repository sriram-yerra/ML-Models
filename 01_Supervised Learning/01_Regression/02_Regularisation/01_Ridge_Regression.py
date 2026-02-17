# ridge_regression_comparison.py

#--------------------------------------------------#
# Importing the Libraries                           #
#--------------------------------------------------#
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np


#--------------------------------------------------#
# Creating a synthetic regression dataset          #
#--------------------------------------------------#
x, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=20,
    random_state=13
)

# Visualizing the generated dataset
plt.figure(figsize=[4.5, 3])
plt.scatter(x, y)
plt.show()


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Linear Regression model                       #
#--------------------------------------------------#
from sklearn.linear_model import LinearRegression

# Create Linear Regression object
regressor = LinearRegression()

# Fit model to data
regressor.fit(x, y)

# Print slope (coefficient) and intercept
print(regressor.coef_)
print(regressor.intercept_)


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Ridge Regression model (Alpha = 10)           #
#--------------------------------------------------#
from sklearn.linear_model import Ridge

# Ridge Regression with moderate regularization
regularizer = Ridge(alpha=10)

# Fit model to data
regularizer.fit(x, y)

# Print slope and intercept
print(regularizer.coef_)
print(regularizer.intercept_)


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Ridge Regression model (Alpha = 100)          #
#--------------------------------------------------#
regularizer2 = Ridge(alpha=100)

# Fit model with stronger regularization
regularizer2.fit(x, y)

# Print slope and intercept
print(regularizer2.coef_)
print(regularizer2.intercept_)


#--------------------------------------------------#
# Comparing all models graphically                  #
#--------------------------------------------------#
plt.figure(figsize=[5.5, 4])

# Plot original data points
plt.plot(x, y, 'b.')

# Linear Regression prediction (alpha = 0)
plt.plot(x, regressor.predict(x), color='red', label='alpha=0')

# Ridge Regression predictions with different alphas
plt.plot(x, regularizer.predict(x), color='green', label='alpha=10')
plt.plot(x, regularizer2.predict(x), color='orange', label='alpha=100')

# Show legend and plot
plt.legend()
plt.show()


#--------------------------------------------------#
# Creating our own Ridge Regression class           #
#--------------------------------------------------#
class MyRidgeRegression:
    
    def __init__(self, alpha=0.1):
        # Constructor for MyRidgeRegression class.
        # Initializes regularization parameter (alpha)
        # and sets slope (m) and intercept (b) as None.
        self.alpha = alpha
        self.m = None
        self.b = None
        
    def fit(self, x_train, y_train):
        # Fits the Ridge Regression model using a
        # closed-form–inspired approach (manual computation)

        num = 0  # Numerator for slope calculation
        den = 0  # Denominator for slope calculation
        
        # Loop over each training example
        for i in range(x_train.shape[0]):
            num = num + (
                (y_train[i] - y_train.mean()) *
                (x_train[i] - x_train.mean())
            )
            den = den + (
                (x_train[i] - x_train.mean()) *
                (x_train[i] - x_train.mean())
            )
        
        # Ridge-adjusted slope formula
        self.m = num / (den + self.alpha)

        # Intercept calculation
        self.b = y_train.mean() - (self.m * x_train.mean())
        
        # Print learned parameters
        print(self.m, self.b)
    
    def predict(x_test):
        # Predicts output for test data
        # (method intentionally left unimplemented in source)
        pass


#--------------------------------------------------#
# Testing custom Ridge Regression implementation   #
#--------------------------------------------------#
regularizer3 = MyRidgeRegression(alpha=100)
regularizer3.fit(x, y)

