# elastic_net_regression_comparison.py

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
# Calculating intercept and slope values           #
# for Linear Regression model                      #
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
# for Elastic Net Regression (alpha=10, l1_ratio=0.5)
#--------------------------------------------------#
from sklearn.linear_model import ElasticNet

# Elastic Net with balanced L1 and L2 penalties
elastic_net = ElasticNet(alpha=10, l1_ratio=0.5)

# Fit model to data
elastic_net.fit(x, y)

# Print slope and intercept
print(elastic_net.coef_)
print(elastic_net.intercept_)


#--------------------------------------------------#
# Calculating intercept and slope values            #
# for Elastic Net Regression (alpha=100, l1_ratio=0.5)
#--------------------------------------------------#
elastic_net2 = ElasticNet(alpha=100, l1_ratio=0.5)

# Fit model with stronger regularization
elastic_net2.fit(x, y)

# Print slope and intercept
print(elastic_net2.coef_)
print(elastic_net2.intercept_)


#--------------------------------------------------#
# Comparing all models graphically                  #
#--------------------------------------------------#
plt.figure(figsize=[5.5, 4])

# Plot original data points
plt.plot(x, y, 'b.')

# sorted_idx = np.argsort(x[:,0])
# x_sorted = x[sorted_idx]
# Linear Regression prediction (alpha = 0)
plt.plot(x, regressor.predict(x), color='red', label='alpha=0')

# Elastic Net predictions with different alphas
plt.plot(x, elastic_net.predict(x), color='green', label='alpha=10')
plt.plot(x, elastic_net2.predict(x), color='orange', label='alpha=100')

# Show legend and plot
plt.legend()
plt.show()


#--------------------------------------------------#
# Creating our own Elastic Net Regression class    #
#--------------------------------------------------#
class MyElasticNetRegression:
    
    def __init__(self, alpha=0.1, l1_ratio=0.5):
        # Constructor for MyElasticNetRegression class.
        # alpha     -> overall regularization strength
        # l1_ratio  -> balance between L1 and L2 penalties
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.m = None
        self.b = None
        
    def fit(self, x_train, y_train):
        # Fits a simplified Elastic Net Regression model
        # NOTE: sklearn uses coordinate descent internally.
        # This is a conceptual approximation.

        num = 0 # covariance(x,y)
        den = 0 # variance(x)
        
        # Computing numerator and denominator
        for i in range(x_train.shape[0]):
            num += (y_train[i] - y_train.mean()) * (x_train[i] - x_train.mean())
            den += (x_train[i] - x_train.mean()) ** 2
        
        # L1 and L2 penalty components
        l1_penalty = self.l1_ratio * self.alpha * np.sign(num)
        l2_penalty = (1 - self.l1_ratio) * self.alpha

        # Elastic Net–adjusted slope
        self.m = (num - l1_penalty) / (den + l2_penalty)

        # Intercept calculation
        self.b = y_train.mean() - (self.m * x_train.mean())
        
        # Print learned parameters
        print(self.m, self.b)
    
    def predict(self, x_test):
        # Predict output values using learned parameters
        return self.m * x_test + self.b


#--------------------------------------------------#
# Testing custom Elastic Net Regression             #
#--------------------------------------------------#
elastic_reg3 = MyElasticNetRegression(alpha=100, l1_ratio=0.5)
elastic_reg3.fit(x, y)
