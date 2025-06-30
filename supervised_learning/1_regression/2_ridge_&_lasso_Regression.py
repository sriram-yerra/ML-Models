# Lasso Regression: == l1 Regularisation 
# Penality term --> (lambda * |w|) : Feature selection happens here, since some become zero.
# Adds L1 penalty → can zero out coefficients (feature selection).
# Penalizes large weights using absolute values.
# Can shrink some weights to zero, thus performing feature selection.

# Ridge Regression: == l2 Regularisation 
# Penality term --> (lambda * (w)^^2)
# Adds L2 penalty → shrinks coefficients.
# Penalizes large weights using squared values.
# Keeps all features but shrinks them.

# pip install scikit-learn matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error, r2_score

# sklearn. : datasets, model_selection, linear_model, metrics

iris = load_iris() 
# iris is the object 

X = pd.DataFrame(iris.data, columns=iris.feature_names)
# y = pd.DataFrame(iris.target)
y = pd.Series(iris.target)

# print(X.head(5))
# print(y.head(5))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# sns.scatterplot(x=y_test, y=y_pred_ridge.ravel(), color='blue')
sns.scatterplot(x=y_test, y=y_pred_ridge.flatten(), color='blue')
plt.title("Ridge: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)

# figure, subplots, scatterplot, title, xlable, ylable, grid

plt.subplot(1, 2, 2)
# sns.scatterplot(x=y_test, y=y_pred_lasso.ravel(), color='orange')
sns.scatterplot(x=y_test, y=y_pred_lasso.flatten(), color='orange')
plt.title("Lasso: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(True)

plt.tight_layout()
plt.show()

# Show coefficients
coef_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Ridge Coef': ridge.coef_.flatten(),
    'Lasso Coef': lasso.coef_.flatten()
})

print("\n📌 Coefficients Comparison:")
print(coef_df)

# print("\n Ridge and Lasso Regression completed.")

