from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# import pandas.DataFrame as df

diabetes = load_diabetes()

# print(diabetes)
print(diabetes.DESCR)
print(diabetes.feature_names)

x = pd.DataFrame(diabetes.data)
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

y = diabetes.target

print(f"🧮 Total samples: {X.shape[0]}, Features: {X.shape[1]}")

# print(x)
print(X.head(5))
# print(diabetes.data)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📊 R² Score (Goodness of Fit): {r2:.4f}")

print("\n🖼️ Plotting Actual vs Predicted values...")
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, color="teal", edgecolor="black")
plt.xlabel("Actual Target Values")
plt.ylabel("Predicted Target Values")
plt.title("Actual vs Predicted (Linear Regression on Diabetes Dataset)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n🧠 Analyzing feature importance (model coefficients)...")
coef_df = pd.DataFrame({
    'Feature': diabetes.feature_names,
    'Coefficient': model.coef_
})
coef_df_sorted = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
print(coef_df_sorted)

print("\n✅ Linear Regression pipeline completed successfully!")



