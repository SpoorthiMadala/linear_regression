import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(42)
x = np.random.randint(1, 10, 50).reshape((50, 1))
y = 0.9 * x + np.random.normal(0, 1, size=x.shape) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
plt.scatter(x, y, color="red", label="Data Points")
plt.plot(x_test, y_pred, color="blue", label="Regression Line")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Estimation of Price Based on Area")
plt.legend()
plt.show()
