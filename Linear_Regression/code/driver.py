from lin_reg import Linear_Regression as classifier
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


data_file_name = 'data/housing.data'

classifier = classifier(data_file_name)
data = classifier.dataLoad()
data = classifier.dataStandardization(data)
print(data.shape)


our_regressor = lr.LinearRegression(X_train, y_train).fit()

# n_data_points, n_features = data.shape
# weights = np.zeros(n_features-1)
# initial_cost = classifier.calculateCost(data, weights)
# print("Initial cost is: ", initial_cost, "\n")

# n_iters = 1500
# learning_rate = 0.01
# (J_history, optimal_params) = classifier.gradient_descent(
#     data, weights, learning_rate, n_iters)
# print("Optimal parameters are: \n", optimal_params, "\n")
# print("Final cost is: ", J_history[-1])
# plt.plot(range(len(J_history)), J_history, 'r')

# plt.title("Convergence Graph of Cost Function")
# plt.xlabel("Number of Iterations")
# plt.ylabel("Cost")
# plt.show()
