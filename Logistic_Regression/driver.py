import Logistic_Regression as logistic
import numpy as np

# data_file_name = 'data/data_banknote_authentication.txt'
data_file_name = 'supporting_files/shuffled.data'

classifier = logistic.Logistic_Regression(data_file_name)

base_data = classifier.dataLoad(data_file_name)
data = classifier.dataNorm(base_data)
classifier.printMeanAndSum(data)
theta = np.zeros((data.shape[1]-1, 1))

# theta = np.zeros((data.shape[1]-1))
error = classifier.errCompute(data, theta)
print(f'Error is {error}')
theta = classifier.stochasticGD(data, theta, 0.01, 1372*20)
y_prediction_cls, accuracy = classifier.predict(data, theta)
print(accuracy)


# K Fold cross validation
print('Starting k fold cross validation')
classifier.trigger_k_fold_cross_validation(base_data)

# data_file_name = 'data/data_banknote_authentication-copy.txt'
# classifier = logistic.Logistic_Regression(data_file_name)
# data = classifier.dataLoad(data_file_name)
# data = classifier.dataNorm(data)

# y_prediction_cls, accuracy = classifier.predict(data, theta)

# print(accuracy)
# print(y_prediction_cls)
# print(data[:, -1])


# data_file_name = 'supporting_files/shuffled.data'
# classifier = logistic.Logistic_Regression(data_file_name)
# data = classifier.dataLoad(data_file_name)
# X_shufnorm = classifier.dataNorm(data)
# theta = classifier.stochasticGD(X_shufnorm, np.zeros(
#     (X_shufnorm.shape[1]-1, 1)), 0.01, 1372*20)
# temp = classifier.errCompute(X_shufnorm, theta)
# print(temp)
