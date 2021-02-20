import logistic_reg as logistic
import numpy as np
from sklearn.metrics import roc_curve
from matplotlib import pyplot


data_file_name = 'data/data_banknote_authentication.txt'
classifier = logistic.Logistic_Regression(data_file_name)

base_data = classifier.dataLoad(data_file_name)
print(f'Data shape {base_data.shape}')
data = classifier.dataNorm(base_data)
classifier.printMeanAndSum(data)

theta = np.zeros((data.shape[1]-1, 1))
error = classifier.errCompute(data, theta)
print(f'Test Error is {error}')

theta = classifier.stochasticGD(
    data, theta, 0.01, 1372*20)
y_prediction_cls, accuracy = classifier.predict(data, theta)
print(accuracy)
print(theta)

print('Starting k fold cross validation')
classifier.trigger_k_fold_cross_validation(base_data)
