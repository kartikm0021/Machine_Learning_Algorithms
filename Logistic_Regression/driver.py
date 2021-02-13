import Logistic_Regression as logistic
import numpy as np

data_file_name = 'data/data_banknote_authentication.txt'

classifier = logistic.Logistic_Regression(data_file_name)

data = classifier.dataLoad(data_file_name)
data = classifier.dataNorm(data)
classifier.printMeanAndSum(data)
print(data[:7, :])

theta = np.zeros((data.shape[1]-1))
temp = classifier.errCompute(data, theta)
print('Kartik')
print(data.shape)
theta = classifier.stochasticGD(data, theta, 0.02, 1372*20)
print(theta)
y_prediction_cls, accuracy = classifier.predict(data)
print(accuracy)
