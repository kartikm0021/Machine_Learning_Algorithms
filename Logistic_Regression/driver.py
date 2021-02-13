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
theta = classifier.stochasticGD(data, theta, 10, 1372*20)
print(theta)
print(theta.shape)
y_prediction_cls, accuracy = classifier.predict(data, theta)
print(accuracy)


data_file_name = 'data/data_banknote_authentication-copy.txt'
classifier = logistic.Logistic_Regression(data_file_name)
data = classifier.dataLoad(data_file_name)
data = classifier.dataNorm(data)

y_prediction_cls, accuracy = classifier.predict(data, theta)

print(accuracy)
print(y_prediction_cls)
print(data[:, -1])
