import Logistic_Regression as logistic

data_file_name = 'data/data_banknote_authentication.txt'

classifier = logistic.Logistic_Regression(data_file_name)

data = classifier.dataLoad(data_file_name)
data = classifier.dataNorm(data)
classifier.printMeanAndSum(data)
