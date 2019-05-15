import numpy
import pandas
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
import sys

# before scaling
dataframe2= pandas.read_csv('C:/Users/Kanverse/Documents/train1.csv', header=0)
dataframe3= pandas.read_csv("C:/Users/Kanverse/Documents/validation.csv", header=0)
# put each unscaled dataset in a dataframe

array2 = dataframe2.values
array3 = dataframe3.values

#set the x-values for training set and validation set
X1 = array2[:, 0:5]
X2 = array3[:, 0:5]

# set the y-values for training set and validatio set
Y1 = array2[:,-1]
Y2 = array3[:,-1]

# set and scale using MinMaxscaler alogorithm
scaler = preprocessing.MinMaxScaler().fit(X1)
scaler = preprocessing.MinMaxScaler().fit(X2)

# initializing the scaled X values (we assigned the scaled values to arbitrarily name variables)
rescaledX1 = scaler.fit_transform(X1)
rescaledX2 = scaler.fit_transform(X2)

# We merge the y values with their respective scaled X values
Z1 = numpy.append(rescaledX1, Y1[:, None], axis=1)
Z2 = numpy.append(rescaledX2, Y2[:, None], axis=1)

# we save the scaled datasets to a desired location
numpy.savetxt("C:/Users/Kanverse/Documents/train1_scaled.csv", Z1, delimiter=",")
numpy.savetxt("C:/Users/Kanverse/Documents/validation1_scaled.csv", Z2, delimiter=",")

numpy.set_printoptions(precision=4)
print(Z1)