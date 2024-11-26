import numpy
import linweighreg as linreg
import matplotlib.pyplot as plt

# Load the data
raw = numpy.genfromtxt('men-olympics-100.txt', delimiter=' ')

t = raw[:, 1]
t = t.reshape((len(raw), 1))

lambdaVals = numpy.logspace(-8, 0, 100, base=10)


print("=== First order polynomial ===")
X = raw[:, 0]
X = X.reshape((len(raw), 1))

model_zero = linreg.LinearRegression(lambdaVal = 0)
model_zero.fit(X, t)
print("Calculated weights for lambda=0: %s", str(model_zero.w))

