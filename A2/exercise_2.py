import numpy
import linweighreg as linreg
import matplotlib.pyplot as plt

# Load the data
raw = numpy.genfromtxt('men-olympics-100.txt', delimiter=' ')

t = raw[:, 1]
t = t.reshape((len(raw), 1))

lambdaVals = numpy.logspace(-8, 0, 100, base=10)

def loocv(X, t, lambdaVal):
    loss = 0

    for i in range(len(X)):
        # Remove the i-th element from the data
        X_train = numpy.delete(X, i, axis=0)
        t_train = numpy.delete(t, i, axis=0)

        # Use the i-th element as test (leave one out cross validation)
        X_test = X[i].reshape(1, -1)
        t_test = t[i]

        # Fit model, predict and add the loss
        model = linreg.LinearRegression(lambdaVal=lambdaVal)
        model.fit(X_train, t_train)
        predictions = model.predict(X_test)
        loss += (predictions - t_test) ** 2
    
    # Return the average loss for the given lambda
    return (loss / len(X)).flatten()

    


print("=== First order polynomial ===")
X = raw[:, 0]
X = X.reshape((len(raw), 1))

# Calculate the loss for each lambda
results = numpy.array([loocv(X, t, lambdaVal) for lambdaVal in lambdaVals])

model_zero = linreg.LinearRegression(lambdaVal = 0)
model_zero.fit(X, t)
print("Calculated weights for lambda=0: %s" % model_zero.w)

bestLambda = lambdaVals[numpy.argmin(results)]
model_bestlambda = linreg.LinearRegression(lambdaVal = bestLambda)
model_bestlambda.fit(X, t)
print("Best lambda: %.10f with loss: %.10f" % (bestLambda, numpy.min(results)))
print("Calculated weights for lambda=%.10f: %s" % (bestLambda, model_bestlambda.w))

plt.plot(lambdaVals, results)
plt.xlabel("Lambda")
plt.ylabel("LOOCV error")
plt.title("LOOCV error with different lambdas for first order polynomial")
plt.savefig("LOOCV_error_firstorder.png")

plt.show()

print("=== Fourth order polynomial ===")
X = raw[:, 0]
X = X.reshape((len(raw), 1))
X = numpy.concatenate((X, X ** 2, X ** 3, X ** 4), axis=1)
print(X[0])

# Calculate the loss for each lambda
results = numpy.array([loocv(X, t, lambdaVal) for lambdaVal in lambdaVals])

model_zero = linreg.LinearRegression(lambdaVal = 0)
model_zero.fit(X, t)
print("Calculated weights for lambda=0: %s" % model_zero.w)

bestLambda = lambdaVals[numpy.argmin(results)]
model_bestlambda = linreg.LinearRegression(lambdaVal = bestLambda)
model_bestlambda.fit(X, t)
print("Best lambda: %.10f with loss: %.10f" % (bestLambda, numpy.min(results)))
print("Calculated weights for lambda=%.10f: %s" % (bestLambda, model_bestlambda.w))

plt.plot(lambdaVals, results)
plt.xlabel("Lambda")
plt.ylabel("LOOCV error")
plt.title("LOOCV error with different lambdas for fourth order polynomial")
plt.savefig("LOOCV_error_fourthorder.png")

plt.show()
