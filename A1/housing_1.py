import numpy
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
mean_price = numpy.mean(t_train)
print("The mean price of the houses is: %f" % mean_price)

# (b) RMSE function
def rmse(t, tp):
    t = t.reshape((len(t), 1))
    tp = tp.reshape((len(tp), 1))
    return numpy.sqrt(numpy.mean((t - tp) ** 2))

# Create a list of the predicted prices
prediction = mean_price * numpy.ones(len(t_test))
print("RMSE using the model (mean): %f" % rmse(prediction, t_test))

# (c) visualization of results
plt.figure()
plt.scatter(t_test, prediction)
plt.ylim(0, 60)
plt.xlim(0, 60)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Mean price - RMSE: %f" % rmse(prediction, t_test))

plt.savefig("housing_1.png")
plt.show()