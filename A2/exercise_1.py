import numpy
import pandas
import linweighreg as linreg
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

# (b) fit linear regression model using all features
model_all = linreg.LinearRegression((t_train ** 2))
model_all.fit(X_train, t_train)
print("Weights: %s" % model_all.w)

def rmse(t, tp):
    t = t.reshape((len(t), 1))
    tp = tp.reshape((len(tp), 1))
    return numpy.sqrt(numpy.mean((t - tp) ** 2))

# evaluate on test data
# all features
pred_all = model_all.predict(X_test)
print("RMSE using the model (all features): %f" % rmse(pred_all, t_test))
rmse_all = rmse(pred_all, t_test)
plt.figure()
plt.scatter(t_test, pred_all)
plt.ylim(0, 60)
plt.xlim(0, 60)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("All features - RMSE: %f" % rmse_all)

plt.savefig("1b.png")
plt.show()

