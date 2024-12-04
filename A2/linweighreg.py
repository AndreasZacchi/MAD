import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self, alpha=None, lambdaVal=0):
        self.alpha = alpha
        self.lambdaVal = lambdaVal
        pass
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        alpha : Array of shape [n_samples, 1], optional
        """        

        # Ensure the arrays are N-dimensional numpy arrays
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # Add a column at the beginning of the feature matrix
        oneCol = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((oneCol, X), axis=1)

        # If no alpha is provided use the identity matrix to not affect the result
        if self.alpha is None:
            A = numpy.identity(X.shape[0])
        # Otherwise use it
        else:
            self.alpha = numpy.array(self.alpha).reshape((len(self.alpha), 1))
            A = numpy.diag(self.alpha.flatten())

        diagLambda = self.lambdaVal * numpy.identity(X.shape[1])

        # calculate the weights using the formula from the lecture
        firstPart = ((X.T @ A) @ X) + diagLambda
        secondPart = (X.T @ A) @ t
        #self.w = numpy.linalg.inv(firstPart) @ secondPart
        self.w = numpy.linalg.solve(firstPart, secondPart)

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     

        # Ensure the array is a N-dimensional numpy array
        X = numpy.array(X).reshape((len(X), -1))

        # Add a column at the beginning of the feature matrix
        oneCol = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((oneCol, X), axis=1)

        # calculate the predictions
        predictions = X @ self.w

        return predictions
