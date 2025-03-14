import numpy

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """        

        # Ensure the arrays are N-dimensional numpy arrays
        X = numpy.array(X).reshape((len(X), -1))
        t = numpy.array(t).reshape((len(t), 1))

        # Add a column at the beginning of the feature matrix
        oneCol = numpy.ones((X.shape[0], 1))
        X = numpy.concatenate((oneCol, X), axis=1)

        # calculate the weights using the formula from the lecture
        firstPart = X.T @ X
        secondPart = X.T @ t
        self.w = numpy.linalg.inv(firstPart) @ secondPart

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
