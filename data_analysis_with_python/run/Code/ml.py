from sklearn.svm import SVR
from sklearn import linear_model
from traits.api import HasTraits


LEARNING_METHODS = ['regression', 'svr']

# Learning models
#----------------------------------------------------------


class WeatherPredictor(HasTraits):
    def __init__(self, wstore):
        self._ws = wstore
        self._learner_map = {
        'regression': self.regression,
        'svr': self.svr
        }

    def regression(self, X, y):
        """ Train linear regression model

        Parameters
        ----------
        X : numpy ndarray with numeric values
            Array containing input parameters
            for the model. Model will try to
            learn the output y[i] in terms of
            inputs X[i]

        y : columnar numpy array with numeric values
            Array containing single column of
            output values. Entry at y[i] corresponds
            to value of the underlying experiment
            for input parameters X[i]

        Returns
        -------
        result : model
                Model learnt from incoming input
                inputs and outputs

        """
        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        return regr

    def svr(self, X, y):
        """ Train support vector regression model

        Parameters
        ----------
        X : numpy ndarray with numeric values
            Array containing input parameters
            for the model. Model will try to
            learn the output y[i] in terms of
            inputs X[i]

        y : columnar numpy array with numeric values
            Array containing single column of
            output values. Entry at y[i] corresponds
            to value of the underlying experiment
            for input parameters X[i]

        Returns
        -------
        result : model
                Model learnt from incoming input
                inputs and outputs

        """
        clf = SVR(C=1.0, epsilon=0.2)
        clf.fit(X, y)
        return clf

    def test_learning(self, learning_method, city,
        field, learn_idx=1600):
        """ Build a model for specified weather field.
        The model learns from a subset of the
        weather data for the specified city. This function
        then uses a non overlapping subset of the weather
        data of the same city as a test set.

        Parameters
        ----------
        learning_method : string
                Use learning_method as the technique
                for learning. Should be a key in
                self._learner_map

        city : string
                City for which we would like to learn
                from weather data

        field : string
                Name of weather field we would like to
                learn

        learn_idx : integer
                This function will learn from the first
                learn_idx items of weather data. Remaining
                items (total_items - learn_idx) will be used
                as the test set

        Returns
        -------
        pred : ndarray of numeric values
                Values predicted by learnt model for test set
        y : ndarray of numeric values
                Actual values of weather field under study for
                the test set

        """
        X, y = self._ws.learning_data(city, field)
        learning_fn = self._learner_map[learning_method]
        model = learning_fn(X[:learn_idx], y[:learn_idx])
        pred = model.predict(X[learn_idx:])
        return pred, y[learn_idx:]

    def cross_learn(self, learning_method, city_one,
        city_two, field):
        """ Learn specified weather field from data for
        city_one. Use learnt model to predict weather field
        for city_two.

        Parameters
        ----------
        learning_method : string
                Use learning_method as the technique
                for learning. Should be a key in
                self._learner_map

        city_one : string
                City for which we would like to learn
                from weather data

        city_two : string
                City for which we would like to predict
                weather data

        field : string
                Weather field under study

        Returns
        -------
        pred : ndarray of numeric values
                Values predicted by learnt model city_two
        y : ndarray of numeric values
                Actual values of weather field under study for
                city_two

        """
        X, y = self._ws.learning_data(city_one, field)
        learning_fn = self._learner_map[learning_method]
        model = learning_fn(X, y)
        X, y = self._ws.learning_data(city_two, field)
        pred = model.predict(X)
        return pred, y
