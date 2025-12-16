import numpy as np
from sklearn.ensemble import RandomForestRegressor
from skada import CORALAdapter, make_da_pipeline


class Model:
    def __init__(self):
        """ Initialize the model.
        """
        self.regressor = RandomForestRegressor(
            n_estimators=20,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )
        
        self.coral_adapter = CORALAdapter()
        
        self.pipe = make_da_pipeline(
            self.coral_adapter,
            self.regressor
        )

    def fit(self, X, y, X_adapt):
        """ Train the model.

        Args:
            X: Training data matrix of shape (num-samples, num-features), type
            np.ndarray.
            y: Training label vector of shape (num-samples), type np.ndarray.
            X_adapt: DA training data matrix of shape (num-samples-DA, num-features),
            type np.ndarray.
        """
        X_train = np.concatenate([X, X_adapt], axis=0)
        y_train = np.concatenate([
        	y.astype(float).ravel(),
        	np.full(X_adapt.shape[0], np.nan)
        ])
        sample_domain = np.concatenate([np.zeros(len(X)), -np.ones(len(X_adapt))])
        sample_domain = sample_domain.astype(int)
        self.pipe.fit(X_train, y_train, sample_domain=sample_domain)   

    def predict(self, X):
        """ Predict the labels.

        Args:
          X: Test data matrix of shape (num-samples, num-features) to pass to the
          model for inference, type np.ndarray.
        """
        y = self.pipe.predict(X)
        return y
