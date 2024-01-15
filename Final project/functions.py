def custom_error(estimator, X_test, y_test):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    y_pred = estimator.predict(X_test)
    return np.mean(np.diag(euclidean_distances(y_test, y_pred)))

def custom_error2(y_true, y_pred):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    return np.mean(np.diag(euclidean_distances(y_true, y_pred)))