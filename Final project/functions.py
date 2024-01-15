def custom_error(estimator, X_test, y_test):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    y_pred = estimator.predict(X_test)
    return np.mean(np.diag(euclidean_distances(y_test, y_pred)))

def custom_error2(y_true, y_pred):
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances
    return np.mean(np.diag(euclidean_distances(y_true, y_pred)))

def custom_error3(estimator, X_test, y_test):
    from scipy.spatial.distance import euclidean
    import numpy as np
    y_pred = estimator.predict(X_test)
    return np.mean([euclidean(y_test_i, y_pred_i) for y_test_i, y_pred_i in zip(y_test, y_pred)])

def save_submission(n_submissions, rf_pred):
    import csv
    n_submissions = int(input("Ingrese el número de submission: "))
    with open(f'./Submissions/submission{n_submissions}_M.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Predicted'])
        for i, (x, y) in enumerate(rf_pred):
            writer.writerow([i, f"{x}|{y}"])
