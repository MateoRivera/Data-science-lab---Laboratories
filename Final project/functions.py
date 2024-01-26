# Custom error function and submission saving function
def mean_euclidean_distance_error(estimator, X_test, y_test):
    from scipy.spatial.distance import euclidean
    import numpy as np
    y_pred = estimator.predict(X_test)
    euclidean_distances = [euclidean(y_test_i, y_pred_i) for y_test_i, y_pred_i in zip(y_test, y_pred)]
    return np.mean(euclidean_distances), np.std(euclidean_distances)

def save_submission(path, y_pred):
    import csv
    n_submissions = int(input("Ingrese el número de submission: "))
    author = input("Ingrese su nombre: ").upper()
    with open(f'./Submissions/submission{n_submissions}_{author[0]}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Predicted'])
        for i, (x, y) in enumerate(y_pred):
            writer.writerow([i, f"{x}|{y}"])

class RandomizedSearchHO():
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None):
        self.__estimator = estimator
        self.__param_distributions = param_distributions
        self.__n_iter = n_iter
        self.__scoring = scoring
        self.best_params_ = None
        self.best_estimator_ = None
        self._best_score = None
        self.ho_results = {param: [] for param in self.__param_distributions.keys()}
        self.ho_results['score'] = []

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.model_selection import train_test_split
        import numpy as np
        from itertools import product

        # Let's split X, y into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=8)

        # Let's sample the param_grid
        samples_param_grid = np.random.choice([{param: param_value  for param, param_value in zip(self.__param_distributions.keys(), param_values)} for param_values in product(*self.__param_distributions.values())], self.__n_iter, replace=False)

        # Let's train the models
        for i, param_sample in enumerate(samples_param_grid):
            print(f"Training model {i+1}/{self.__n_iter}")
            # We clone the estimator because we want to keep the best one
            estimator = clone(self.__estimator)

            # We set the parameters of the estimator and we fit it
            estimator.set_params(**param_sample)
            estimator.fit(X_train, y_train)

            # We evaluate the estimator
            current_score = self.__scoring(estimator, X_val, y_val)

            # We save the results            
            for param in param_sample:
                self.ho_results[param].append(param_sample[param])
            self.ho_results['score'].append(current_score)

            # We update the best estimator
            if i == 0:
                self.best_params_ = param_sample
                self.best_estimator_ = estimator
                self._best_score = current_score
            else:
                if current_score < self._best_score:
                    self.best_params_ = param_sample
                    self.best_estimator_ = estimator
                    self._best_score = current_score

            print(f"Hyperparameters: {param_sample}")
            print(f"Score: {current_score}")
            print("--------------------------------------------------"*50)
            
        return self
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
class SFSExtractor(object):
    def __init__(self, features_selected):
        self.selected_features = features_selected
    def transform(self, X):
        return X.loc[:, self.selected_features]
    
    def fit(self, X, y=None):
        return self
