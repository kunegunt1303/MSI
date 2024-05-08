Opis projektu
Ten projekt zawiera implementację niestandardowej regresji logistycznej oraz porównuje ją z dwoma popularnymi klasyfikatorami: k-NN (K Nearest Neighbors) oraz drzewem decyzyjnym (Decision Tree). Wyniki są oceniane za pomocą systemu walidacji krzyżowej RepeatedStratifiedKFold oraz testu statystycznego t-studenta. Cały kod jest napisany w języku Python.

Zawartość pliku
LogisticRegressionCustom: Niestandardowa implementacja regresji logistycznej.
kNN: Klasyfikator k-Nearest Neighbors z biblioteki sklearn.
DecisionTreeClassifier: Drzewo decyzyjne z biblioteki sklearn.
Analiza wyników: Test statystyczny t-studenta porównujący wyniki poszczególnych klasyfikatorów.

Wymagania:
Python 3.6+
Biblioteki:
numpy
pandas
scipy
scikit-learn

Pobierz dane:
Dane używane w projekcie to zestaw Breast Cancer Wisconsin Diagnostic Database (wdbc.data), który można znaleźć po wpisaniu w wyszukiwarkę:
UCI Machine Learning Repository

Umieść plik wdbc.data w folderze projektu i zaktualizuj zmienną data_file_path w kodzie, aby wskazywała na odpowiednią ścieżkę do pliku.

Uruchom program:
Uruchom poniższy skrypt, aby porównać wyniki klasyfikatorów:

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X.shape[1])

        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    def predict_prob(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

    def get_params(self, deep=True):
        return {'learning_rate': self.learning_rate, 'num_iterations': self.num_iterations}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

data_file_path = r'C:\Users\kingu\pythonProject\.venv\MSI\wdbc.data'

columns = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

data = pd.read_csv(data_file_path, header=None, names=columns)

X = data.drop(columns=['id', 'diagnosis'])
y = data['diagnosis'].replace({'M': 1, 'B': 0}).values

clfs = [
    LogisticRegressionCustom(learning_rate=0.01, num_iterations=10000),
    KNeighborsClassifier(),
    DecisionTreeClassifier(random_state=0)
]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)


results = np.zeros((10, len(clfs)))

for fold, (train, test) in enumerate(rskf.split(X, y)):
    for c_id, c in enumerate(clfs):
        clone_clf = clone(c)
        clone_clf.fit(X.iloc[train], y[train])
        preds = clone_clf.predict(X.iloc[test])
        acc = accuracy_score(y[test], preds)
        results[fold, c_id] = acc

t_stat_matrix = np.zeros((3, 3))
p_val_matrix = np.zeros((3, 3))
better_matrix = np.zeros((3, 3)).astype(bool)

for i in range(3):
    for j in range(3):
        res_i = results[:, i]
        res_j = results[:, j]

        t_stat, p_val = ttest_rel(res_i, res_j)
        t_stat_matrix[i, j] = t_stat
        p_val_matrix[i, j] = p_val

        better_matrix[i, j] = np.mean(res_i) > np.mean(res_j)

print('\nT-statistics matrix:')
print(t_stat_matrix)
print('\nP-values matrix:')
print(p_val_matrix)
print('\nBetter matrix:')
print(better_matrix)

alpha = 0.05
stat_significant = p_val_matrix < alpha
print('\nStatistically significant matrix:')
print(stat_significant)

stat_better = stat_significant * better_matrix
print('\nStatistically better matrix:')
print(stat_better)

classifiers = ['LR Custom', 'kNN', 'DT']

for i in range(3):
    for j in range(3):
        if i != j:
            if stat_better[i, j]:
                print('%s (acc=%0.3f) jest lepszy statystycznie od %s (acc=%0.3f)' %
                      (classifiers[i], np.mean(results[:, i]), classifiers[j], np.mean(results[:, j])))
            else:
                print('%s (acc=%0.3f) nie jest statystycznie lepszy od %s (acc=%0.3f)' %
                      (classifiers[i], np.mean(results[:, i]), classifiers[j], np.mean(results[:, j])))



Klasa `LogisticRegressionCustom` implementuje niestandardowy model regresji logistycznej:
- `__init__`: Inicjalizuje model z parametrami: `learning_rate` oraz `num_iterations`.
- `sigmoid`: Funkcja aktywująca Sigmoid.
- `fit`: Dopasowuje model do danych.
- `predict_prob`: Oblicza prawdopodobieństwa.
- `predict`: Przewiduje etykiety na podstawie progów prawdopodobieństwa.

Wyniki są prezentowane w macierzy t-statystyki, wartości p oraz macierzy istotnych statystycznie różnic między klasyfikatorami. Na końcu kodu znajduje się analiza wyników, która wyświetla, który klasyfikator jest statystycznie lepszy od innych.

