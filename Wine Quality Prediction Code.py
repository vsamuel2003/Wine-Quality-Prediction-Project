import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("winequality-red.csv", sep = ';')
X = data[['volatile acidity', 'sulphates', 'alcohol']]
y = data['quality']

# converting y into good and bad binary options
y_copy = []
for i in range(len(y)):
    if y[i] >= 7:
        y_copy.append(1)
    else:
        y_copy.append(0)


sel = feature_selection.SelectKBest(feature_selection.f_classif, k=3)
X_new = sel.fit_transform(test, y_copy)

kneighbors = KNeighborsClassifier(n_neighbors = 13)
print(model_selection.cross_val_score(kneighbors, X, y_copy, scoring = 'accuracy', cv = 10).mean())

forest = RandomForestClassifier(n_estimators= 100,
                               criterion= 'entropy')
print(model_selection.cross_val_score(forest, X, y_copy, scoring = 'accuracy', cv = 10).mean())

support_vector = SVC(kernel = 'linear')
print(model_selection.cross_val_score(support_vector, X, y_copy, scoring = 'accuracy', cv = 10).mean())

log = LogisticRegression()
print(model_selection.cross_val_score(log, X, y_copy, scoring = 'accuracy', cv = 10).mean())

# Normalize feature variables
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_copy, test_size = 0.2)

forest.fit(X_train, y_train)
acc = forest.score(X_test, y_test)
y_pred = forest.predict(X_test)
print(classification_report(y_test, y_pred))