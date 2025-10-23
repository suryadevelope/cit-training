# Experiment 6 code - Random Forest examples
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

wine = datasets.load_wine()
diabetes = datasets.load_diabetes()
cancer = datasets.load_breast_cancer()

# Wine
X, y = wine.data, wine.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4)
dt = DecisionTreeClassifier(random_state=0); dt.fit(Xtr, ytr)
print('Wine - Decision Tree acc:', dt.score(Xte, yte))
rfc = RandomForestClassifier(n_estimators=100, random_state=0); rfc.fit(Xtr, ytr)
print('Wine - RandomForest acc:', rfc.score(Xte, yte))
imp = rfc.feature_importances_
top = np.argsort(imp)[-5:][::-1]
print('Top features (Wine):')
for i in top:
    print(wine.feature_names[i], imp[i])

# Diabetes regression
X, y = diabetes.data, diabetes.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4)
rfr = RandomForestRegressor(n_estimators=100, random_state=0); rfr.fit(Xtr, ytr)
print('Diabetes - RandomForest MSE:', mean_squared_error(yte, rfr.predict(Xte)))

# Breast Cancer
X, y = cancer.data, cancer.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=4)
rfc2 = RandomForestClassifier(n_estimators=100, random_state=0); rfc2.fit(Xtr, ytr)
print('Breast Cancer - RandomForest acc:', rfc2.score(Xte, yte))
