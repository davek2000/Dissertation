from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.calibration import CalibratedClassifierCV as calib_clf

import pandas as pd

from matplotlib import pyplot as plt
# Import dummy data with 10 features
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

# x is input data
# y is labelled data of either 0 OR 1 
x, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_redundant=10,
                           n_informative=10,
                           random_state=42,
                           n_clusters_per_class=3,
                           n_classes=3)
# Load data
#data = pd.read_csv()

# Check if data has any missing values
#data.isnull().sum()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Create the calibrated models for Decision Tree and Random Forest
# classifiers = [calib_clf(base_estimator=SVC(kernel="rbf", probability=True), method="sigmoid", cv=5),
#                calib_clf(base_estimator=SVC(kernel="rbf", probability=True), method="isotonic", cv=5),
#                calib_clf(base_estimator=dt(), method="sigmoid", cv=5),
#                calib_clf(base_estimator=dt(), method="isotonic", cv=5),
#                calib_clf(base_estimator=rf(), method="sigmoid", cv=5),
#                calib_clf(base_estimator=rf(), method="isotonic", cv=5)]
classifier = calib_clf(base_estimator=SVC(kernel="linear", probability=True), method="sigmoid", cv=5)
print("y-train: ",y_train.shape)
classifier.fit(x_train, y_train)
print(classifier.score(x_test,y_test))

print(classifier.predict_proba(x_train))

print(classifier.calibrated_classifiers_)

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    #plt.show()

features_names = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
coef_avg = 0
for i in classifier.calibrated_classifiers_:
    coef_avg = coef_avg + i.base_estimator.coef_
coef_avg = coef_avg/len(classifier.calibrated_classifiers_)

f_importances(coef_avg[0], features_names)
