import sklearn
from sklearn.utils import shuffle
from sklearn import datasets
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Data file import
test_data = pd.read_csv("/Users/sarahannaford/Desktop/UC Folders/Software Tech/Capstone Project/exams.csv")


# Attribute to be predicted
predict = "pass_npass"

# Pre-processing the data 
le = preprocessing.LabelEncoder()
gender = le.fit_transform(list(test_data["gender"])) # Gender (0.0 = Female, 1.0 = Male)
race_ethnicity = le.fit_transform(list(test_data["race_ethnicity"])) # Race/ethnicity of student (0.00 = Group A, 0.25 = Group B, 0.50 = Group C, 0.75 = Group D, 1.00 = Group A)
parent_education = le.fit_transform(list(test_data["parent_level_of_education"])) # Level of parental education (0.0 = Associate's degree, 0.4 = High school, 0.8 = Some college)
lunch = le.fit_transform(list(test_data["lunch"])) # Lunch (1.0 = standard, 0.0 = free/reduced)
test_prep_course = le.fit_transform(list(test_data["test_prep_course"])) # Test preperation course (0.0 = completed, 0.1 = none)
math_score = le.fit_transform(list(test_data["math_score"]))
reading_score = le.fit_transform(list(test_data["reading_score"]))
writing_score = le.fit_transform(list(test_data["writing_score"]))
average_score = le.fit_transform(list(test_data["average_score"]))

pass_npass = le.fit_transform(list(test_data["pass_npass"])) # Pass (1.0 = Pass, 0.0 = No Pass)

# Predictive analytics model development by comparing different Scikit-learn classification algorithms
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

x = list(zip(gender, race_ethnicity, parent_education, lunch, test_prep_course, math_score, writing_score, 
             average_score))
y = list(pass_npass)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows
#size of train and test subsets after splitting
np.shape(x_train), np.shape(x_test)

models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
print("Performance on Training set")
for name, model in models:
  kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  msg += '\n'
  print(msg)

# Compare Algorithms' Performance
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Model Evaluation by testing with independent/external test data set.
# Make predictions on validation/test dataset
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
svm = SVC()

best_model = gb
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score (y_test, y_pred)
print("Best Model Accuracy Score on Test Set:", model_accuracy)

#Model Performance Evaluation Metric 1 - Classification Report
print(classification_report(y_test, y_pred))

#Model Performance Evaluation Metric 2
#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Model Evaluation Metric 3- ROC-AUC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
best_model = gb
best_model.fit(x_train, y_train)
gb_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label = 'Random Forest(area = %0.2f)'% gb_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

#Model Evaluation Metric 4-prediction report
for x in range(len(y_pred)):
  print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
