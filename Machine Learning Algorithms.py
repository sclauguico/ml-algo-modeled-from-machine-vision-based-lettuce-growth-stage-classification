# To import libraries
import numpy as np # for computation
import pandas as pd # for data manipulation
import matplotlib.pyplot as plt # for visualization

# To import dataset
dataset = pd.read_csv('Lettuce_Balanced.csv')

# To create the matrix of independent variable, x
X = dataset.iloc[:,2:16].values

# To create the matrix of dependent variable, y
Y = dataset.iloc[:,0].values

# To handle the missing data
# A. To know how much of the data is missing
dataset.isnull().sum().sort_values(ascending=False)

# To encode the categorical data
from sklearn.preprocessing import LabelEncoder

# B. To encode the categorical data if existing in the dependent variable, Y
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

# To split the whole dataset into training dataset and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0) #train_size=0.8, you can either still put this or not since test_size is already defined. By default, remaining is for training

# To perform feature scaling 
# A. For standardization feature scaling
from sklearn.preprocessing import StandardScaler # for not normally distributed samples
standard_scaler = StandardScaler ()
X_train_standard = X_train.copy()
X_test_standard = X_test.copy()
X_train_standard = standard_scaler.fit_transform(X_train_standard) # X_train_standard[:,3:5] -> for specifying features to be scaled 
X_test_standard = standard_scaler.fit_transform(X_test_standard)  # X_test_standard[:,3:5] -> for specifying features to be scaled 

# To fit the training dataset into a multiple linear regression model
from sklearn.linear_model import LogisticRegression 
logistic_regression = LogisticRegression(random_state=0)

logistic_regression.fit(X_train_standard,Y_train)

# To predict the output of the testing dataset
Y_predict_logreg = logistic_regression.predict_proba(X_test_standard) #predict proba is in terms of probability kaya continuous ang output. Tanggalin dapat para categorical
Y_predict_logreg = logistic_regression.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_logreg)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=logistic_regression, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=logistic_regression, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To evaluate the performance of the logistic regression model using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_logreg)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (If looking for positive results, how frequent it attains postive results, should have same performance of predicting for + and - to avoid bias in one category)
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (If looking for negative results, how frequent it attains negative results, counterpart of sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# H. For the Classification Report
from sklearn.metrics import classification_report
log_reg_classification_report = classification_report(Y_test, Y_predict_logreg)

#####################################LOGREG OPTIMIZATION#######################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized Under the Logistic Regression Model
parameters = [{'C': [0.001, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver':['liblinear', 'saga']},
              {'C':[0.001, 0.1, 1, 10, 100], 'penalty': ['none', 'l2'], 'solver':['newton-cg','lbfgs', 'saga']}]
                                                                                                                                                 
grid_search = GridSearchCV(estimator = logistic_regression,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

# To Identify the Best Accuracy and Best Features

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("BEST ACCURACY SCORE:")
print(best_accuracy)
print('')

print("BEST PARAMETERS:")
print(best_parameters)

# To Instantiate the Model (Using the Optimized Parameters)
logistic_regression = LogisticRegression(C=100, penalty='l1', solver='liblinear', random_state=0)

# To Fit the Training Dataset into Logistic Regression Model
logistic_regression.fit(X_train_standard, Y_train)

# To Predict the Output of the Testing Dataset
Y_predict_logreg = logistic_regression.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_logreg)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=logistic_regression, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

F1 = cross_val_score(estimator=logistic_regression, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To evaluate the performance of the logistic regression model using holdout
    # A. For the Classification Accuracy
    from sklearn.metrics import accuracy_score
    classification_accuracy = accuracy_score(Y_test, Y_predict_logreg)
    print('Classification Accuracy: %.4f'
          % classification_accuracy)
    print(' ')
    
    # B. For the Classification Error
    from sklearn.metrics import accuracy_score
    classification_error = 1-classification_accuracy
    print('Classification Error: %.4f'
          % classification_error)
    print(' ')
    
    # C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
    # True Positive Rate:  Actual Value +, How often Correct
    from sklearn.metrics import recall_score
    sensitivity = recall_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
    print('Sensitivity or Recall Score: %.4f'
          % sensitivity)
    print(' ')
    
    # D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
    # True Negative Rate: Actual Value -, How often Correct
    specificity = TN/(TN+FP)
    print('Specificity: %.4f'
          % specificity)
    print(' ') 
    
    # yung result bias kay negative compared to positive
    
    # E. For the FP rate .
    # False Positive Rate: Actual Value -, How often Inorrect
    false_positve_rate = 1-specificity
    print('False Positive Rate: %.4f'
          % false_positve_rate)
    print(' ')  
    
    # F. For the precision.
    # False Negative Rate: Predicted Value +, How often the prediction is Correct
    from sklearn.metrics import precision_score
    precision = precision_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
    print('Precision: %.4f'
          % precision)
    print(' ')  
    
    #Wag ma amaze sa accuracy 
    
    # G. For the F1 score. Relating precision and sensitivity
    # False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
    from sklearn.metrics import f1_score
    f1_score = f1_score(Y_test, Y_predict_logreg, pos_label = 'positive', average = 'weighted')
    print('F1 Score: %.4f'
          % f1_score)
    print(' ')  

# H. For classification report
from sklearn.metrics import classification_report
log_reg_optimized_classification_report = classification_report(Y_test, Y_predict_logreg)

#####################################KNN######################################################################

# To fit the training dataset into a KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_standard, Y_train)

# To predict the output of the testing dataset
Y_predict_KNN = knn.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_KNN )

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import StratifiedKFold # mas maganda kung StratifiedFold para well represented si 1 and 0
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=knn, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

F1 = cross_val_score(estimator=knn, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To evaluate the performance of the logistic regression model using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_KNN)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

#H. For the Classification Report
from sklearn.metrics import classification_report
knn_classification_report = classification_report(Y_test, Y_predict_KNN)

############################ KNN OPTIMIZED #################################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) 

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized Under the KNN Model
k_range = list(range(1,51))
weight = ['uniform','distance']
algorithm = ['auto','ball_tree', 'kd_tree', 'brute']
leaf_size = [10, 20, 30, 40 ,50, 60, 70, 80, 90, 100]
parameters = dict(n_neighbors=k_range, weights=weight, algorithm=algorithm, leaf_size=leaf_size)                              
grid_search = GridSearchCV(estimator = knn,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy =grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

# To I
knn = KNeighborsClassifier(n_neighbors=24, weights='distance', algorithm='auto', leaf_size = 10)

# To Fit the Whole Dataset into Logistic Regression Model
knn.fit(X_train_standard,Y_train)

# To Predict the Output of the Whole Dataset
Y_predict_KNN = knn.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_KNN)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

from sklearn.model_selection import StratifiedKFold # mas maganda kung StratifiedFold para well represented si 1 and 0
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=knn, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=knn, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_KNN)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_KNN, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# For the Classification Report
from sklearn.metrics import classification_report
knn_optimized_classification_report = classification_report(Y_test, Y_predict_KNN)

############################################# SVM ##################################################

# To fit the training dataset into a SVM model
from sklearn.svm import SVC
svm = SVC(random_state=0)
svm.fit(X_train_standard, Y_train)

# To predict the output of the testing dataset
Y_predict_SVM = svm.predict(X_test_standard)

# To show the confusion matrix.
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_SVM)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]

# To apply K-fold cross-validation for the logistic regression model
from sklearn.model_selection import StratifiedKFold # mas maganda kung StratifiedFold para well represented si 1 and 0
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=svm, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

F1 = cross_val_score(estimator=svm, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# To evaluate the performance of the logistic regression model using holdout
# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_SVM)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

#H. For the Classification Report
from sklearn.metrics import classification_report
svm_classification_report = classification_report(Y_test, Y_predict_SVM)

################################# SVM OPTIMIZED #################################

# To Import the kFold Class
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# To Import the GridSearch Class
from sklearn.model_selection import GridSearchCV

# To Set Parameters to be Optimized Under the Support Vector Machine Model
C = [0.001, 0.1, 1, 10, 100]
kernel = ['linear','poly','rbf','sigmoid'] #['linear','poly','rbf', 'sigmoid', 'precomputed']
degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gamma = [0.001, 0.1, 1, 10, 100]
#shrinking = [0, 1]
#probability = [0, 1]
#verbose = [0, 1]
#decision_function_shape = ['ovo', 'ovr']
#random_state = [0,1]
parameters = dict(C=C, kernel=kernel, degree=degree, gamma=gamma) #decision_function_shape=decision_function_shape)
                #shrinking=shrinking, verbose=verbose, probability=probability,
                #random_state=random_state)                              
grid_search = GridSearchCV(estimator = svm,
              param_grid = parameters,
              scoring = 'accuracy',
              cv = k_fold,
              n_jobs = -1)
grid_search = grid_search.fit(X,Y)
print(grid_search)

# To View the Results of the GridSearch
pd.DataFrame(grid_search.cv_results_)[['mean_test_score','std_test_score','params']]

# To Identify the Best Accuracy and Best Parameters
best_accuracy=grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy Score:")
print(best_accuracy)
print('')

print("Best Parameters Score:")
print(best_parameters)
print('')

# To Instantiate the Model (Using the Optimized Parameters)
svm = SVC(C=0.001, kernel='poly', degree=4, gamma=10)
                #shrinking=False, verbose=False, probability=False,


# To Fit the Training Dataset into Support Vector Machine Model
svm.fit(X_train_standard,Y_train)

# To Predict the Output of the Training Dataset
Y_predict_SVM = svm.predict(X_test_standard)

# To Show the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_predict_SVM)

import seaborn as sns
sns.heatmap(confusion_matrix, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')

TP = confusion_matrix[1,1]
TN = confusion_matrix[0,0]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]


from sklearn.model_selection import StratifiedKFold # mas maganda kung StratifiedFold para well represented si 1 and 0
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # shuffle = True para sa small dataset

from sklearn.model_selection import cross_val_score

# For the accuracy as scoring for for cross-validation 
accuracies = cross_val_score(estimator=svm, X=X_train_standard, y=Y_train, cv=k_fold, scoring='accuracy')

accuracies_average = accuracies.mean()
accuracies_variance = accuracies.std()

print('Accuracy of K-FOLDS:')
print (accuracies)
print(' ')
print('Average Accuracy of K-FOLDS:')
print(accuracies_average)
print(' ')
print('Accuracy Variance of K-FOLDS:')
print(accuracies_variance)
print(' ')

# For the F1 as scoring for for cross-validation 
F1 = cross_val_score(estimator=knn, X=X_train_standard, y=Y_train, cv=k_fold, scoring='f1_weighted')

F1_average = F1.mean()
F1_variance = F1.std()

print('F1 of K-FOLDS:')
print (F1)
print(' ')
print('Average F1 of K-FOLDS:')
print(F1_average)
print(' ')
print('F1 Variance of K-FOLDS:')
print(F1_variance)
print(' ')

# A. For the Classification Accuracy
from sklearn.metrics import accuracy_score
classification_accuracy = accuracy_score(Y_test, Y_predict_SVM)
print('Classification Accuracy: %.4f'
      % classification_accuracy)
print(' ')

# B. For the Classification Error
from sklearn.metrics import accuracy_score
classification_error = 1-classification_accuracy
print('Classification Error: %.4f'
      % classification_error)
print(' ')

# C. For the Sensitivity or Recall Score / True Positive Rate (Kung posotive ang hinahanap, gano kadalas ang positive) Dapat same ang performance ng predicting ng + and - para walang bias sa isa
# True Positive Rate:  Actual Value +, How often Correct
from sklearn.metrics import recall_score
sensitivity = recall_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('Sensitivity or Recall Score: %.4f'
      % sensitivity)
print(' ')

# D. For the Specficity (kung ang actual value is negative, gaano kadalas negative. Counterpart ni sensitivity)
# True Negative Rate: Actual Value -, How often Correct
specificity = TN/(TN+FP)
print('Specificity: %.4f'
      % specificity)
print(' ') 

# yung result bias kay negative compared to positive

# E. For the FP rate .
# False Positive Rate: Actual Value -, How often Inorrect
false_positve_rate = 1-specificity
print('False Positive Rate: %.4f'
      % false_positve_rate)
print(' ')  

# F. For the precision.
# False Negative Rate: Predicted Value +, How often the prediction is Correct
from sklearn.metrics import precision_score
precision = precision_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('Precision: %.4f'
      % precision)
print(' ')  

#Wag ma amaze sa accuracy 

# G. For the F1 score. Relating precision and sensitivity
# False Negative Rate: Predicted Value and Actual Value+, How often the prediction is Correct relation
from sklearn.metrics import f1_score
f1_score = f1_score(Y_test, Y_predict_SVM, pos_label = 'positive', average = 'weighted')
print('F1 Score: %.4f'
      % f1_score)
print(' ')  

# For the Classification Report
from sklearn.metrics import classification_report
svm_optimized_classification_report = classification_report(Y_test, Y_predict_SVM)


