#############################
# Project-2                 #
# Author : Manan Tarun Shah #
# Python version 3.9.7      #
#############################

import pandas as pd                                    # for data frame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA                  # for Principal Component Analysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier       # for Multi-level Perceptron Neural network
# from warnings import filterwarnings
# filterwarnings('ignore')

# read the database provided using pandas
df = pd.read_csv('sonar_all_data_2.csv', header=None)
# print(df)

X = df.iloc[:, :-2].values       # features are in all columns but last 2
y = df.iloc[:, -2].values        # classes are in 2nd last column
# print(X, y)

# now split the data with 30% test and 70% training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()                     # apply standardization
X_train_std = stdsc.fit_transform(X_train)   # applying fit and transform to training data
X_test_std = stdsc.transform(X_test)         # applying SAME transform to test data

list_acc = []         # list of accuracies for different number of components
list_top_comp = []    # list of number of components for each case
list_y_pred = []      # list of y_pred for different number of components

###########################################################################################
# iterating to find, PCA with how many components lead to the best accuracy for test data #
# n_top_comp is the variable that means n top components                                  #
###########################################################################################
for n_top_comp in range(1, 61):

    pca = PCA(n_components=n_top_comp)              # apply PCA
    X_train_pca = pca.fit_transform(X_train_std)    # applying fit and transform to standardized training data
    X_test_pca = pca.transform(X_test_std)          # applying SAME transform to standardized test data

    # using Multi-Level Perceptron to create neural network (parameters fine-tuned for best results)
    model = MLPClassifier(hidden_layer_sizes=(20), activation='tanh', max_iter=1700, alpha=0.00001, solver='adam', tol=0.0001, random_state=1)

    model.fit(X_train_pca, y_train)                 # fit the model on the training data

    y_pred = model.predict(X_test_pca)              # how did we do on the test data
    print('Number of components: ', n_top_comp)
    print('Test Accuracy: %.2f\n' % accuracy_score(y_test, y_pred))

    list_acc.append(round(100 * accuracy_score(y_test, y_pred)))
    list_top_comp.append(n_top_comp)
    list_y_pred.append(y_pred)

# print(list_y_pred)
print('Maximum accuracy:', max(list_acc))

# number of components would be at the index where the max accuracy was found in list of accuracies
print('Number of components that achieved this accuracy:', list_top_comp[list_acc.index(max(list_acc))])

# y_pred_max would be at the index where the max accuracy was found in list of accuracies
y_pred_max = list_y_pred[list_acc.index(max(list_acc))]
# print(y_test, y_pred_max)

cmat = confusion_matrix(y_test, y_pred_max)
print('\nConfusion matrix:\n', cmat)

# Visualize the results
plt.plot(list_top_comp, list_acc)
plt.xlabel('Number of PCA components')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Number of components')
plt.show()