import numpy as np, pandas as pd
from matplotlib import pyplot as plt
plt.style.use = ('seaborn')

df = pd.read_csv('Cleaned_data.csv'); df.head()

X = df.drop('Age', axis=1)
X = X.drop('Shape', axis=1)
X = X.drop('Margin', axis=1)
X = X.drop('Severity', axis=1).values
y = df['Severity'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
    
from sklearn.metrics import confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else: print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def compute_cnf(classifier,x_test,y_test):
    
    cnf_matrix = confusion_matrix(classifier.predict(x_test),y_test)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign','Malignant'], title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['Benign','Malignant'], normalize=True, title='Normalized confusion matrix')

    plt.show()
    
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth = 5, n_estimators = 100)
classifier.fit(X_train,y_train)
print (classifier.score(X_test,y_test))
compute_cnf(classifier,X_test,y_test)

import joblib
joblib.dump(classifier, 'LinearSVC_Model (Density-to-Cancer).joblib')

#classifier.predict([[4, 4]])[0]