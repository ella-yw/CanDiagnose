import time
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from clr import plot_classification_report; from cm import plot_confusion_matrix

dataset = pd.read_csv('cell_nuclei_characteristics_data.csv')

def dataSetAnalysis(df):
    #view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)
    
    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)
    
    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)
    
    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)
    
    # view distribution of categorical features across the data set
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 30)
    
X = dataset.iloc[:,2:32] # [all rows, col from index 2 to the last one excluding 'Unnamed: 32']
y = dataset.iloc[:,1] # [all rows, col one only which contains the classes of cancer]

print("Before encoding: ")
print(y[100:110])

from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

print("\nAfter encoding: ")
print(y[100:110])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense

# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
#     classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#     return classifier
# classifier = KerasClassifier(build_fn = build_classifier)
# parameters = {'batch_size': [1, 5],
#               'epochs': [100, 120],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train, y_train)

# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
# print("best_parameters: ")
# print(best_parameters)
# print("\nbest_accuracy: ")
# print(best_accuracy)

#best_parameters: {'batch_size': 1, 'epochs': 100, 'optimizer': 'rmsprop'}
#best_accuracy: 0.978021978022

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
import keras

classifier = Sequential() # Initialising the ANN

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

mcp = ModelCheckpoint('cell_nuclei_characteristics_model.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)
tb = TensorBoard(log_dir="cell_nuclei_characteristics_logs/{}".format(time.time()))

classifier.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 1, epochs = 100, callbacks = [mcp, tb])

loss = classifier.evaluate(X_test, y_test, verbose=1, batch_size=30)

classifier = keras.models.load_model("cell_nuclei_characteristics_model.hdf5")

y_pred_sig = classifier.predict(X_test)
y_pred_rounded = [ 1 if y>=0.5 else 0 for y in y_pred_sig ]

import sklearn
from sklearn import metrics

# Accuracies
print(sklearn.metrics.accuracy_score(y_test, y_pred_rounded))
print(sklearn.metrics.roc_auc_score(y_test, y_pred_rounded))

# Losses
print(sklearn.metrics.brier_score_loss(y_test, y_pred_rounded)) #print(sklearn.metrics.hamming_loss(y_true, y_pred)) # print(sklearn.metrics.zero_one_loss(y_test, y_pred)) #ALL SAME!!!
print(sklearn.metrics.hinge_loss(y_test, y_pred_rounded))
print(sklearn.metrics.log_loss(y_test, y_pred_rounded))

# Similarities
print(sklearn.metrics.cohen_kappa_score(y_test, y_pred_rounded))
print(sklearn.metrics.jaccard_similarity_score(y_test, y_pred_rounded))
print(sklearn.metrics.matthews_corrcoef(y_test, y_pred_rounded))

pd.Series([ y[0] for y in y_pred_sig ]).plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Predictions (y) Ranges Histogram')
plt.grid(axis='y', alpha=0.75)
plt.savefig('Y_Pred_Histogram.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred_sig)
auc = metrics.roc_auc_score(y_test, y_pred_sig)
plt.plot(fpr, tpr, label="Model: " + str(time.time()) + " | AUC=" + str(auc))

plt.xlabel('Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_AUC.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

cm = metrics.confusion_matrix(y_test, y_pred_rounded)
plot_confusion_matrix(cm, classes=['Benign', 'Malign'], title='Confusion Matrix')
plt.savefig('Confusion_Matrix.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

cr = metrics.classification_report(y_test, y_pred_rounded, target_names=['0', '1']); cr = cr.split("\n")
plot_classification_report(cr[0] + '\n\n' + cr[2] + '\n' + cr[3] + '\n' + cr[7] + '\n',
                           title = 'Classification Report')
plt.savefig('Classification_Report.png', dpi=200, format='png', bbox_inches='tight'); plt.close();

acc = metrics.accuracy_score(y_test, y_pred_rounded)
k = metrics.cohen_kappa_score(y_test, y_pred_rounded)

ben_acc = cm[0][0]/sum(cm[0]); mal_acc = cm[1][1]/sum(cm[1])
category_wise_acc = "Benignity: " + str(ben_acc) + "% | Malignancy: " + str(mal_acc) + "% "     

f = open("Metrics.txt","w+")
f.write("\n::::::::::::::::: OVERALL OUTPUT DERIVATIONS :::::::::::::::::")
f.write('\n\nAccuracy: ' + str(acc) + '%' +
      '\n\nCategorized Accuracies: ' + str(category_wise_acc) +
      
      '\n\nCohen\'s Kappa Co-efficient (K): ' + str(k) +
      
      '\n\nArea Under the Curve (AUC): ' + str(auc) + '\n')
f.close()

fig = plt.figure(figsize=(16, 8))
fig.suptitle('Output Derivations', fontsize=20)
fig.add_subplot(211); plt.imshow(plt.imread("Classification_Report.png")); plt.axis('off')
fig.add_subplot(234); plt.imshow(plt.imread("Confusion_Matrix.png")); plt.axis('off')
fig.add_subplot(212); plt.imshow(plt.imread("ROC_AUC.png")); plt.axis('off')
fig.add_subplot(236); plt.imshow(plt.imread("Y_Pred_Histogram.png")); plt.axis('off')
plt.savefig("Output_Derivations.png", dpi=500, format='png'); plt.close();