from __future__ import print_function
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn import model_selection
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import time
from clr import plot_classification_report; from cm import plot_confusion_matrix

df = pd.read_csv('fine_needle_aspiration_cytology_data.csv')
df.columns = ['id','clump_thickness','unif_cell_size','unif_cell_shape','marg_adhesion','single_epith_size','bare_nuclei','bland_chrom','norm_nucleoli','mitoses','class']
df.drop(['id'], inplace=True, axis=1)
df.replace('?', -99999, inplace=True)
df['class'] = df['class'].map(lambda x: 1 if x == "MALIGNANT" else 0)
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

model = Sequential()
model.add(Dense(9, activation='sigmoid', input_shape=(9,)))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(54, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(27, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['acc'])

mcp = ModelCheckpoint('fine_needle_aspiration_cytology_model.hdf5', monitor="val_acc", save_best_only=True, save_weights_only=False)
tb = TensorBoard(log_dir="fine_needle_aspiration_cytology_logs/{}".format(time.time()))

history = model.fit(X_train, y_train, batch_size=30, epochs=500, verbose=1, validation_data=(X_test, y_test), callbacks=[mcp, tb])
loss = model.evaluate(X_test, y_test, verbose=1, batch_size=30)

model = keras.models.load_model("fine_needle_aspiration_cytology_model.hdf5")

y_pred_sig = model.predict(X_test)
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

cr = metrics.classification_report(y_test, y_pred_rounded, target_names=['Benign', 'Malign']); cr = cr.split("\n")
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