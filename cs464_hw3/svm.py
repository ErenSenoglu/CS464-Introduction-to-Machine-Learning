import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
def evaluate_model(tp,tn,fp,fn,score):
    print("Confusion Matrix -> ")
    print(tp,tn,fp,fn)
    print("Accuracy: " + str(score))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    npv = tn / (tn + fn)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    """
    print('Negative Predictive Value(NPV): ' + str(npv))
    fpr = fp / (fp + tn)
    print('False Positive Rate(FPR): ' + str(fpr))
    fdr = fp / (tp + fp)
    print('False Discovery Rate(FDR): ' + str(fdr))
    """
    f1 = (2 * precision * recall) / (precision + recall)
    print('F1 Measure: ' + str(f1))
    f2 = (5 * precision * recall) / (4 * precision + recall)
    print('F2 Measure: ' + str(f2))
    return tp,tn,fp,fn

features = pd.read_csv('breast_cancer.csv')
#print(len(features))
train = features.iloc[:500,:8]
train_labels = features.iloc[:500,8:]
test = features.iloc[501:,:8]
test_labels = features.iloc[501:,8:]

#print(test_labels.head())
#print(test.head())
np_train = train.to_numpy()
np_train_labels = train_labels.to_numpy()
np_test = test.to_numpy()
np_test_labels = test_labels.to_numpy()
#print(np_train.shape)
kfold_test = np.asarray((np_train[:50], np_train[50:100], np_train[100:150], np_train[150:200], np_train[200:250],np_train[250:300], np_train[300:350], np_train[350:400], np_train[400:450], np_train[450:500]))
kfold_train = np.asarray((np_train[50:], np.concatenate((np_train[:50], np_train[100:])), np.concatenate((np_train[:100], np_train[150:])), np.concatenate((np_train[:150], np_train[200:])),np.concatenate((np_train[:200], np_train[250:])),np.concatenate((np_train[:250], np_train[300:])),np.concatenate((np_train[:300], np_train[350:])),np.concatenate((np_train[:350], np_train[400:])),np.concatenate((np_train[:400], np_train[450:])) ,np_train[:450]))
kfold_test_labels = np.asarray((np_train_labels[:50], np_train_labels[50:100], np_train_labels[100:150], np_train_labels[150:200], np_train_labels[200:250],np_train_labels[250:300], np_train_labels[300:350], np_train_labels[350:400], np_train_labels[400:450], np_train_labels[450:500]))
kfold_train_labels = np.asarray((np_train_labels[50:], np.concatenate((np_train_labels[:50], np_train_labels[100:])), np.concatenate((np_train_labels[:100], np_train_labels[150:])), np.concatenate((np_train_labels[:150], np_train_labels[200:])),np.concatenate((np_train_labels[:200], np_train_labels[250:])),np.concatenate((np_train_labels[:250], np_train_labels[300:])),np.concatenate((np_train_labels[:300], np_train_labels[350:])),np.concatenate((np_train_labels[:350], np_train_labels[400:])),np.concatenate((np_train_labels[:400], np_train_labels[450:])) ,np_train_labels[:450]))
"""
first =  np.delete(np_train, np.s_[50:100], 0)
sec =  np.delete(np_train, np.s_[100:150], 0)
print(first.dtype)
print(sec.dtype)
a = np.concatenate((first,sec),axis=0)
first =  np.delete(np_train, np.s_[150:200], 0)
first =  np.delete(np_train, np.s_[200:250], 0)
first =  np.delete(np_train, np.s_[250:300], 0)
first =  np.delete(np_train, np.s_[300:350], 0)
first =  np.delete(np_train, np.s_[350:400], 0)
first =  np.delete(np_train, np.s_[400:450], 0)
first =  np.delete(np_train, np.s_[450:500], 0)"""
#kfold_train= np.asarray(np_train[50:],a)
#print(kfold_train.shape)
#print(kfold_train_labels[0])
cons = np.array((10**-3, 10**-2, 10**-1, 1, 10**1, 10**2, 10**3))
errors = np.zeros(7)
index = 0
for c in cons:
    mse = np.zeros(10)
    for i in range(10): # kfold
        model = svm.LinearSVC(C=c, dual=False, fit_intercept=True, loss = "squared_hinge", max_iter=1000, penalty = "l1", random_state=0, tol=1e-05)
        model.fit(kfold_train[i], kfold_train_labels[i].ravel())
        predictions = model.predict(kfold_test[i])
        mse[i] = np.sum(np.abs(kfold_test_labels[i] - predictions))
    errors[index] = float(np.sum(mse) / 10)
    index = index + 1

c = cons[np.argmin(errors)]
print("Linear SVM with C = " + str(c))
# Using the c value that we have found as best
model = svm.LinearSVC(C=c, dual=False, fit_intercept=True, loss = "squared_hinge", max_iter=1000, penalty = "l1", random_state=0, tol=1e-05)
model.fit(np_train, np_train_labels.ravel())
score = model.score(np_test, np_test_labels)
predictions = model.predict(np_test)
tp,tp1,tp2 = 0,0,0
tn = 0
fp,fp1,fp2 = 0,0,0
fn = 0

for i in range(len(test_labels)):
    if (test_labels.iloc[i,-1] == 1) & (predictions[i] == 1): # tp
        tp = tp + 1
        tp1 = tp1 + 1
    elif (test_labels.iloc[i,-1] == 0) & (predictions[i] == 0): # tn
        tn = tn + 1
        tp2 = tp2 + 1
    elif (test_labels.iloc[i,-1] == 0) & (predictions[i] == 1): # fp
        fp = fp + 1
        fp1 = fp1 + 1
    elif (test_labels.iloc[i,-1] == 1) & (predictions[i] == 0): # fn
        fn = fn + 1
        fp2 = fp2 + 1
evaluate_model(tp,tn,fp,fn,score)
"""
print("Micro Precision: "+str((tp1+tp2)/(tp1+fp1+tp2+fp2)))
print("Micro Recall: "+str((tp1+tp2)/(tp1+fp1+tp2+fp2)))
print("Macro Precision: "+ str(((tp1/(tp1+fp1))+(tp2/tp2+fp2))/2))
print("Macro Recall: "+ str(((tp1/(tp1+fp2))+(tp2/(tp2+fp1)))/2))
"""
c_vals = pd.DataFrame(cons)
e_vals = pd.DataFrame(errors)
#print(c_vals)
#print(e_vals)
x = [i for i in range(7)]

plt.plot(x, errors, c='red')
plt.xticks(x, cons)
#plt.plot(cons, errors, label='Predictions')
plt.xlabel('C Value')
plt.ylabel('MSE')
plt.title('SVM C Values and MSEs')
plt.show()


gammas = np.array((2**-4, 2**-3, 2**-2, 2**-1, 1, 2**1))
errors = np.zeros(len(gammas))
index = 0
for val in gammas:
    mse = np.zeros(10)
    for i in range(10): # kfold
        model = svm.SVC(kernel = "rbf", gamma = val)
        model.fit(kfold_train[i], kfold_train_labels[i].ravel())
        predictions = model.predict(kfold_test[i])
        mse[i] = np.sum(np.abs(kfold_test_labels[i] - predictions))
    errors[index] = float(np.sum(mse) / 10)
    index = index + 1

gamma = gammas[np.argmin(errors)]
print("SVM with RBF kernel, gamma = " + str(gamma))

model = svm.SVC(kernel = "rbf", gamma = gamma)
model.fit(np_train, np_train_labels.ravel())
score = model.score(np_test, np_test_labels)
preds_rbf = model.predict(np_test)
tp,tp1,tp2 = 0,0,0
tn = 0
fp,fp1,fp2 = 0,0,0
fn = 0
for i in range(len(test_labels)):
    if (test_labels.iloc[i,-1] == 1) & (preds_rbf[i] == 1): # tp
        tp = tp + 1
        tp1 = tp1 + 1
    elif (test_labels.iloc[i,-1] == 0) & (preds_rbf[i] == 0): # tn
        tn = tn + 1
        tp2 = tp2 + 1
    elif (test_labels.iloc[i,-1] == 0) & (preds_rbf[i] == 1): # fp
        fp = fp + 1
        fp1 = fp1 + 1
    elif (test_labels.iloc[i,-1] == 1) & (preds_rbf[i] == 0): # fn
        fn = fn + 1
        fp2 = fp2 + 1
evaluate_model(tp,tn,fp,fn,score)
"""
print("Micro Precision: "+str((tp1+tp2)/(tp1+fp1+tp2+fp2)))
print("Micro Recall: "+str((tp1+tp2)/(tp1+fp1+tp2+fp2)))
print("Macro Precision: "+ str(((tp1/(tp1+fp1))+(tp2/tp2+fp2))/2))
print("Macro Recall: "+ str(((tp1/(tp1+fp2))+(tp2/(tp2+fp1)))/2))
"""
x = [i for i in range(6)]

plt.plot(x, errors, c='red')
plt.xticks(x, gammas)
#plt.plot(cons, errors, label='Predictions')
plt.xlabel('Gamma Value')
plt.ylabel('MSE')
plt.title('SVM Gamma Values and MSEs')
plt.show()

