import numpy as np
import csv
def safe_mult( x,y ):
    r = x*y
    if np.isnan(r):
        r = 0
    return r
f = open("vocabulary.txt", "r")
vocab = {}
# This equals to vocabulary size
vocab_size = 0
for line in f:
  word = line.split(",")
  #print(word[0])
  vocab[word[0]] = vocab_size
  vocab_size = vocab_size +1
print("Vocabulary created with size: " , vocab_size)
# Creating the feature set as 2D array
feature_set = np.zeros((4085,44020))
index = 0
index_t = 0
labels_t = []
with open('y_test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        #print(row[0])
        labels_t.append(int(row[0]))

feature_set_t = np.zeros((1086,44020))
with open('x_test.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        #print(len(row))
        feature_set_t[index_t] = row
        index_t = index_t +1
with open('x_train.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        #print(len(row))
        feature_set[index] = row
        index = index +1
# Labels
labels = []
with open('y_train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        #print(row[0])
        #np.append(labels, int(row[0]))
        labels.append(int(row[0]))
np.seterr(all="ignore")
npLabels = np.array(labels)
print("Row Count: ", index)
spam_rows_sum = np.sum(feature_set, axis=1)*npLabels
sum_tjy_spam = np.sum(spam_rows_sum, axis=0)
inversed = np.where(npLabels==1, 0, 1)
normal_rows_sum = np.sum(feature_set, axis=1)*inversed
sum_tjy_normal = np.sum(normal_rows_sum, axis=0)


n_spam = np.count_nonzero(npLabels==1)
n_normal = 4085-n_spam
tjy_spam = np.zeros(int(vocab_size))
tjy_normal = np.zeros(int(vocab_size))

theta_s_j_y_normal = np.zeros(int(vocab_size))
theta_s_j_y_spam = np.zeros(int(vocab_size))

theta_j_y_normal = np.zeros(int(vocab_size))
theta_j_y_spam = np.zeros(int(vocab_size))

MAP_theta_j_y_spam = np.zeros(int(vocab_size))
MAP_theta_j_y_normal = np.zeros(int(vocab_size))

# Normal cols
normal_set = (feature_set*inversed[:,None])
spam_set = (feature_set*npLabels[:,None])
tjy_normal = np.sum(normal_set, axis=0)
sjy_normal = np.where(normal_set>0, 1,0)
sjy_normal = np.sum(sjy_normal, axis=0)
# Spam cols
tjy_spam = np.sum(spam_set, axis=0)
sjy_spam = np.where(spam_set>0, 1,0)
sjy_spam = np.sum(sjy_spam, axis=0)

theta_spam = np.zeros(vocab_size)            
theta_spam = tjy_spam / sum_tjy_spam
theta_normal = np.zeros(vocab_size)
theta_normal = (tjy_normal / sum_tjy_normal)
log_theta_normal = np.where(theta_normal != 0, np.log(theta_normal), np.nan_to_num(-np.inf))

log_theta_spam = np.where(theta_spam != 0, np.log(theta_spam), np.nan_to_num(-np.inf))

# MAP Estimation
tjy_spam_map = (tjy_spam+1)
tjy_normal_map = (tjy_normal+1)
MAP_theta_j_y_spam = (tjy_spam_map / (sum_tjy_spam+vocab_size))
MAP_theta_j_y_normal = ( tjy_normal_map / (sum_tjy_normal+vocab_size))
MAP_theta_j_y_spam = np.log(MAP_theta_j_y_spam)
MAP_theta_j_y_normal = np.log(MAP_theta_j_y_normal)

# Bernoulli 
theta_s_j_y_normal = (sjy_normal / n_normal)
theta_s_j_y_spam = (sjy_spam / n_spam)
#log_theta_normal_s = np.where(theta_s_j_y_normal != 0, np.log(theta_s_j_y_normal), np.nan_to_num(-np.inf))
#log_theta_spam_s = np.where(theta_s_j_y_spam != 0, np.log(theta_s_j_y_spam), np.nan_to_num(-np.inf))
truePositive = [0,0,0]
falsePositive = [0,0,0]
trueNegative = [0,0,0]
falseNegative = [0,0,0]
trues = 0
print("Models have been trained.")
print("Testing phase of the models has been started.")
pNormal = np.zeros(len(feature_set_t))
pSpam = np.zeros(len(feature_set_t))
# Question 3.2
for i in range(len(feature_set_t)):
    
    spam_Row = feature_set_t[i]*log_theta_spam
    spam_sum = np.sum(spam_Row)
    normal_Row = feature_set_t[i]*log_theta_normal
    normal_sum = np.sum(normal_Row)
    logP = np.log(float(n_normal/ 4085))
    pNormal[i]= logP + normal_sum
    pSpam[i] = logP + spam_sum

predicts = np.where(pNormal<pSpam, 1,0)
#print(predicts)
results = np.where(labels_t==predicts,1,0)
total = np.sum(results)
print("MLE Testing is done with accuracy: ", total/len(feature_set_t))



# Question 3.4
for i in range(len(feature_set_t)):
    """
    # MLE
    summationNormal = float(0)
    summationSpam = float(0)
    """
    # MAP
    m_w_theta_j_y_normal = float(0)
    m_w_theta_j_y_spam = float(0)
    
    # Bernoulli
    w_spam = float(1)
    w_normal = float(1)

    for j in range(vocab_size):
        # MLE
        #summationSpam = summationSpam + float(feature_set_t[i][j] * log_theta_spam[j])
        #summationNormal = summationNormal + float(feature_set_t[i][j] * log_theta_normal[j] )
        #w_theta_j_y_normal = w_theta_j_y_normal + float(feature_set[i][j] * theta_j_y_normal[j] )
        #w_theta_j_y_spam = w_theta_j_y_spam + float(feature_set[i][j] * theta_j_y_spam[j])
        
        # MAP
        m_w_theta_j_y_normal = m_w_theta_j_y_normal + (MAP_theta_j_y_normal[j] * feature_set_t[i][j])
        m_w_theta_j_y_spam = m_w_theta_j_y_spam + (MAP_theta_j_y_spam[j] * feature_set_t[i][j])
        
        # Bernoulli
        if feature_set_t[i][j] != 0:
            temp = 1
        else:
            temp = 0
        w_normal = w_normal * (  float (temp * theta_s_j_y_normal[j]) + safe_mult( (1 - temp), (1 - theta_s_j_y_normal[j]) ) )
        w_spam = w_spam * (  float (temp * theta_s_j_y_spam[j]) + safe_mult( (1 - temp), (1 - theta_s_j_y_spam[j]) ) )
        
    #print("W_Normal: ",w_normal, temp, theta_s_j_y_normal )
    #print("W_Spam: ",w_spam, temp, theta_s_j_y_spam )
    # MLE
    """
    logP = np.log(float(n_normal/ 4085))
    pNormal= logP + summationNormal
    pSpam = logP + summationSpam
    """
    # MAP
    MpNormal = logP + m_w_theta_j_y_normal
    MpSpam = logP + m_w_theta_j_y_spam
    
    # Bernoulli
    BpNormal = logP +  np.nan_to_num(np.log(w_normal))
    BpSpam = logP +  np.nan_to_num(np.log(w_spam))
    if BpNormal >= BpSpam:
        if labels_t[i] == 1:
            falseNegative[2] +=1
        else:
            trueNegative[2] +=1        
    else:
        if labels_t[i] == 1:
            truePositive[2] +=1
        else:
            falsePositive[2] +=1
    
    # MAP Estimation belongs to confusion[1]
    if MpNormal >= MpSpam:
        if labels_t[i] == 1:
            falseNegative[1] +=1
        else:
            trueNegative[1] +=1        
    else:
        if labels_t[i] == 1:
            truePositive[1] +=1
        else:
            falsePositive[1] +=1
    
    # MLE Estimation belongs to confusion[0]
    if pNormal[i] < pSpam[i]:
        prediction = 1
        if labels_t[i] == 0:
            falsePositive[0] +=1
        else:
            truePositive[0] +=1    
    else:
        prediction = 0
        if labels_t[i] == 1:
            falseNegative[0] +=1
        else:
            trueNegative[0] +=1
    if labels_t[i] == prediction:
        trues = trues + 1

accuracy = float((trueNegative[0]+truePositive[0]) / len(labels_t))
accuracy1 = float((truePositive[1]+trueNegative[1]) / len(labels_t))
accuracy2 = float((truePositive[2]+trueNegative[2]) / len(labels_t))
#print("Sum Spam: ",sum_tjy_spam)
#print("Sum Normal: ",sum_tjy_normal)
print("P normal: ", float(n_normal/ 4085))
print("Confusion Matrix")
print("MLE-MAP-Bernoulli")
print("True Positives: ", truePositive)
print("True Negatives: ", trueNegative)
print("False Negatives: ", falseNegative)
print("False Positives: ", falsePositive)
print("Total Estimations: ", len(labels_t))
print("Accuracy MLE: ", accuracy)
print("Accuracy MAP: ", accuracy1)
print("Accuracy Bernoulli: ", accuracy2)
