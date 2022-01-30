import pandas as pd
import numpy as np

features_train = pd.read_csv('question-3-features-train.csv') 
features_test = pd.read_csv('question-3-features-test.csv')
labels_train = pd.read_csv('question-3-labels-train.csv')
labels_test = pd.read_csv('question-3-labels-test.csv')
#print(features_train)
# Normalizing the features
print(features_train["Amount"])
min_val = features_train["Amount"].min(axis=0)
max_val = features_train["Amount"].max(axis=0)
features_train["Amount"].sub(min_val)
features_train["Amount"] = features_train["Amount"].div((max_val-min_val))
#print(features_train)
feature_matrix = features_train.assign(V0 = np.ones((features_train.shape[0],1)))
feature_matrix_t = features_test.assign(V0 = np.ones((features_test.shape[0],1)))
#print(feature_matrix["V0"])
n_feature_matrix = np.array(feature_matrix)
labels_test = np.array(labels_test)
labels_train = np.array(labels_train)
features_test = np.array(feature_matrix_t)

def decide_decision_boundary(train_matrix, weights, labels):
    positives = 0
    negatives = 0
    for data_idx in range(train_matrix.shape[0]):
        boundary = np.sum(np.dot(train_matrix[data_idx], weights))
        if boundary >= 0:
            if labels[data_idx] == 1:
                positives += 1
            else:
                negatives += 1
    if negatives > positives:
        #print("Negatives:"+ str(negatives_above))
        #print("Positives: "+ str(positives_above))
        return (weights, 0)
    return (weights, 1)
print(decide_decision_boundary(n_feature_matrix, np.zeros((30,1)), labels_train))

def full_batch(features, weights, labels, boundary):
    y = labels if boundary == 1 else 1 - labels
    # Calculate prediction
    #print("Weights in full batch are: "+ str(weights))
    coef = np.dot(features, weights)
    #print("Coef is: " + str(coef))
    #print("End of coef")
    coef = np.exp(coef)
    predictions = coef / (1+coef)
    return np.dot(np.transpose(features), (y - predictions))

def stoch_batch(features, weights, labels, boundary):
    y = labels if boundary == 1 else 1 - labels
    # Calculate prediction
    #print("Weights in full batch are: "+ str(weights))
    coef = np.dot(features, weights)
    #print("Coef is: " + str(coef))
    #print("End of coef")
    coef = np.exp(coef)
    predictions = coef / (1+coef)
    features = features.reshape(1,30)
    error = y - predictions
    #print(error)
    error = error.reshape(1,1)
    return np.dot(np.transpose(features), error)


print(full_batch(n_feature_matrix, np.zeros((30,1)), labels_train, 0))
def gaussian_decision_boundary(train_matrix, mean, std, labels):
    w = np.random.normal(mean, std, train_matrix.shape[1])
    w = w.reshape(30,1)
    positives_above = 0
    negatives_above = 0
    for i in range(train_matrix.shape[0]):
        boundary = np.sum(np.dot(train_matrix[i], w))
        if boundary > 0:
            if labels[i] == 1:
                positives_above += 1
            else:
                negatives_above += 1
    if negatives_above > positives_above:
        return (w, 0)
    return (w, 1)
def train_mini_batch_model(feature_matrix, labels, step_size, batch_size):
    decision = gaussian_decision_boundary(feature_matrix, 0, 0.01, labels)
    weights = decision[0]
    boundary = decision[1]
    batch = batch_size
    for i in range(1000):
        start = 0
        for j in range(feature_matrix.shape[0] // batch):
            weights = np.add(weights, step_size * full_batch(feature_matrix[start: start + batch], weights, labels[start: start + batch], boundary))
            start += batch
    return [boundary, weights]    
def train_stochastic_model(feature_matrix, labels, step_size):
    decision = gaussian_decision_boundary(feature_matrix, 0, 0.01, labels)
    weights = decision[0]
    boundary = decision[1]
    for i in range(1000):
        for j in range(feature_matrix.shape[0]):
            #print("LABELS J :" + str(labels[j]))
            #print("WEIGHTS J:" + str(weights.shape))
            weights = np.add(weights, step_size * stoch_batch(feature_matrix[j], weights, labels[j], boundary))
    return [boundary, weights]

def train_full_batch(features, labels, stepsize):
    boundary = decide_decision_boundary(n_feature_matrix, np.zeros((30,1)), labels_train)
    initial_decision = boundary[1]
    weights = boundary[0]
    #print("Weights in train are:  "+ str(weights))
    for i in range(1000):
        #print("Weights are: "+ str(weights))
        #print("Decision is: "+ str(initial_decision))
        weights = np.add(weights, stepsize * full_batch(features, weights, labels, initial_decision))
        #print(full_batch(features, weights, labels, initial_decision))
        #if i % 500 == 499:
            #print("Weights at iteration " + str(i+1) + ": " + str(weights))

    return weights, initial_decision 
def predict_labels(features, weights, boundary):
    predictions = []
    for item in features:
        if boundary == 0:
            if np.dot(item, weights) > 0:
                predictions.append(0)
            else:
                predictions.append(1)
        else:
            if np.dot(item, weights) > 0:
                predictions.append(1)
            else:
                predictions.append(0)
    return predictions
mini_model = train_mini_batch_model(n_feature_matrix, labels_train, 0.0001, 100)
stoch_model = train_stochastic_model(n_feature_matrix, labels_train, 0.0001)
model = train_full_batch(n_feature_matrix, labels_train, 0.0001)
decision = model[1]
weights = model[0]
stoch_decision = stoch_model[0]
stoch_weights = stoch_model[1]
m_decision = mini_model[0]
m_weights = mini_model[1]
#print("Stoch Weights: " + str(pd.DataFrame(stoch_weights)))
#print("Weights: " + str(pd.DataFrame(weights)))
stoch_predictions = predict_labels(features_test, stoch_weights, stoch_decision)
predictions = predict_labels(features_test, weights, decision)
mini_predictions = predict_labels(features_test, m_weights, m_decision)
def test_model(predictions, labels_test, name):
    tp, fp, fn, tn = 0,0,0,0
    for row in range(labels_test.size):
            if labels_test[row] == 1:
                if labels_test[row] == predictions[row]:
                    tp += 1
                else:
                    fn += 1
            else:
                if labels_test[row] == predictions[row]:
                    tn += 1
                else:
                    fp += 1
    print('Performance Metrics for '+ name +' Model:')
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print('Accuracy:' + str(accuracy))
    precision = tp / (tp + fp)
    print('Precision:' + str(precision))
    recall = tp / (tp + fn)
    print('Recall:' + str(recall))
    npv = tn / (tn + fn)
    print('Negative Predictive Value(NPV): ' + str(npv))
    fpr = fp / (fp + tn)
    print('False Positive Rate(FPR): ' + str(fpr))
    fdr = fp / (tp + fp)
    print('False Discovery Rate(FDR): ' + str(fdr))
    f1 = (2 * precision * recall) / (precision + recall)
    print('F1 Measure: ' + str(f1))
    f2 = (5 * precision * recall) / (4 * precision + recall)
    print('F2 Measure: ' + str(f2))
test_model(predictions, labels_test, "Full-Batch")
test_model(mini_predictions, labels_test, "Mini-Batch")
test_model(stoch_predictions, labels_test, "Stochastic")
