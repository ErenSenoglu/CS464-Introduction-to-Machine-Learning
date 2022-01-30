import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = pd.read_csv('question-2-features.csv') 
labels = pd.read_csv('question-2-labels.csv')
features = features.assign(V0 = np.ones((features.shape[0],1)))
features = features[["V0","sqft_living"]]
features_matrix = features["sqft_living"]
#print(features)
#print(labels)
# Calculating optimal weights
inversed = np.dot(np.transpose(features), features)
inversed = np.linalg.inv(inversed)
beta = np.dot(inversed, np.transpose(features))
weights = np.dot(beta, labels)
print("Results for linear regression")
print('Weight 0: ' + str(weights[0]))
print('Weight 1: ' + str(weights[1]))
# Making predictions based on calculated weights
#predictions = []
coefs = features_matrix * weights[1]
#print(coefs)
preds = coefs + weights[0]
"""
for i in range(labels.shape[0]):
    predictions.append(weights[0] + weights[1] * features_matrix.iloc[i])
preds = pd.DataFrame(predictions)
"""
#print(preds)
# Calculating MSE of the predictions
mse = 0
for i in range(preds.shape[0]):
    mse += ((int(labels.iloc[i]) - int(preds.iloc[i]))**2)
mse = mse / preds.shape[0]
#mse = int(mse)
print("MSE: " + str(mse))

# Polynomial Regression
poly_features = pd.read_csv('question-2-features.csv') 
poly_features = poly_features["sqft_living"]
squared = poly_features ** 2
poly_features = pd.concat([poly_features,squared], axis=1)
ones = pd.DataFrame(np.ones((poly_features.shape[0],1)))
poly_features = pd.concat([ones, poly_features], axis=1)
#print(poly_features)
# Calculating optimal weights
p_inversed = np.dot(np.transpose(poly_features), poly_features)
p_inversed = np.linalg.inv(p_inversed)
p_beta = np.dot(p_inversed, np.transpose(poly_features))
p_weights = np.dot(p_beta, labels)

# Making predictions based on calculated weights
coef_1 = features_matrix * p_weights[1]
coef_2 = squared * p_weights[2]
p_preds = coef_1 + coef_2 + p_weights[0]
#print(p_preds)
p_mse = 0
for i in range(p_preds.shape[0]):
    p_mse += ((float(labels.iloc[i]) - float(p_preds.iloc[i]))**2)
p_mse = p_mse / p_preds.shape[0]
#mse = int(mse)
print("Results for polynomial linear regression")
print('Weight 0: ' + str(p_weights[0]))
print('Weight 1: ' + str(p_weights[1]))
print('Weight 2: ' + str(p_weights[2]))
print("MSE: " + str(p_mse))
plot_labels = np.array(labels)
plot_features = np.array(features_matrix)
sorted_features = np.sort(plot_features)
sorted_indices = np.argsort(plot_features)
plot_p_preds = np.array(p_preds)
plot_preds = np.array(preds)

sorted_labels = []
sorted_preds = []
for i in range(len(sorted_indices)):
    sorted_labels.append(plot_labels[sorted_indices[i]])
    sorted_preds.append(plot_p_preds[sorted_indices[i]])


plt.scatter(plot_features, plot_labels, label='Original Data', c='red')
plt.plot(plot_features, plot_preds, label='Predictions')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.title('Linear Regression Predictions')
plt.legend()
plt.show()
# Second plot
plt.scatter(sorted_features, sorted_labels, label='Original Data', c='red')
plt.plot(sorted_features, sorted_preds, label='Predictions')
plt.xlabel('Sqft_living')
plt.ylabel('Price')
plt.title('Polynomial Linear Regression Predictions')
plt.legend()
plt.show()