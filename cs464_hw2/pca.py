import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df=pd.read_csv('digits.csv', sep=',',header=None, skiprows = 1)
# Reading features
features = pd.read_csv('digits.csv', sep=',',header=None, skiprows = 1,usecols=range(1,785) )
# Extracting labels
labels = df[0]
column_count = 785
row_count = 10000 

# Calculating mean and mean centring
col_means = features.mean(axis=0)
#print(col_means)
centered_digits = features.sub(col_means)



#mean = df.mean()
#print("Mean: "+str(mean))
# Calculating std
#std = df.values.std(ddof=1)
#print("Std: "+str(std))
#cov_matrix = centered_digits.cov()
#print(cov_matrix)
#values, vectors = np.linalg.eig(cov_matrix)
#print(values)

#Finding eigenvalues with SVD
centered_digits = np.array(centered_digits)
u, s, v = np.linalg.svd(centered_digits)
eig_vals = s**2

# Calculating PVE
total = int(np.sum(eig_vals))
#print("Sum: "+str(total))
#print(eig_vals)
PVE = [eig / total for eig in eig_vals]
datas = np.multiply(PVE[0:256],100)
plt.plot(range(1,257), datas)
plt.xlabel("Principal Component")
plt.ylabel("PVE's Percentage")
plt.title("PC vs PVE Graph")
plt.show()
exit()

# Bar chart for PVE values
plt.bar(range(1,11), PVE[0:10])
plt.xticks(range(1,11))
plt.xlabel("Principal Component")
plt.ylabel("PVE's Percentage")
plt.title("PC vs PVE Graph")
plt.show()

#Method to plot the digits
def display(ims, cmap="gray", title = "", index = []):
    """sp_row = int(np.ceil(np.sqrt(len(ims))))
    sp_col = int(np.ceil(np.sqrt(len(ims))))
    fig, ax = plt.subplots(sp_row, sp_col, figsize = (17, 9)) 
    i = 0
    j = -1  
    # Plot the sample images"""
    for k in range(len(ims)):

        im = ims[k]
        """
        if (j + 1 == sp_col):
            i = (i + 1) % sp_row
            
        j = (j + 1) % sp_col
        """
        plt.gray()
        if(title =="Principal Component"):
          title = "Principal Component " + str(k+1)
        else:
          title = title + " " + str(index[k]) + "PC's"
        plt.title(title)
        plt.imshow(im.reshape(28,28))
        plt.show()
        if(title =="Principal Component "+ str(k+1)):
          title = "Principal Component"
        else:
          title = "Reconstructed with"
        """
        ax[i, j].imshow(im.reshape(28,28), cmap=cmap)
        ax[i, j].axis("off")
        """
#Getting PCs
data = []
for i in range(1,11):
    # Individual principal components
    transpose = v.T[ : , i-1:i ]
    pc = transpose.reshape(28,28,1)
    data.append(pc)
# Normalize
data = (data - np.min(data)) / (np.max(data) - np.min(data))

display(data, title = "Principal Component")

# Different k values
pve_k = [2**a for a in range(3,9)]
pve_vals = []
for index in pve_k:
  pve_vals.append(float(PVE[index-1])* 100)
# Bar chart for PVE values
#print(pve_vals)

plt.bar(range(1,7), pve_vals, color="orange")
plt.xticks(range(1,7), pve_k)
plt.xlabel("Principal Component")
plt.ylabel("PVE's Percentage")
plt.title("PC vs PVE Graph")
plt.show()



#Reconstruction
k = [1, 3, 5, 10, 50, 100, 200, 300]
reconstructed_digits = []
for i in k:
    PC = v.T[:, 0:i].reshape(784,i)
    recons = np.dot(np.dot(centered_digits[0,:], PC), PC.T)
    recons = recons + col_means
    recons = (recons - np.min(recons, axis = 0)) / (np.max(recons, axis = 0) - np.min(recons, axis = 0))
    reconstructed_digits.append(recons.values.reshape(28,28,1))

display(reconstructed_digits, index = k, title = 'Reconstructed with')