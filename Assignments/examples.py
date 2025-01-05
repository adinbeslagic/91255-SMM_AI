import pandas as pd 
import numpy as np
import skimage
import matplotlib.pyplot as plt

# # Consider an example matrix
# A = np.array(
#     [
#         [-1, -2, 0, 1, -2, -3],
#         [-1, -2, -3, -2, 0, -3],
#         [-1, -3, 1, 3, 2, -4],
#         [2, 1, -1, 0, -2, 3],
#         [0, -3, -1, 2, -1, -3],
#         [1, -3, 2, 6, 0, -2],
#         [-3, 1, 0, -4, 2, -2],
#         [-2, 2, -2, -6, -2, 0],
#         [-3, -1, 2, 0, 2, -4],
#         [2, -2, 0, 4, -1, 0],
#     ]
# )

# Measure the shape of A: which is the maximum rank?
m, n = A.shape
print(f"The shape of A is: {(m, n)}.")

# Compute the SVD decomposition of A and check the shapes
U, s, VT = np.linalg.svd(A, full_matrices=True)
print(U.shape, s.shape, VT.shape)

# Define the full matrix S
S = np.zeros((m, n))
S[:n, :n] = np.diag(s)

def check_svd(U, S, VT):
    # Check if the SVD decomposition is correct
    A_reconstructed = U @ S @ VT
    print(A_reconstructed, "testing")
    return np.allclose(A, A_reconstructed)
print(check_svd(U, S, VT))
def two_norm(A,U,S,VT):
    return np.linalg.norm((A - U @ S @ VT),2)
print(two_norm(A,U,S,VT))


# # Loading the "cameraman" image
# x = skimage.data.camera()

# # Printing its shape
# print(f"Shape of the image: {x.shape}.")
# # Visualize the image


# plt.imshow(x, cmap="gray")
# plt.show()

data = pd.read_csv('data/train.csv')
# Inspect the data
print(f"Shape of the data: {data.shape}")
print("")
print(data.head())
# Convert data into a matrix
data = np.array(data)

# Split data into a matrix X and a vector Y where:
#
# X is dimension (42000, 784)
# Y is dimension (42000, )
# Y is the first column of data, while X is the rest
X = data[:, 1:]
X = X.T

Y = data[:, 0]

print(X.shape, Y.shape)

d, N = X.shape
def visualize(X, idx):
    # Visualize the image of index 'idx' from the dataset 'X'

    # Load an image in memory
    img = X[:, idx]
    
    # Reshape it
    img = np.reshape(img, (28, 28))

    # Visualize
    plt.imshow(img, cmap='gray')
    plt.show()

# Visualize image number 10 and the corresponding digit.
idx = 10
visualize(X, idx)
print(f"The associated digit is: {Y[idx]}")
def split_data(X, Y, N_train):
    d, N = X.shape

    idx = np.arange(N)
    np.random.shuffle(idx)

    train_idx = idx[:N_train]
    test_idx = idx[N_train:]

    X_train = X[:, train_idx]
    Y_train = Y[train_idx]
    
    X_test = X[:, test_idx]
    Y_test = Y[test_idx]

    return (X_train, Y_train), (X_test, Y_test)

# Test it
(X_train, Y_train), (X_test, Y_test) = split_data(X, Y, 30_000)

print(X_train.shape, X_test.shape)
# Compute centroid
cX = np.mean(X, axis=1)

# Make it a column vector
cX = np.reshape(cX, (d, 1))
print(cX.shape)

# Center the data
Xc = X - cX

# Compute SVD decomposition
U, s, VT = np.linalg.svd(Xc, full_matrices=False)

# Given k, compute reduced SVD
k = 2
Uk = U[:, :k]

# Define projection matrix
P = Uk.T

# Project X_train -> Z_train
Z_train = P @ X_train
print(Z_train.shape)
# Visualize the clusters
ax = plt.scatter(Z_train[0, :], Z_train[1, :], c=Y_train)
plt.legend(*ax.legend_elements(), title="Digit") # Add to the legend the list of digits
plt.xlabel(r"$z_1$")
plt.ylabel(r"$z_2$")
plt.title("PCA projection of MNIST digits 0-9")
plt.grid()
plt.show()
# Define the boolean array to filter out digits
filter_3or4 = (Y==3) | (Y==4)

# Define the filtered data
X_3or4 = X[:, filter_3or4]
Y_3or4 = Y[filter_3or4]