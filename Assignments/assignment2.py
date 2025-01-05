import skimage
from skimage import img_as_float
import numpy as np
from matplotlib import pyplot as plt

# Loading the "cameraman" image
image = img_as_float(skimage.data.camera())
print(f"Shape of the image: {image.shape}.")
# Visualize the image
plt.imshow(image, cmap="gray")
plt.show()

# Computing the SVD 
U, s, VT = np.linalg.svd(image, full_matrices=False)
print(U.shape, s.shape, VT.shape)
print(s)


# # Visualise dyad
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    dyad = np.outer(U[:, i], VT[i, :])
    ax.imshow(dyad, cmap="gray")
plt.show()

# Plot singular values
plt.figure(figsize=(15, 5))
plt.plot(s,'-o' ,linewidth=1.5)
plt.title("Singular values of the 'cameraman' image")
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.grid()
plt.show()

# k-rank approximation

def k_rank(U,s,VT,k):
    return U[:,:k] @ np.diag(s[:k]) @ VT[:k,:]

k_values = [2,5,10,20,50]
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, k in enumerate(k_values):
    image_k = k_rank(U,s,VT,k)
    axes[i].imshow(image_k, cmap="gray")
    axes[i].set_title(f"k = {k}")
plt.show()

# Approximation error
def approx_error(U,s,VT,k):
    x_k = k_rank(U,s,VT,k)
    return np.linalg.norm(image - x_k,'fro')

# Compute the approximation errors for different k values
k_values = range(1,50)
errors = [approx_error(U,s,VT,k) for k in k_values]

plt.figure(figsize=(15, 5))
plt.plot(k_values, errors, '-o', linewidth=1.5)
plt.title('Approximation Error ||X - X_k||_F for Increasing Values of k')
plt.xlabel('k')
plt.ylabel('Approximation Error')
plt.grid()
plt.show()

#Plot Compression Factor

def compression_factor(m, n, k):
    return 1 - (k * (m + n + 1)) / (m * n)

# Get the dimensions of the image
m, n = image.shape

# Compute the compression factor for increasing values of k
compression_factors = [compression_factor(m, n, k) for k in k_values]

# Plot the compression factor
plt.figure(figsize=(10, 6))
plt.plot(k_values, compression_factors, 'g-', linewidth=2)
plt.title('Compression Factor for Increasing Values of k')
plt.xlabel('k')
plt.ylabel('Compression Factor')
plt.grid()
plt.show()
