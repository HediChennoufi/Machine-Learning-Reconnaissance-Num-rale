from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import data, color
from skimage.filters import sobel
import matplotlib.pyplot as plt
import numpy as np

##########################################
## Data loading and first visualisation
##########################################

# Load the handwritten digits dataset
digits = load_digits()
print(digits.images[0])

# Visualize some images
plt.matshow(digits.images[0], cmap="gray")
plt.show()

# Display at least one random sample par class (some repetitions of class... oh well)
def plot_multi(data, y):
    '''Plots 16 digits'''
    nplots = 16
    nb_classes = len(np.unique(y))
    cur_class = 0
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        to_display_idx = np.random.choice(np.where(y == cur_class)[0])
        plt.imshow(data[to_display_idx].reshape((8,8)), cmap='binary')
        plt.title(cur_class)
        plt.axis('off')
        cur_class = (cur_class + 1) % nb_classes
    plt.show()


plot_multi(digits.data, digits.target)

##########################################
## Data exploration and first analysis
##########################################

def get_statistics_text(targets):
    # TODO: Write your code here, returning at least the following useful infos:
    # * Label names
    names = targets.target_names
    # * Number of elements per class
    nbr = np.bincount(targets.target)
    return (names,nbr)

# TODO: Call the previous function and generate graphs and prints for exploring and visualising the database
plt.bar(get_statistics_text(digits)[0],get_statistics_text(digits)[1])
plt.show()

##########################################
## Start data preprocessing
##########################################

# Access the whole dataset as a matrix where each row is an individual (an image in our case)
# and each column is a feature (a pixel intensity in our case)
## X = [
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 1 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 2 as a row
#  [Pixel1, Pixel2, ..., Pixel64],  # Image 3 as a row
#  [Pixel1, Pixel2, ..., Pixel64]   # Image 4 as a row
#]

# TODO: Create a feature matrix and a vector of labels
X = np.zeros(digits.data.shape)
for i in range(0,digits.data.shape[0]):
    X[i] = digits.images[i].ravel()
y = digits.target

# Print dataset shape
print(f"Feature matrix shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")


# TODO: Normalize pixel values to range [0,1]
scaler = MinMaxScaler()
scaler.fit(X)
F = scaler.transform(X)  # Feature matrix after scaling

# Print matrix shape
print(f"Feature matrix F shape: {F.shape}. Max value = {np.max(F)}, Min value = {np.min(F)}, Mean value = {np.mean(F)}")

##########################################
## Dimensionality reduction
##########################################


### just an example to test, for various number of PCs
sample_index = 0
original_image = F[sample_index].reshape(8, 8)  # Reshape back to 8×8 for visualization

# TODO: Using the specific sample above, iterate the following:
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Original and PCA Approximations", fontsize=16)

for i, ax in enumerate(axes.flatten()):
    if i == 0:  # Première image = originale
        ax.imshow(original_image, cmap='gray')
        ax.set_title("Original")
    else:  # Autres images = approximations PCA
        pca = PCA(n_components=i)
        X_pca = pca.fit_transform(F)
        reconstructed_vector = pca.inverse_transform(X_pca)
        image_approx = reconstructed_vector[sample_index].reshape(8,8)
        error = np.linalg.norm(original_image-image_approx)
        ax.imshow(image_approx, cmap='gray')
        ax.set_title(f"PC={i}\nErr={error:.6f}")
    ax.axis('off')

plt.tight_layout()
plt.show()




#### TODO: Expolore the explanined variance of PCA and plot
# Create the visualization plot
variance_expliquee = pca.explained_variance_ratio_
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(variance_expliquee) + 1), variance_expliquee, marker='o', linestyle='-')
plt.xlabel("Nombre de Composantes Principales")
plt.ylabel("Ratio de Variance Expliquée")
plt.title("Variance expliquée par les composantes ACP")
plt.grid()
plt.show()



### TODO: Display the whole database in 2D:



pca_2d = PCA(n_components=2)
F_2d = pca_2d.fit_transform(F)

plt.figure(figsize=(8, 6))
labels=digits.target
scatter = plt.scatter(F_2d[:, 0], F_2d[:, 1],c=labels, cmap='tab10', edgecolor='k', s=40)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection 2D de la base de données via ACP")
plt.colorbar(scatter, label='Classe')  # Shows label if classification task
plt.grid(True)
plt.xticks([])
plt.yticks([])
plt.show()


### TODO: Create a 20 dimensional PCA-based feature matrix
pca_20d = PCA(n_components=20)
F_pca = pca_20d.fit_transform(F)



# Print reduced feature matrix shape
print(f"Feature matrix F_pca shape: {F_pca.shape}")

##########################################
## Feature engineering
##########################################
### # Function to extract zone-based features
###  Zone-Based Partitioning is a feature extraction method
### that helps break down an image into smaller meaningful regions to analyze specific patterns.
def extract_zone_features(images):
    '''Break down an 8x8 image in 3 zones: row 1-3, 4-5, and 6-8'''
    zone1 = np.zeros((digits.images.shape[0]))
    zone2 = np.zeros((digits.images.shape[0]))
    zone3 = np.zeros((digits.images.shape[0]))
    for i in range(0,digits.images.shape[0]):
      zone1[i] = np.average(images[i, 0:3, :])  # Lignes 1-3
      zone2[i] = np.average(images[i, 3:5, :])  # Lignes 4-5
      zone3[i] = np.average(images[i, 5:8, :])
    return np.transpose(np.array([zone1,zone2,zone3]))

# Apply zone-based feature extraction
F_zones = extract_zone_features(digits.images)

# Print extracted feature shape
print(f"Feature matrix F_zones shape: {F_zones.shape}")


### Edge detection features

## TODO: Get used to the Sobel filter by applying it to an image and displaying both the original image
# and the result of applying the Sobel filter side by side
sobel_edges = sobel(digits.images[0],mode="constant")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(digits.images[0], cmap='gray')
plt.title('Image originale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sobel_edges, cmap='gray')
plt.title('Filtre de Sobel appliqué')
plt.axis('off')
plt.show()


# TODO: Compute the average edge intensity for each image and return it as an n by 1 array
F_edges = np.zeros((digits.images.shape[0],1))
for i in range(0,digits.images.shape[0]):
  edges = sobel(digits.images[i],mode="constant")
  F_edges[i] = np.mean(edges)


# Print feature shape after edge extraction
print(f"Feature matrix F_edges shape: {F_edges.shape}")

### connect all the features together

# TODO: Concatenate PCA, zone-based, and edge features
F_final = np.hstack((F_pca, F_zones, F_edges))

# TODO: Normalize final features
scaler = StandardScaler()
scaler.fit(F_final)
F_final = scaler.transform(F_final)

# Print final feature matrix shape
print(f"Final feature matrix F_final shape: {F_final.shape}")