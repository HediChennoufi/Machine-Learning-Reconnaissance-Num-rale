
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.ndimage import sobel
from sklearn.model_selection import train_test_split

# Chargement des données
digits = load_digits()
X = digits.data
y = digits.target
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# TODO: Create your first fully connected neural network with the following caracteristics:
# * 1 input layer
# * 1 hidden layer using 32 neurons
# * Input and hidden layers shall have relu activation functions
# * Output layer is activated with softmax
# * As input, we consider the pixels directly -- we'll look later into using engineered features
model = keras.Sequential([keras.layers.Input(shape=(64,)), keras.layers.Dense(32, activation='relu'),keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(X_train, y_train,validation_split=0.2,epochs=30,batch_size=32,verbose=1)




# Extract loss values
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot loss vs epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='s')
plt.xlabel("Epochs")
plt.yscale("log")  # Apply log scale to the y-axis
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.grid()
plt.show()

# Save the trained model
model.save("digit_classifier_model.keras")  # Native Keras format

# TODO:
# For the remaining time of the class, find a way to combine preprocessing + learning algorithm to achieve the best possible scores.
# Your final tuned model should be written in the test script provided for us to test during the oral exam.
#

#Le meilleur obetnu est le code du du cvs rbg avec 99% de réussite


# Enjoy, and good luck!