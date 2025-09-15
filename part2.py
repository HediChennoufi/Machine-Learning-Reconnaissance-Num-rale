
# Split the data
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Load processed feature matrix and labels
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import svm
import time

# TODO: Add any util functions you may have from the previous script
def get_statistics_text(targets, target_names=None):
    if hasattr(targets, 'target'):
        # targets est l'objet complet comme 'digits'
        names = targets.target_names
        counts = np.bincount(targets.target)
    else:
        # targets est un vecteur (ex: y_train)
        names = target_names if target_names is not None else np.unique(targets)
        counts = np.bincount(targets)
    return (names, counts)

# TODO: Load the raw data
digits = load_digits()

X = np.zeros(digits.data.shape)
for i in range(0,digits.data.shape[0]):
    X[i] = digits.images[i].ravel()
y = digits.target

print(f"Feature matrix X shape: {X.shape}. Max value = {np.max(X)}, Min value = {np.min(X)}, Mean value = {np.mean(X)}")
print(f"Labels shape: {y.shape}")

##### --------------------------------------------------------------
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets -----------------------


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


### If you want, you could save the data, this would be a good way to test your final script in the same evaluation mode as what we will be doing
np.save("X_train.npy", X_train)
np.save("test_data.npy", X_test)
np.save("y_train.npy", y_train)
np.save("test_label.npy", y_test)
####

# TODO: Print dataset split summary...
print("===== Dataset Split Summary =====")
print(f"Total samples       : {len(y)}")
print(f"Training samples    : {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
print(f"Testing samples     : {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")

# Nombre d'exemples par classe
names, original_counts = get_statistics_text(digits)
_, train_counts = get_statistics_text(y_train, names)
_, test_counts = get_statistics_text(y_test, names)

print("\nNumber of samples per class:")
print("Class\tOriginal\tTrain\tTest")
for i, name in enumerate(names):
    print(f"{name}\t{original_counts[i]}\t\t{train_counts[i]}\t{test_counts[i]}")




# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)
class_names = digits.target_names
names, original_counts = get_statistics_text(digits)
_, train_counts = get_statistics_text(y_train, class_names)
_, test_counts = get_statistics_text(y_test, class_names)
x = np.arange(len(class_names))
width = 0.25
plt.figure(figsize=(10, 6))
bar1 = plt.bar(x - width, original_counts/original_counts, width, label='Original')
plt.bar(x, train_counts/original_counts, width, label='Train')
plt.bar(x + width, test_counts/original_counts, width, label='Test')
for bar in bar1:
      height = bar.get_height()
      x_center = bar.get_x() + bar.get_width() / 2
      y = height * 0.2
      plt.hlines(y, bar.get_x(), bar.get_x() + bar.get_width(), colors='red', linestyles='dashed', linewidth=1)

plt.xticks(x, class_names)
plt.xlabel("Classe")
plt.ylabel("Nombre d'exemples")
plt.title("Distribution des classes : original, train, test")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()


# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.

##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
#
# The PCA shall not be fitted using the test set.
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and
# edge features.
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
#
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method.
#
# Check this documentation to understand how to work with these things
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self # No fitting needed for this processing

    def transform(self, X):
       sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
       return sobel_feature

# TODO: Fill out the useful code for this class
class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self # No fitting needed for this processing

    def transform(self, X):
        zone_features = []
        for img_vector in X:
          img = img_vector.reshape((8,8))
          top = np.mean(img[:3, :])  # Top region
          middle = np.mean(img[3:5, :])  # Middle region
          bottom = np.mean(img[5:, :])  # Bottom region
          zone_features.append([top, middle, bottom])
        return np.array(zone_features)

# TODO: Create a single sklearn object handling the computation of all features in parallel
all_features = FeatureUnion([('pca', PCA(n_components=20)), ('zones', ZonalInfoPreprocessing()), ('sobel', EdgeInfoPreprocessing())])

F = all_features.fit(X_train,y_train).transform(X_train)
scaler = StandardScaler()
scaler.fit(F)
F = scaler.transform(F)
# Let's make sure we have the number of dimensions that we expect!
print("Nb features computed: ", F.shape[1])

# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to
# avoid forgetting a scaling, or a feature, or ...
#
# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
clf = Pipeline([('features', all_features), ('classifier', SVC(kernel='linear'))])

##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline
clf.fit(X_train, y_train)

# TODO: Predict the outcome of the learned algorithm on the train set and then on the test set
predict_test = clf.predict(X_test)
predict_train = clf.predict(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))


# TODO: Look at confusion matrices from sklearn.metrics and
# 1. Display a print of it
disp = confusion_matrix(y_test, predict_test)
disp = ConfusionMatrixDisplay(confusion_matrix=disp)
disp.plot()
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
# 3. Report on how you understand the results

# TODO: Work out the following questions (you may also use the score function from the classifier)
print("\n Question: How does changing test_size influence accuracy?")
print("Try different values like 0.1, 0.3, etc., and compare results.\n")

##########################################
## Hyper parameter tuning and CV
##########################################
# TODO: Change from the linear classifier to an rbf kernel*
clf = Pipeline([('features', all_features), ('classifier', SVC(kernel='rbf'))])
# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
parametres = {
    'features__pca__n_components': [10, 20, 30],
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.001]
}

# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested
dict_grid=dict(features__pca__n_components = [10, 20, 30], classifier__C=[0.1, 1, 10], classifier__gamma= ['scale', 'auto', 0.01, 0.001])
# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters
grid_search = GridSearchCV(clf, param_grid=dict_grid, cv=10, verbose=10, n_jobs=-1) #, verbose=10)
grid_search.fit(X_train, y_train)
# TODO: fit the grid search CV and
# 1. Check the results
# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
# 3. Retrain on the whol train set, and evaluate on the test set
# 4. Answer the questions below and report on your findings

print(" K-Fold Cross-Validation Results:")
print(f"- Best Cross-validation score: {grid_search.best_score_}")
print(f"- Best parameters found: {grid_search.best_estimator_}")
best_model = grid_search.best_estimator_
predict_test = best_model.predict(X_test)
predict_train = best_model.predict(X_train)
print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))
disp = confusion_matrix(y_test, predict_test)
disp = ConfusionMatrixDisplay(confusion_matrix=disp)
disp.plot()
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()


#####
print("\n Question: What happens if we change K from 5 to 10?")
print("Test different K values and compare the accuracy variation.\n")

##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy

# Print OvO results
start_ovo = time.time()
clf = OneVsOneClassifier(svm.SVC())
clf.fit(X_train, y_train)
ovo_time = time.time() - start_ovo
print(" One-vs-One (OvO) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print(f"ovo time : {ovo_time:.4f}s")
print("- Impact: Suitable for small datasets but increases complexity.")

print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")
###################
# TODO:  One-vs-Rest (OvR) Classification


# Print OvR results
start_ovr = time.time()
clf = OneVsRestClassifier(svm.SVC())
clf.fit(X_train, y_train)
ovr_time = time.time() - start_ovr
print(" One-vs-Rest (OvR) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print(f"ovr time : {ovr_time:.4f}s")
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")

print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")
########

