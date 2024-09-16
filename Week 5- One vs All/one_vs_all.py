import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

# Generating a sample dataset
X, y = make_classification(n_samples=300, n_features=5, n_classes=3, n_informative=4, n_redundant=1, random_state=42)
# Convert to pandas DataFrame for better visualization
df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(X.shape[1])])
df['Class'] = y

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Binarize the output labels
y_binarized = label_binarize(y, classes=[0, 1, 2])
n_classes = y_binarized.shape[1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.3, random_state=42)

# One-vs-all approach using kNN
knn = KNeighborsClassifier(n_neighbors=3)
ovr = OneVsRestClassifier(knn)
ovr.fit(X_train, y_train)

# Predicting probabilities
y_score = ovr.predict_proba(X_test)

# Plotting ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotting the ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve for class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-All ROC Curve using kNN Classifier')
plt.legend(loc='lower right')
plt.show()
