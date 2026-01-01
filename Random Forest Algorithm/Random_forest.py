import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Decision_Tree import DecisionTree # i used my own Decision Tree Algorithm 
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common


class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)


# Testing
if __name__ == "__main__":
    # Imports

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # -------- Dataset visualization using PCA --------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=15)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Breast Cancer Dataset (2D PCA Projection)")
    plt.colorbar(label="Class")
    plt.show()
    # ----------------------------------------------------

    clf = RandomForest(n_trees=3, max_depth=10)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)

    #----------------------------------------------------
    tree_counts = [1, 3, 5, 10, 20]
    accuracies = []

    for n in tree_counts:
        clf = RandomForest(n_trees=n, max_depth=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy(y_test, y_pred))

    plt.figure(figsize=(6, 4))
    plt.plot(tree_counts, accuracies, marker='o')
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title("Random Forest: Accuracy vs Number of Trees")
    plt.grid(True)
    plt.show()

    #----------------------------------------------------
    # if you want to se decision boundary visualization, reduce to 2D using PCA
    # Or used a dataset with 2 features directly

