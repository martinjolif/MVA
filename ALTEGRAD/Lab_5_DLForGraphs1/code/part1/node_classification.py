"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network
nx.draw_networkx(G, node_color=y)
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, walk_length, n_walks, n_dim) # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
LR = LogisticRegression()
LR.fit(X_train, y_train)
LR.predict(X_test)
y_pred = LR.predict(X_test)
print("Accuracy score Logistic regression after Deepwalk embeddings:", accuracy_score(y_test, y_pred))


############## Task 8
# Generates spectral embeddings

SE = SpectralEmbedding(n_components=2, affinity="precomputed")
spectral_embeddings = SE.fit_transform(nx.to_numpy_array(G))


X_train_spectral = spectral_embeddings[idx_train,:]
X_test_spectral = spectral_embeddings[idx_test,:]

y_train_spectral = y[idx_train]
y_test_spectral = y[idx_test]

LR_spectral = LogisticRegression()
LR_spectral.fit(X_train_spectral, y_train_spectral)
LR_spectral.predict(X_test_spectral)
y_pred_spectral = LR_spectral.predict(X_test_spectral)
print("Accuracy score Logistic regression after spectral embedding:", accuracy_score(y_test, y_pred_spectral))

#Logistic regression after Deepwalk embeddings gives a better accuracy score