import numpy as np
import re

from grakel import Graph
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def load_file(filename):
    labels = []
    docs =[]

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])
    
    return docs,labels  


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs): 
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])
    
    return preprocessed_docs
    
    
def get_vocab(train_docs, test_docs):
    vocab = dict()
    
    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
        
    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))

import networkx as nx
import matplotlib.pyplot as plt

# Task 11

def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    for idx, doc in enumerate(docs):
        G = nx.Graph()
        ##################
        # your code here #
        ##################
        for word in doc:
            if word in vocab:
                G.add_node(word, word=word)
        for i in range(len(doc) - window_size + 1):
            for k in range(1, window_size):
                G.add_edge(doc[i], doc[i + k])
        
        graphs.append(G)
    
    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3) 
G_test_nx = create_graphs_of_words(test_data, vocab, 3)

print("Example of graph-of-words representation of document")
print("Original sentence:", train_data[3])
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
#from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, SvmTheta, Propagation, VertexHistogram, NeighborhoodHash, ShortestPath, GraphletSampling, CoreFramework
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Task 12

# Transform networkx graphs to grakel representations

G_train = graph_from_networkx(G_train_nx, node_labels_tag="word") # your code here #
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")# your code here #
# Initialize a Weisfeiler-Lehman subtree kernel
WL = WeisfeilerLehman()# your code here #

# Construct kernel matrices
K_train = WL.fit_transform(G_train) # your code here #
K_test = WL.transform(G_test) # your code here #

#Task 13

# Train an SVM classifier and make predictions

##################
# your code here #
##################
clf_WL = SVC(kernel='precomputed')
clf_WL.fit(K_train, y_train)
y_pred_WL = clf_WL.predict(K_test)

# Evaluate the predictions
print("Accuracy for a Weisfeiler-Lehman subtree kernel:", accuracy_score(y_test, y_pred_WL))


#Task 14


##################
# your code here #
##################


NH = NeighborhoodHash()
G_train = graph_from_networkx(G_train_nx, node_labels_tag="word")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")

# Construct kernel matrices
K_train = NH.fit_transform(G_train)
K_test = NH.transform(G_test)

clf_NH = SVC(kernel='precomputed')
clf_NH.fit(K_train, y_train)
y_pred_NH = clf_NH.predict(K_test)

# Evaluate the predictions
print("Accuracy for the NeighborhoodHash kernel:", accuracy_score(y_test, y_pred_NH))

SP = ShortestPath()
G_train = graph_from_networkx(G_train_nx, node_labels_tag="word")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")

# Construct kernel matrices
K_train = SP.fit_transform(G_train)
K_test = SP.transform(G_test)

clf_SP = SVC(kernel='precomputed')
clf_SP.fit(K_train, y_train)
y_pred_SP = clf_SP.predict(K_test)

# Evaluate the predictions
print("Accuracy for the shortest path kernel:", accuracy_score(y_test, y_pred_SP))


ST = SvmTheta()
G_train = graph_from_networkx(G_train_nx, node_labels_tag="word")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")

# Construct kernel matrices
K_train = ST.fit_transform(G_train)
K_test = ST.transform(G_test)

clf_ST = SVC(kernel='precomputed')
clf_ST.fit(K_train, y_train)
y_pred_ST = clf_ST.predict(K_test)

# Evaluate the predictions
print("Accuracy for the SvmTheta kernel:", accuracy_score(y_test, y_pred_ST))



P = Propagation()
G_train = graph_from_networkx(G_train_nx, node_labels_tag="word")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")

# Construct kernel matrices
K_train = P.fit_transform(G_train)
K_test = P.transform(G_test)

clf_P = SVC(kernel='precomputed')
clf_P.fit(K_train, y_train)
y_pred_P = clf_P.predict(K_test)

# Evaluate the predictions
print("Accuracy for the propagation kernel:", accuracy_score(y_test, y_pred_P))


GS = GraphletSampling()
G_train = graph_from_networkx(G_train_nx, node_labels_tag="word")
G_test = graph_from_networkx(G_test_nx, node_labels_tag="word")

# Construct kernel matrices
K_train = GS.fit_transform(G_train)
K_test = GS.transform(G_test)

clf_GS = SVC(kernel='precomputed')
clf_GS.fit(K_train, y_train)
y_pred_GS = clf_GS.predict(K_test)

# Evaluate the predictions
print("Accuracy for the graphlet-sampling kernel:", accuracy_score(y_test, y_pred_GS))


### The kernel with the best accuracy is the Shortest-path kernel with an accuracy of 0.966