import networkx as nx
from preprocess import data_to_adj
import numpy as np
from others import similarity
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def similarity_common (non_edges,result_mat):
    for k in range (len(non_edges)):
        non_edges[k]= list(non_edges[k])
        non_edges[k].append(result_mat[non_edges[k][0]][non_edges[k][1]])
    non_edges.sort(key=lambda x: x[2],reverse=True)
    return non_edges

def AUC (test_data,feature_mat,non_edges):
    great = 0
    equal = 0
    result = 0
    for m in range (len(test_data)):
        sim = feature_mat[test_data[m][0]][test_data[m][1]]
        for c in range (len(non_edges)):
            if sim > non_edges[c][2] :
                great = great+1
            if sim == non_edges[c][2] :
                equal = equal + 1
        result = (result + (great+ (0.5*equal)))/(len(test_data)*len(non_edges))
    return result

def k_fold_robust (X,nodes):
    auc = 0
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        X_train = list(tuple(map(tuple, X_train)))
        Graph = nx.Graph()
        Graph.add_nodes_from(nodes)
        Graph.add_edges_from(X_train)
        non_existing_edges = list(nx.non_edges(Graph))
        train_graph = nx.adjacency_matrix(Graph).todense()
        feature_mat=cn(train_graph)
        non_existing_edges = similarity_common (non_existing_edges,feature_mat)
        auc = auc + AUC(list(tuple(map(tuple, X_test))),feature_mat,non_existing_edges)
    return (auc/10)


def cn_robustness (adj):
    y=[]
    x=[]
    Graph = nx.Graph(adj)
    edges = np.array(list(Graph.edges))
    nodes = list (range(len(adj)))
    np.random.shuffle(edges)
    et = edges
    nonedges = np.array(list(nx.non_edges(Graph)))
    np.random.shuffle(nonedges)
    for i in np.arange(0.6,0.8,0.2):
        etnew = np.array(et,copy=True)
        np.random.shuffle(etnew)
        etnew = etnew[:int(i*(len(etnew)))]
        x.append(k_fold_robust (etnew , nodes))
        y.append(i-1)
    for i in np.arange(0,1.2,0.2):
        etnew = np.array(et,copy=True)
        etnew = np.concatenate((etnew,nonedges[0:int(len(etnew)*i)]),axis=0)
        x.append(k_fold_robust (etnew , nodes))
        y.append(i)
    print (x)
    print (y)
    plt.plot(y,x)
    plt.show()

def cn (adj) :
    Graph = nx.Graph(adj)
    common = np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common[i][j] = len((sorted((nx.common_neighbors(Graph,i,j)))))
    similarity_mat = similarity(common)
    return common

