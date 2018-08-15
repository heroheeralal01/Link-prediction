import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy import spatial
from train import deep_test
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def normalize (n):
    for i in n:
        max = np.amax(i)
        for j in range(len(i)):
            i[j] = i[j] / max
    return  n

def similarity (feature_mat):
    result_mat = []
    for k in range (len(feature_mat)):
        result_mat.append([])
        for j in range (len(feature_mat)):
            result_mat[k].append(1-spatial.distance.cosine(feature_mat[k], feature_mat[j]))
    return result_mat

def non_existing_similarity (non_edges , result_mat):
    for k in range (len(non_edges)):
        non_edges[k]= list(non_edges[k])
        non_edges[k].append(result_mat[non_edges[k][0]][non_edges[k][1]])
    non_edges.sort(key=lambda x: x[2],reverse=True)
    return non_edges

def precision (testdata, non_edges, length):
    non_edges=non_edges[0:length]
    for k in range (len(non_edges)):
        non_edges[k].pop()
        non_edges[k]=tuple(non_edges[k])
    prec=len(set(testdata).intersection(set(non_edges)))/len(set(testdata))
    return prec

def AUC (test_data,feature_mat,non_edges):
    great = 0
    equal = 0
    result = 0
    for m in range (len(test_data)):
        sim = 1 - spatial.distance.cosine(feature_mat[test_data[m][0]], feature_mat[test_data[m][1]])
        for c in range (len(non_edges)):
            if sim > non_edges[c][2] :
                great = great+1
            if sim == non_edges[c][2] :
                equal = equal + 1
        result = (result + (great+ (0.5*equal)))/(len(test_data)*len(non_edges))
    return result

def randwalk (adj,T,c,degree):
    P =np.zeros(shape = (len(adj)))
    S =np.zeros(shape = (len(adj), len(adj)))
    p =np.zeros(shape = (len(adj), len(adj)))
    for i in range(len(adj)):
        P[i]=0
        for j in range(len(adj)):
            S[i][j]= 1 - spatial.distance.cosine(T[i], T[j])
            p[i][j]=S[i][j]*adj[i,j]
            P[i] = P[i]+p[i][j]
    Sp = np.zeros(shape = (len(adj), len(adj)))
    costp = 1
    cost = 0.001
    nodes = len(adj)
    while costp > cost and cost > 0.00001:
        print ("IN WHILE!" , ((Sp - S) ** 2).mean(axis=None))
        costp = ((Sp - S) ** 2).mean(axis=None)
        Sp = np.array(S,copy=True)
        for i in range(nodes):
            for j in range(nodes):
                if i != j :
                    const = c / ((degree[j]*P[i]) + (degree[i]*P[j]))
                    sum = 0
                    for k in range(len(p[i])):
                        for l in range(len(p[j])):
                            if p[k][i] != 0 and p[l][j] !=0 :
                                sum = sum + (p[k][i] + p[l][j])*S[k][l]
                    S[i][j] = const*sum
                else :
                    S[i][j]=1
        cost = ((Sp - S) ** 2).mean(axis=None)                    
    return S


def k_fold_AUC (adj):
    auc = 0
    X = np.array(list(nx.Graph(adj).edges))
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    nodes = list (range(len(adj)))
    for train_index, test_index in kf.split(X):
        print ("lol")
        X_train, X_test = X[train_index], X[test_index]
        X_train = list(tuple(map(tuple, X_train)))
        Graph = nx.Graph()
        Graph.add_nodes_from(nodes)
        Graph.add_edges_from(X_train)
        non_existing_edges = list(nx.non_edges(Graph))
        train_graph = nx.adjacency_matrix(Graph).todense()
        feature_mat=deep_test(train_graph,[16,8],5000,1)
        feature_mat=normalize(feature_mat)
        result_mat = similarity(feature_mat)
        non_existing_edges = non_existing_similarity(non_existing_edges,result_mat)
        # print (non_existing_edges)
        auc = auc + AUC(list(tuple(map(tuple, X_test))),feature_mat,non_existing_edges)
    print ("AUC : ", auc/10)


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
            feature_mat=deep_test(train_graph,[16,8],5000,1)
            feature_mat=normalize(feature_mat)
            result_mat = similarity(feature_mat)
            non_existing_edges = non_existing_similarity(non_existing_edges,result_mat)
            auc = auc + AUC(list(tuple(map(tuple, X_test))),feature_mat,non_existing_edges)
    return (auc/10)


def robustness (adj):
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
    f= open("results.txt","w+")
    f.write(x,y)
    f.close()
    plt.plot(y,x)
    plt.show()


def cost_graph (adj):
    cost = []
    features =[]
    for i in range(8,50):
        features.append(i)
        cost.append(deep_test(adj,[16,i],5000,0))
    plt.plot(features,cost)
    plt.show()
    