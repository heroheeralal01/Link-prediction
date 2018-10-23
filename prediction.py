from random import shuffle
from train import deep_test
from scipy import spatial
from preprocess import data_to_adj
import networkx as nx
from others import normalize
import math

def pred_features (adj):
    graph = list(nx.Graph(adj).edges)
    shuffle(graph)
    graph_copy = graph
    length = math.floor(len(graph)/20)
    precision, i = 0, 0
    auc = 0
    while i < len(graph):
        print ("\nIteration : " ,math.floor(i/length))
        test_data = graph[i:i+length]
        graph = graph[0:i]+graph[i+length:len(graph)]
        Graph = nx.Graph(graph)
        print ("no of edges : ",Graph.number_of_edges())
        non_edges = list(nx.non_edges(Graph))
        print ("non++++++++++++++++++++++",len(non_edges))
        train_graph = nx.adjacency_matrix(Graph)
        train_graph=train_graph.todense()

        feature_mat=deep_test(train_graph,[16,8],5000,1)
        feature_mat=normalize(feature_mat)
        result_mat = []
        i=i+length+1
        for k in range (len(feature_mat)):
            result_mat.append([])
            for j in range (len(feature_mat)):
                result_mat[k].append(1-spatial.distance.cosine(feature_mat[k], feature_mat[j]))

        for k in range (len(non_edges)):
            non_edges[k]= list(non_edges[k])
            nmm=non_edges[k][1]
            non_edges[k].append(result_mat[non_edges[k][0]][non_edges[k][1]])

        great = 0
        equal = 0
        result = 0

        print (non_edges)
        for m in range (len(test_data)):
            sim = 1 - spatial.distance.cosine(feature_mat[test_data[m][0]], feature_mat[test_data[m][1]])
            for c in range (len(non_edges)):
                if sim > non_edges[c][2] :
                    great = great+1
                if sim == non_edges[c][2] :
                    equal = equal + 1
            result = (result + (great+ (0.5*equal)))/(len(test_data)*len(non_edges))
        auc = result+auc
        non_edges=non_edges[0:length]
        for k in range (len(non_edges)):
            non_edges[k].pop()
            non_edges[k]=tuple(non_edges[k])
        prec=len(set(test_data).intersection(set(non_edges)))/len(set(test_data))
        precision +=prec
        graph=graph_copy
    print ("Precision : ",precision/20)
    print ("AUC : ", auc/20)

s = ['adjnoun','LesMiserables','polbooks','jazz']
ds = 'usair97'
adj = data_to_adj(ds+'.gml')
pred_features(adj[0])













