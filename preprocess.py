import networkx as nx 
from scipy.sparse import csr_matrix
import numpy as np
import math
import win_unicode_console
win_unicode_console.enable()

def data_to_adj(dataset):
    h=nx.read_gml(dataset,label='id')
    adj_mat_s= nx.adjacency_matrix(h)
    n=adj_mat_s.shape[0]
    print (n)
    adj_mat_d=adj_mat_s.todense()
    x=h.degree()
    degree_matrix = np.identity(n)
    degree_matrix_2 = np.identity(n)
    for i in range(1,n):
        degree_matrix[i-1,i-1] = x[i]
        degree_matrix_2[i-1,i-1] = 1/(x[i]**0.5)
    adj_orig = adj_mat_d
    adj_mat_i=adj_mat_d+np.identity(n)
    input_mat = np.matmul(adj_mat_i,degree_matrix)
    return input_mat, degree_matrix_2, adj_orig

def facebook_adj (dataset):
    h=nx.read_edgelist (dataset)
    h.to_undirected()
    adj_mat_s= nx.adjacency_matrix(h)
    adj_mat_d=adj_mat_s.todense()
    x=h.degree()
    degree_matrix = np.identity(4039)
    degree_matrix_2 = np.identity(4039)

    for i in range(4039):
        degree_matrix[i,i] = x[str(i)]
        degree_matrix_2[i-1,i-1] = 1/(x[str(i)]**0.5)

    adj_mat_i=adj_mat_d+np.identity(4039)
    input_mat = np.matmul(adj_mat_i,degree_matrix)
    return input_mat, degree_matrix_2



    
