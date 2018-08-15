from random import shuffle
from train import deep_test
from scipy import spatial
from preprocess import data_to_adj
import networkx as nx
from others import normalize,similarity,non_existing_similarity,precision,AUC,randwalk,k_fold_AUC,robustness,cost_graph
import math
import numpy as np
from cn import cn_robustness
from jc import jc_robustness


def predict_features (adj):
        # k_fold_AUC(adj)
        robustness(adj)


def common_neighbor (adj):
        cn_robustness(adj)

def jc (adj):
        cn_robustness(adj)

s = ['adjnoun','LesMiserables','polbooks','jazz']
ds = 'karate'
adj = data_to_adj(ds+'.gml')
# predict_features(adj[0])

# common_neighbor(adj[2])

# jc(adj[2])
cost_graph(adj[2])
