import pandas as pd
import numpy as pd
from itertools import combinations
from collections import deque
from BayesNet.utils import *
from BayesNet.BayesNet import *
     
def chow_liu(df: pd.DataFrame, bn_name: str, root:str = ''):
    """
    Function for creating a BayesNet Tree based on the Chow Liu algorithm.
    Inputs: Database as pandas Dataframe, name of BayesNet as string and name of root node as string.
    
    Returns: BayesNet object based on database
    """
    
    vars_names = df.columns.to_list()
    mi_edges = []
    
    # Get mutual info between all pairs of variables
    for var1, var2 in combinations(vars_names, 2):
        mutual_info = get_mutual_info(df, var1, var2)
        mi_edges.append((var1, var2, mutual_info))
        
    # Order list based on mutual info (maximizing its value)
    # Ordenar por información mutua descendente
    mi_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Use Kruskal algorithm to generate Maximum Spanning Tree
    # Use union-find algorithm to detect possible loops
    
    parents_dict = {var: var for var in vars_names}
    
    def find(v):
        # Find root of tree where v is located, recursively going back
        # on the dict keys
        while parents_dict[v] != v:
            parents_dict[v] = parents_dict[parents_dict[v]]
            # Only keep latest added node. Path comprehension
            v = parents_dict[v]
        return v
    
    def union(u, v):
        # Find root of both variables
        root_u, root_v = find(u), find(v)
        # Only add edge if theres not cycle
        if root_u != root_v:
            parents_dict[root_v] = root_u
            return True
        return False
    
    # Create undirected tree, then decide root and create edges from there
    undirected_tree = {var:[] for var in vars_names}
    for u,v, _ in mi_edges:
        if union(u,v):
            # Since is undirected, edges go both ways
            undirected_tree[u].append(v)
            undirected_tree[v].append(u)
            
    # Choose root
    if not root:
        root = mi_edges[0][0] # Select edge with biggest MI as default
    elif root not in vars_names:
        raise ValueError(f"Root node '{root}' not in df variables!")
    
    # BFS to give direction to tree, starting from root
    visited = set()
    queue = deque([root])
    directed_edges = []
    
    while queue:
        current = queue.popleft()
        visited.add(current)
        for neigbor in undirected_tree[current]:
            if neigbor not in visited:
                # Create edge going from current node to neigbor
                directed_edges.append((current, neigbor))
                queue.append(neigbor)
                
    # Create BN to be filled with nodes and directed edges
    bn = BayesNet(bn_name) 
       
    # Create nodes
    for var in vars_names:
        bn.add_node(var)
        
    # Add edges to create Maximun Spanning Tree, starting from root 
    for edge in directed_edges:
        bn.add_edge(edge)
            
    return bn