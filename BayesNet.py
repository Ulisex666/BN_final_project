import warnings
import numpy as np
import pandas as pd

class Node:
    '''
    Class defining the nodes that form the Bayesian Network. 
    Name and number of parents and children as attributes.
    Attribute node contains the names of parents and children of current node.
    Current implementation only needs the name of the variable being represented.
    TODO: Define and add CPT
    '''
    def __init__(self, var_name:str):
        """
        Initially the node doesnt have either parents or children. TODO: Add capability to
        define parents and children.
        """
        self.var_name = var_name
        self.node = {'Parents':[], 'Children':[], 'CPT':None}
        self.num_parents = len(self.node['Parents'])
        self.num_children = len(self.node['Children'])
        
    def add_parent(self, parent_name:str): 
        """
        Add parent to node.
        Inputs: Name of parent as a string
        Return: None
        """
        self.node['Parents'].append(parent_name)
        self._update_num_parents()
        return None
        
    def add_child(self, child_name:str): 
        """
        Add child to node.
        Inputs: Name of child as a string
        Return: None
        """
        self.node['Children'].append(child_name)
        self._update_num_children()
        
    def get_parents(self) -> list:
        """
        Returns name of parents of node as a list
        """
        return self.node['Parents']
    
    def _update_num_parents(self):
        """
        Update the property number of parents of node
        """
        self.num_parents = len(self.node['Parents'])
    
    def get_children(self) -> list:
        """
        Returns name of children of node as a list
        """
        return self.node['Children']
    
    def _update_num_children(self):
        """
        Update property number of children of node
        """
        self.num_children = len(self.node['Children'])
    

class BayesNet:
    """
    Main class defining the bayesian network. Attribute name as a string.
    Attribute graph contains the DAG itself as a dict, with the corresponding nodes
    and edges. 
    nodes is a dict, containing the name of the node as a key and the
    value being the actual Node object.
    edges is a list, containing the edges as tuples of two elements, where the first
    one is the parent node, and the second one is the child node.
    """
    def __init__(self, name: str, nodes:dict = {}, edges:list=[]) -> None:
        """
        Name of BN as a string, graphical model contained in graph.  
        """
        self.BN_name = name
        self.graph = {'Nodes':nodes, 'Edges':edges}
        
    def add_node(self, var_name:str):
        """
        Adds node to BN. Takes the name of the node (variable) as a string
        """
        new_node:Node = Node(var_name)
        self.graph['Nodes'][var_name] = new_node

    def add_edge(self, parent_node:str, child_node:str):
        """
        Adds an edge between two nodes in the net. If one or both are missing, it adds
        them to the BN. If the edge is already present, then it raises a warning and 
        DOESN'T add it.
        """
        if (parent_node, child_node) in self.graph['Edges']:
            warnings.warn('Edge already in BN')
            return
        
        if parent_node not in self.graph['Nodes']:
            self.add_node(parent_node)
        
        if child_node not in self.graph['Nodes']:
            self.add_node(child_node)
        
        self.graph['Edges'].append((parent_node, child_node))
        self.graph['Nodes'][parent_node].add_child(child_node)
        self.graph['Nodes'][child_node].add_parent(parent_node)
        
    def get_nodes(self) -> list:
        """
        Returns the nodes present in the net, without any kind of ordering.
        """
        return list(self.graph['Nodes'].keys())
    
    def get_num_nodes(self) -> int:
        """
        Returns number of nodes in the net.
        """
        return len(self.get_nodes())
    
    def get_edges(self) -> list:
        """
        Returns the edges present in the BN as a list of tuples.
        """
        return self.graph['Edges']
    
    def get_orphans(self) -> list:
        """
        Finds all nodes without parents in the BN, and returns their names as a list.
        """
        roots = [node.var_name for node in self.graph['Nodes'].values()
                 if node.num_parents == 0]
        return roots
    
    def has_edge(self, edge: tuple) -> bool:
        """
        Checks if input edge is already present in the BN
        """
        return edge in self.graph['Edges']
    
    def get_children(self, node:str) -> list:
        """
        Finds all children of the specified nodel. Returns a list containing their names.
        """
        return self.graph['Nodes'][node].get_children()
    
    def get_num_children(self, node:str) -> int:
        """
        Returns the number of children of the specified node
        """
        return self.graph['Nodes'][node].num_children
    
    def get_parents(self, node:str) -> list:
        """
        Returns the parents of the specified node as a list, containing the names as strings
        """
        return self.graph['Nodes'][node].get_parents()
    
    def get_num_parents(self, node:str) -> int:
        """
        Returns the number of parents of the specified node
        """
        return self.graph['Nodes'][node].num_parents
    
    def top_sort(self):
        """
        Topological sorting of nodes in the BN using Kahns algorithm. It also detects loops in the BN
        """
        roots = self.get_orphans()
        sorting = []
        while roots:
            node = roots.pop(0)
            sorting.append(node)
            
            for child in self.get_children(node):
                if self.get_num_parents(child) - 1 == 0:
                    roots.append(child) 
                    
        if len(sorting) < self.get_num_nodes():
            warnings.warn('Cycle detected!')
            
        return sorting
        
