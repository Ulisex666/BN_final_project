import warnings
import pandas as pd
from itertools import product
from BayesNet.utils import *

"""
Defining classes to be used in learning a Bayesian Network from data.
"""
class Node:
    '''
    Class defining the nodes that form the Bayesian Network. 
    Name and number of parents and children as attributes.
    Attribute node contains the names of parents and children of current node.
    Can specify multiple parents and children.
    '''
    def __init__(self, var_name:str, var_values:list[str] = [], parents:list[str] = [], children:list[str] = []):
        """
        Initialize node. By default, it's variable doesn't have any set values, and the node doesn't have
        any parents or children. They can be added as a list of strings, or numeric values for the variable values
        """
        self.var_name = var_name
        self.var_values = [str(value) for value in var_values] 
        self.parents = [parent for parent in parents]
        self.children = [child for child in children]
        self.num_parents = len(self.parents)
        self.num_children = len(self.children)
        
    def add_parent(self, parent_name:str): 
        """
        Add parent to node.
        Inputs: Name of parent as a string
        Return: None
        """
        self.parents.append(parent_name)
        self._update_num_parents()
        return None
        
        
    def add_child(self, child_name:str): 
        """
        Add child to node.
        Inputs: Name of child as a string
        Return: None
        """
        self.children.append(child_name)
        self._update_num_children()
        return None
    
    def add_var_val(self, var_val):
        """
        Adds specified value to list of values taken by node, as a string.
        """
        if var_val is not str:
            str_val = str(var_val)
            self.var_values.append(str_val)
        else:
            self.var_values.append(var_val) # type: ignore
        
    def del_child(self, child_name:str):
        """
        Deletes child from children list.
        Inputs: Child to be killed name.
        Returns: None
        """
        if not child_name in self.children:
            raise KeyError(f'{child_name} not in children')
        self.children.remove(child_name)
        return None
        
    def del_parent(self, parent_name:str):
        """
        Deletes parent from parents list.
        Inputs: Parent to be killed.
        Returns: None
        """
        if not parent_name in self.parents:
            raise KeyError(f'{parent_name} not in parents')
        self.parents.remove(parent_name)
        return None
        
    def get_parents(self) -> list[str]:
        """
        Returns name of parents of node as a list
        """
        return self.parents
    
    def _update_num_parents(self):
        """
        Update the property number of parents of node
        """
        self.num_parents = len(self.parents)
        return None
    
    def get_children(self) -> list[str]:
        """
        Returns name of children of node as a list
        """
        return self.children
    
    def _update_num_children(self):
        """
        Update property number of children of node
        """
        self.num_children = len(self.children)
        return None
        
    def get_var_values(self) -> list:
        """
        Get the values the variable can take as a list
        """
        return self.var_values
    
class CPT:
    def __init__(self, var_name: str, var_values: list[str], 
                 parent_values_dict: dict[str, list[str]]):
        """
        Creates a CPT for a given variable. It know what values the variable can take,
        its parents and the values the parents can take
        """
        self.var = var_name
        self.var_values = var_values
        self.parents = list(parent_values_dict.keys())
        self.parent_values_dict = parent_values_dict
        
        self.table = self._build_empty_table()
    
    def _build_empty_table(self):
        """
        Create an empty CPT for the variable, where all default values are None
        """
        parent_combinations = list(product(*[self.parent_values_dict[p] for p in self.parents]))
        default_prob = None

        # Dataframe, where every row is a combination of the values the parents can take,
        # and the probability the variable will take a certain value given its parents values
        rows = []
        for parent_vals in parent_combinations:
            row = dict(zip(self.parents, parent_vals))
            for val in self.var_values:
                row[f'P({self.var}={val})'] = default_prob # type: ignore
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def to_dataframe(self):
        """
        Return CPT as a dataframe
        """
        return self.table
    
    def update_from_data(self, df: pd.DataFrame):
        """
        Update CPT based on observations in DB. Assumes a discretized Db without missing values.
        Values without observations get assigned 0.
        """
        parent_cols = self.parents
        child_col = self.var
        # Values are converted to strings to avoid type problems
        child_vals = [str(v) for v in self.var_values]

        # Vaues in DB are converted to string for child and parents
        df_str = df.copy()
        df_str[child_col] = df_str[child_col].astype(str)
        for p in parent_cols:
            df_str[p] = df_str[p].astype(str)

        # Case for when variable doesn't have parents, root node
        if not parent_cols:
            # Count of how many time the variable has taken a certain value
            counts = df_str[child_col].value_counts(normalize=True)

            # Divides every count over the total number of observations, and returns it as a prob ability
            for i, row in self.table.iterrows():
                for val in child_vals:
                    col = f'P({child_col}={val})'
                    self.table.at[i, col] = counts.get(val, 0.0)
            return

        # When variable has parents: P(var | parents)
        
        # Count of how many times the variable takes a certain value for a given parent combination
        counts = (
            df_str.groupby(parent_cols + [child_col])
            .size()
            .reset_index(name='count')
        )

        # Count of how many times the parents have taken a certain value combination
        totals = (
            df_str.groupby(parent_cols)
            .size()
            .reset_index(name='total')
        )

        # Merge the two dataframes and calculate probabilities
        merged = pd.merge(counts, totals, on=parent_cols)
        merged['P'] = merged['count'] / merged['total']

        # Create lookup dictionary for probabilities (parent_combo, child_val) -> P
        prob_lookup = {
            (tuple(row[p] for p in parent_cols), row[child_col]): row['P']
            for _, row in merged.iterrows()
        }

        # Populate CPT from lookup probablities table
        for i, row in self.table.iterrows():
            parent_key = tuple(str(row[p]) for p in parent_cols)
            for val in child_vals:
                col = f'P({child_col}={val})'
                prob = prob_lookup.get((parent_key, val), 0.0)
                self.table.at[i, col] = prob

    
    
    def __str__(self):
        var = self.var
        parents = self.parents
        sep = '-'*40
        return (sep + '\n' + f'Variable "{var}" with parents ' + f'"{','.join(str(p) for p in parents)}"'
                + '\n' + sep +  '\n'  + self.table.to_string(index=False))


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
        self.CPTs = {}
        
    def add_node(self, var_name:str):
        """
        Adds node to BN. Takes the name of the node (variable) as a string
        """
        new_node:Node = Node(var_name)
        self.graph['Nodes'][var_name] = new_node
        return None

    def add_edge(self, edge:tuple[str, str]):
        """
        Adds an edge between two nodes in the net. If one or both are missing, it adds
        them to the BN. If the edge is already present, then it raises a warning and 
        DOESN'T add it.
        Input: Edge as a tuple, e.g., ('A', 'B').
        Returns: None
        """
        parent_node = edge[0]
        child_node = edge[1]
        
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
        return None
        
    def del_edge(self, edge:tuple[str, str]):
        """
        Deletes an edge from the network. Returns a warning if the edge or any
        of the nodes is missing from the net. It DOESN'T add or delete any edges in
        that case.
        Inputs: Edge as a tuple, e.g., ('A', 'B')
        Returns: None
        """
        parent_node = edge[0]
        child_node = edge[1]
        
        if not (parent_node, child_node) in self.graph['Edges']:
            warnings.warn('Edge not present in BN')
            return
        
        if not parent_node in self.graph['Nodes']:
            warnings.warn(f'{parent_node} not present in BN')
            return
        
        if not child_node in self.graph['Nodes']:
            warnings.warn(f'{child_node} not present in BN')
            return
        
        self.graph['Edges'].remove((parent_node, child_node))
        
        self.graph['Nodes'][parent_node].del_child(child_node)
        self.graph['Nodes'][child_node].del_parent(parent_node)
        return None
        
    def reverse_edge(self, edge:tuple[str, str]):
        """
        Reverses an edge from the network. Returns a warning if the edge or any
        of the nodes is missing from the net. It DOESN'T add or delete any edges in
        that case.
        Inputs: Edge as a tuple, e.g., ('A', 'B')
        Returns: None
        """
        old_parent_node = edge[0]
        old_child_node = edge[1]
        
        if not (old_parent_node, old_child_node) in self.graph['Edges']:
            warnings.warn('Edge not present in BN')
            return
        
        if not old_parent_node in self.graph['Nodes']:
            warnings.warn(f'{old_parent_node} not present in BN')
            return
        
        if not old_child_node in self.graph['Nodes']:
            warnings.warn(f'{old_child_node} not present in BN')
            return
        
        self.graph['Edges'].remove((old_parent_node, old_child_node))
        self.graph['Nodes'][old_parent_node].del_child(old_child_node)
        self.graph['Nodes'][old_child_node].del_parent(old_parent_node)
        
        new_parent_node = old_child_node
        new_child_node = old_parent_node
        
        self.graph['Edges'].append((new_parent_node, new_child_node))
        self.graph['Nodes'][new_parent_node].add_child(new_child_node)
        self.graph['Nodes'][new_child_node].add_parent(new_parent_node)
        return None
    
    def add_var_values(self, var_name:str, var_values:list):
        """
        Adds all possible values a node can take, take from a list
        """
        for value in var_values:
            self.graph['Nodes'][var_name].add_var_val(value)
            
    def get_vars_values(self, vars_name:list[str]) -> dict:
        """
        Gets all values a variable or a list of variables can take, as a dictionary
        where the key is the variable and the value is a list of the values it can take
        """
        values = {var:None for var in vars_name}
        for var in vars_name:
            values[var] = self.graph['Nodes'][var].get_var_values()
        return values
        
    def get_nodes(self) -> list[str]:
        """
        Returns the nodes present in the net, without any kind of ordering.
        """
        return list(self.graph['Nodes'].keys())
    
    def get_num_nodes(self) -> int:
        """
        Returns number of nodes in the net.
        """
        return len(self.get_nodes())
    
    def get_edges(self) -> list[tuple]:
        """
        Returns the edges present in the BN as a list of tuples.
        """
        return self.graph['Edges']
    
    def get_roots(self) -> list[str]:
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
    
    
    def get_children(self, node:str) -> list[str]:
        """
        Finds all children of the specified nodel. Returns a list containing their names.
        """
        return self.graph['Nodes'][node].get_children()
    
    def get_num_children(self, node:str) -> int:
        """
        Returns the number of children of the specified node
        """
        return self.graph['Nodes'][node].num_children
    
    def get_parents(self, node:str) -> list[str]:
        """
        Returns the parents of the specified node as a list, containing the names as strings
        """
        return self.graph['Nodes'][node].get_parents()
    
    def get_num_parents(self, node:str) -> int:
        """
        Returns the number of parents of the specified node
        """
        return self.graph['Nodes'][node].num_parents
    
    def top_sort(self) -> list[str]:
        """
        Topological sorting of nodes in the BN using Kahns algorithm. It also detects loops in the BN
        """
        cycle = False
        roots = self.get_roots()
        sorting = []
        
        # Save number of parents as a dictionary, where the key is the node and the value is its number of parents
        parent_num = {node:self.get_num_parents(node)
                      for node in self.graph['Nodes'].keys()}
        
        while roots:
            node = roots.pop(0)
            sorting.append(node)
            
            for child in self.get_children(node):
                # If a child has no parents after deleting the current node, add it to roots
                if parent_num[child] - 1 == 0:
                    roots.append(child) 
                # Otherwise delete one of its number of parents
                else:
                    parent_num[child] -= 1
                    
        if len(sorting) < self.get_num_nodes():
            warnings.warn('Cycle detected!')
            cycle = True
            
        return sorting
    
    def show_graphviz(self, filename:str):
        """
        Prints current BayesNet to a .png image file
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("'graphviz' library and software required")
        
        dot = graphviz.Digraph(name=self.BN_name, format='png')
        
        for node_name in self.get_nodes():
            dot.node(node_name)
            
        for edge in self.get_edges():
            parent = edge[0]
            child = edge[1]
            dot.edge(parent, child)
            
        dot.render(filename, view=True)
        return 
    
    def to_dot(self, filename:str):
        """
        Saves current BayesNet to a .dot file
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("'graphviz' library and software required")
        
        dot = graphviz.Digraph(name=self.BN_name, format='dot')

        for node in self.get_nodes():
            dot.node(node)
            
        for edge in self.get_edges():
            parent, child = edge[0], edge[1]
            dot.edge(parent, child)
            
        dot.save(filename)
        return dot
    
    def add_var_vals_from_df(self, df:pd.DataFrame):
        """
        Adds all possible values a variable can take from a given DataFrame
        """
        vars = df.columns.to_list()
        for var in vars:
            vals_list = list(df[var].unique())
            self.add_var_values(var, vals_list)
            
    def add_CPT(self, var_name:str):
        """
        Creates empty CPT for specified values. It assumes that the variable has been assigned
        a set of (finite) possible values it can take
        """
        var_values = self.graph['Nodes'][var_name].get_var_values()
        if not var_values:
            raise IndexError(f'Variable {var_name} has no set values!')
        
        parent_names = self.get_parents(var_name)
        parent_values = self.get_vars_values(parent_names)
        
        cpt = CPT(var_name, var_values, parent_values)
        self.CPTs[var_name] = cpt
        return None
    
    def add_all_CPTs(self):
        """
        Adds empty CPTs for all nodes present in the BN
        """
        for var in self.graph['Nodes'].keys():
            self.add_CPT(var)
        
    def get_CPT(self, var_name:str) -> CPT:
        return self.CPTs[var_name]
    
    def learn_CPTs_from_data(self, df: pd.DataFrame):
        """
        Learn all CPTs in the net from the data given in a DataFrame
        """
        for var_name, cpt in self.CPTs.items():
            cpt.update_from_data(df)
                