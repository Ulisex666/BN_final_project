from BayesNet.BayesNet import *
from BayesNet.net_learning import *
import numpy as np
import pandas as pd

# Testing basi functionality for the BayesNet class
# test_net = BayesNet('test')
# test_net.add_node('A')
# test_net.add_node('B')
# test_net.add_node('D')
# test_net.add_edge(('A', 'B'))
# test_net.add_edge(('B', 'C'))
# print(test_net.get_nodes(), test_net.graph['Edges'], test_net.get_roots())
# test_net.reverse_edge(('A', 'B'))
# print(test_net.get_edges())

# Testing for cycle detection. Resulting graph in test_cycle.png
# bn = BayesNet('Prueba')
# bn.add_edge(('A', 'B'))
# bn.add_edge(('C', 'B'))
# bn.add_edge(('D', 'B'))
# bn.add_edge(('B', 'E'))
# bn.add_edge(('E', 'A'))
# print(bn.top_sort())
# print(bn.to_graphviz('test_cycle'))

# Original data found in Cooper article for K2 algorithm. Resulting graph in test_cooper.png
data_cooper = {
    'x1': [1,1,0,1,0,0,1,0,1,0],
    'x2': [0,1,0,1,0,1,1,0,1,0],
    'x3': [0,1,1,1,0,1,1,0,1,0]
}

# This data gives a different graph with the chow_liu function
# data_cooper = {
#     'x1': [1,1,1,1,0,0,1,1,1,0],
#     'x2': [0,1,1,1,1,1,1,0,1,0],
#     'x3': [0,1,1,1,0,1,1,0,1,0]
# }


df = pd.DataFrame(data_cooper)
bn = chow_liu(df, 'Chow Liu')
print(bn.show_graphviz('test_cooper'))




