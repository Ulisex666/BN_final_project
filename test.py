from BayesNet import *
from utils import get_mutual_info
import numpy as np
import pandas as pd

# test_net = BayesNet('test')
# test_net.add_node('A')
# test_net.add_node('B')
# test_net.add_node('D')

# test_net.add_edge(('A', 'B'))
# test_net.add_edge(('B', 'C'))

# print(test_net.get_nodes(), test_net.graph['Edges'], test_net.get_roots())
# test_net.reverse_edge(('A', 'B'))
# print(test_net.get_edges())

data = {
    'x1': [1,1,0,1,0,0,1,0,1,0],
    'x2': [0,1,0,1,0,1,1,0,1,0],
    'x3': [0,1,1,1,0,1,1,0,1,0]
    }

df = pd.DataFrame(data)
bn = BayesNet('ChowLiu')

bn = chow_liu(df, bn)
print(f'Edges: {bn.get_edges()}')