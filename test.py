from BayesNet import BayesNet
import numpy as np
import pandas as pd

test_net = BayesNet('test')
test_net.add_node('A')
test_net.add_node('B')
test_net.add_node('D')

test_net.add_edge('A', 'B')
test_net.add_edge('B', 'C')

print(test_net.get_nodes(), test_net.graph['Edges'], test_net.get_orphans())
print(test_net.top_sort())
