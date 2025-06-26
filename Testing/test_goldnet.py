import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BayesNet.BayesNet import *
from BayesNet.net_learning import *

import pandas as pd

df = pd.read_csv('Gold_Nets/goldNet1_data.csv')
bn = chow_liu(df, 'GoldNet1_root1', 'Node1')
bn.learn_CPTs_from_data(df)
bn.print_all_CPT('Gold_Nets/goldNet1_root1_CPTs')
bn.show_graphviz('Gold_Nets/learned_goldNet1_root1')
