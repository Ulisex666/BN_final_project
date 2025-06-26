import os
import sys
# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BayesNet.BayesNet import *
from BayesNet.net_learning import *
import pandas as pd

df = pd.read_csv('Databases/7Raisins.csv')
bn = chow_liu(df, 'Raisins')
bn.learn_CPTs_from_data(df)
bn.print_all_CPT('Nets/CPTs/RaisinsCPTs')
bn.to_dot('Nets/dots/raisins_bn')
bn.show_graphviz('Nets/images/raisins_bn')