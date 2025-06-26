import os
import sys
# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BayesNet.BayesNet import *
from BayesNet.net_learning import *
import pandas as pd
 
df = pd.read_csv('Testing/3WinesD.csv')
bn = chow_liu(df, 'Wine')
bn.learn_CPTs_from_data(df)
for var in bn.top_sort():
    print(bn.get_CPT(var))
bn.show_graphviz('Nets/wine_bn')