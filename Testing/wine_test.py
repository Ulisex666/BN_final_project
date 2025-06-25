import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BayesNet.utils import discretize_kbins
from BayesNet.BayesNet import *
from BayesNet.structure_learning import *
from sklearn.datasets import load_iris

import pandas as pd
 
df = pd.read_csv('Testing/3WinesD.csv')
bn = chow_liu(df, 'Wine')
bn.to_graphviz('Nets/wine_bn')