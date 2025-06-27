import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BayesNet.BayesNet import *
from BayesNet.net_learning import *

import pandas as pd

df = pd.read_csv('Gold_Nets/databases/goldNet3_data.csv')
bn = chow_liu(df, 'GoldNet3')
bn.learn_CPTs_from_data(df)
#bn.show_graphviz('Gold_Nets/images/goldNet5_root2')
gum_bn = export_to_pyagrum(bn)
gum_bn.saveBIF("Gold_Nets/gum/goldNet3")

