import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BayesNet.utils import discretize_kbins
from BayesNet.BayesNet import *
from BayesNet.structure_learning import *
from sklearn.datasets import load_iris

import pandas as pd

iris = load_iris(as_frame=True)
df:pd.DataFrame = iris.frame.copy()
df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]

df_discrete = discretize_kbins(df, 'uniform')


bn = chow_liu(df_discrete, 'IrisBayesNet')
bn.to_graphviz("Nets/IrisBayesNet")