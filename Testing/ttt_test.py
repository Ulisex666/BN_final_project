import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BayesNet.net_learning import chow_liu
from BayesNet.BayesNet import *
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data"
columns = [
    'top-left', 'top-middle', 'top-right',
    'middle-left', 'middle-middle', 'middle-right',
    'bottom-left', 'bottom-middle', 'bottom-right',
    'class'
]

df = pd.read_csv(url, names=columns)
bn = chow_liu(df, bn_name='TicTacToe')
for var in bn.top_sort():
    print('-'*20)
    print(bn.get_CPT(var))
    print('-'*20)
bn.show_graphviz('Nets/TicTacToe')