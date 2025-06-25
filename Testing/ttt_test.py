import os
import sys

# Agrega la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BayesNet.structure_learning import chow_liu
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
df_vars = df.drop(columns=['class'])  # solo variables del tablero
bn = chow_liu(df_vars, bn_name='TicTacToe')
bn.to_graphviz('Nets/TicTacToe')