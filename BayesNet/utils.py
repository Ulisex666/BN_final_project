import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from pandas.plotting import table

def get_mutual_info(df: pd.DataFrame, var1:str, var2:str) -> float:
    """
    Function to calculate mutual information between two variables in a 
    database.
    Inputs: database as a pandas Dataframe, variable names as string.
    Returns: Mutual information as a float
    """
    
    # Calculate joint probabilities of both variables, given data observer
    # in dataframe
    freq_table = df.groupby([var1, var2]).size().reset_index(name='count')
    freq_table['Pxy'] = freq_table['count']/len(df)
    joint_probs = freq_table.drop('count', axis=1)
    
    # Get marginal probabilities
    Px = df[var1].value_counts(normalize=True).to_dict()
    Py = df[var2].value_counts(normalize=True).to_dict()
    
    mutual_info = .0
    
    for _, row in joint_probs.iterrows():
        x = row[var1]
        y = row[var2]
        
        pxy = row['Pxy']
        px = Px[x]
        py = Py[y]
        
        mutual_info += pxy * np.log2(pxy / (px*py))
        
    return mutual_info

    