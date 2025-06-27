import pandas as pd
import numpy as np
import pyagrum as gum

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

def export_to_pyagrum(bn) -> gum.BayesNet:
    """
    Convierte un objeto BayesNet propio a un objeto pyAgrum.BayesNet.
    """
    gum_bn = gum.BayesNet(bn.BN_name)

    # 1. Agregar variables
    var_mapping = {}  # Mapea nombre de variable a ID de pyAgrum
    for var_name in bn.get_nodes():
        var_values = bn.graph['Nodes'][var_name].get_var_values()
        pyagrum_var = gum.LabelizedVariable(var_name, var_name, len(var_values))
        for val in var_values:
            pyagrum_var.changeLabel(var_values.index(val), str(val))
        var_id = gum_bn.add(pyagrum_var)
        var_mapping[var_name] = var_id

    # 2. Agregar arcos
    for parent, child in bn.get_edges():
        gum_bn.addArc(var_mapping[parent], var_mapping[child])

    # 3. Agregar CPTs
    for var_name in bn.get_nodes():
        cpt = bn.get_CPT(var_name).to_dataframe()
        var_vals = bn.graph['Nodes'][var_name].get_var_values()
        parents = bn.get_parents(var_name)
        
        # Si no tiene padres (nodo raíz)
        if not parents:
            prob_vector = [float(cpt.iloc[0][f'P({var_name}={val})']) for val in var_vals]
            gum_bn.cpt(var_name).fillWith(prob_vector)
        else:
            # Para cada fila de la tabla, insertar los valores
            for _, row in cpt.iterrows():
                parent_inst = {p: str(row[p]) for p in parents}
                for val in var_vals:
                    full_inst = {**parent_inst, var_name: str(val)}
                    prob = float(row[f'P({var_name}={val})'])
                    gum_bn.cpt(var_name)[full_inst] = prob
    return gum_bn

def classify_and_accuracy(gum_bn: gum.BayesNet, df_test: pd.DataFrame, target_var: str) -> float:
    correct = 0
    total = len(df_test)
    
    ie = gum.LazyPropagation(gum_bn)
    
    for _, row in df_test.iterrows():
        # Extraer evidencia (omitimos la clase objetivo)
        evidence = {
            var: str(row[var])
            for var in df_test.columns
            if var != target_var and pd.notna(row[var])
        }
        
        # Aplicar inferencia
        ie.setEvidence(evidence)
        ie.makeInference()
        posterior = ie.posterior(target_var)
        
        # Predecir la clase con mayor probabilidad
        # Obtener el índice del valor con mayor probabilidad para la variable target
        predicted_index = posterior.argmax()[0][0][target_var] # type: ignore
        predicted_class = gum_bn.variable(target_var).label(predicted_index)
        actual_class = str(row[target_var])
        
        if predicted_class == actual_class:
            correct += 1

    accuracy = correct / total
    return accuracy
