import pandas as pd

def load_data(path):
    data_df = pd.read_csv(path)
    
    return data_df