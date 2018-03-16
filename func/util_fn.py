import pandas as pd
from modules import *

def load_data(data_path):
    train_variants = pd.read_csv(data_path + 'training_variants')
    test_variants = pd.read_csv(data_path + 'test_variants')
    train_text = pd.read_csv(data_path + 'training_text', sep='\\|\\|', engine='python', skiprows=1, names=['ID', 'Text'])
    test_text = pd.read_csv(data_path + 'test_text', sep='\\|\\|', engine='python', skiprows=1, names=['ID', 'Text'])
    train_full = train_variants.merge(train_text, how='inner', left_on='ID', right_on='ID')
    test_full = test_variants.merge(test_text, how='inner', left_on='ID', right_on='ID')
    return (train_full, test_full)


