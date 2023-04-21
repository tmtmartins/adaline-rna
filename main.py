import pandas as pd
from adaline import run
import sys

data_file = "./Dados.xls"
training_df = pd.read_excel(data_file, sheet_name="Treinamento")
test_df = pd.read_excel(data_file, sheet_name="Teste")
learning_rate = 0.0025
interactions_amount_limit = 10000
precision = 1e-6
if __name__ == "__main__":
    with open('output.txt', 'a') as f:
        sys.stdout = f
        print("\n\n5th round:")
        run(training_df, test_df, learning_rate, interactions_amount_limit, precision)
