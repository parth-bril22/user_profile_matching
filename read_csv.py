import pandas as pd
def load_df():
    df = pd.read_csv("final_data.csv")
    #drop the Unnamed:0 column
    df = df.drop("Unnamed: 0", axis = 1)
    return df