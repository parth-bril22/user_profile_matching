import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pickle as pk
from vector import get_all_vectors


def reduce_dim():
    df = get_all_vectors()
    X = df['vec'].values.tolist()

    # vec = vec.astype(float).reshape(1,-1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Converting the numpy array into a pandas DataFrame
    X_normalized = pd.DataFrame(X_scaled)
    X = X_normalized


    #reducing the 100 dimensions to 2 to increase time and space efficiency
    
    pca = PCA(n_components = 2)
    X_principal = pca.fit_transform(X)
    pk.dump(pca, open("pca.pkl", "wb"))

    #converting to dataframe
    X_principal = pd.DataFrame(X_principal)

    #normalize the data for better clustering
    X_principal = normalize(X_principal)
    X = X_principal
    return X