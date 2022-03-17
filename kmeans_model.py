from sklearn.cluster import KMeans
import pickle as pk
from pca import reduce_dim
from vector import get_all_vectors


def train_kmeans():
    X = reduce_dim()
    kmeans = KMeans(n_clusters = 8,n_init = 200).fit(X)

    df = get_all_vectors()
    df['cluster'] = kmeans.fit_predict(X)

    # df.to_csv("clusters.csv")
    pk.dump(kmeans, open("kmeans_model.pkl", 'wb')) #Saving the model

    return df