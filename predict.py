from dis import dis
import pandas as pd
import numpy as np
from scipy import spatial
import pickle as pk
from kmeans_model import train_kmeans
from vector import get_avg_vector



input_data = input("Want to meet:")


def get_similar_ids(userData):
    #get vector of input data
    df = train_kmeans()
    k = get_avg_vector(userData)

    #If we fit PCA only on new data then results might not be accurate as PCA is context and data specific. So, we will use the PCA trained/fitted on our dataset earlier. 
    #transform our vector to 2 dimensions by applying PCA on it.
    pca = pk.load(open("pca.pkl", "rb"))
    y_principal = pca.transform(k.astype(float).reshape(1, -1))#reshape(1,-1) is used when there is only one sample for PCA 
    y_principal = pd.DataFrame(y_principal)

    #predict cluster of our data
    kmeans = pk.load(open("kmeans_model.pkl", "rb"))
    predicted_cluster = kmeans.predict(np.array(y_principal))
    

    #make a new dataframe with rows only of our data's cluster and take only the word vector and id from each row
    new_df = df.loc[df['cluster'] == predicted_cluster[0]][["vec", "id"]]

    #initalize an empty list. This list will be used to store the id and distance of each relevant neighbour
    l = []

    #for each relevant row
    for row in new_df.itertuples(index = False):

        #calcluate euclidean distance between our data and the row's data
        distance = spatial.distance.euclidean(k, row[0])
        #append a tuple of distance and id to the list
        l.append((distance,row[1]))

    #sort our list in ascending order to get the most similar results first. By default, this will sort by the first entry in the tuple i.e. distance
    l = sorted(l)

    #get only ids from the list
    ids = [i[1] for i in l]

    for id in ids[:5]:
        print(df.loc[df['id'] == id][["id","combined"]].values,id)
    

    #return the first 11 entries in the list
    return ids 


get_similar_ids(input_data)


