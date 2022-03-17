from nltk.corpus import stopwords
import numpy as np
from gensim.models import KeyedVectors
import nltk
nltk.download('stopwords')

from read_csv import load_df

model = KeyedVectors.load('custom_data_model_100_e5.vec')
df = load_df()

"""
Constructing our own get_sentence_vector function as the default does not work here(we have gensim model, not fasttext's)
According to get_Setnence_vector's implemenatation in fasttext library
(https://github.com/facebookresearch/fastText/blob/master/src/fasttext.cc#L428),
We take vec of each word and divide it by its norm,all such vecs are then averaged.
"""

#get the norm of a word
def get_norm(word):
    return np.linalg.norm(model.wv[word], 2) #2 for L2 norm



#implement our own variation of get_sentence_vector
def get_avg_vector(myString):
    
    #stop words are common words like is,am,are.... They will be removed to better gather important imformation/words and make vectors more distant.
    stop_words = set(stopwords.words('english'))
    
    #initialize array of zeros
    avg = np.zeros(shape = model.wv["hello"].shape)

    #if string is empty, return vec of zeros
    if(len(myString.strip()) == 0): return avg
    
    
    #to take only each word once, we will take a set of the list of words
    word_list = list(set(myString.split(" ")))

    #if word_list only has one word and it is longer than one letter
    if(len(word_list) == 1 and len(word_list[0]) > 0): return model.wv[word_list[0]]

    #count of total no of words
    count = 0

    for word in word_list:
        #according to fasttext's implementation, if the norm of a word is > 0, only then it will be taken into consideration 
        # other conditions like length of word greater than 1 and ignoring stop_words and converting all words to lower case are taken to ignore unimportant data  
        if(np.linalg.norm(model.wv[word], 2) > 0 and len(word) > 1 and word not in stop_words and word != ''):
            avg =  np.add(avg, model.wv[word]/np.linalg.norm(model.wv[word], 2))#convert words to lowercase
            count = count + 1

    count = max(count, 1)#to prevent divide by 0 error, make sure that count is 1
    avg = avg/count
    return avg



def get_all_vectors():
    df['vec'] = df['combined'].str.replace("\n","").apply(lambda x: (get_avg_vector(x)))
    return df