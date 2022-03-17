### Import libraries
from gensim.models import FastText

from read_csv import load_df

df = load_df()

### FastText 
"""
fastText is a library for efficient learning of word representations and sentence classification.Also fastText provides the pre-trained model 
which are used to train the custom data in this project.Here we used 100 dimensions contain compressed model where as original has 300 dimensionas 
which are quite high.
Apart from that here, we used gensim library instead of fastText original library because compare to original gensim will load easily and 
also help to compressed the pre-trained fasttext model.
"""
print("Hi")
#load the compressed,pre-trained 100 dimensioned model
model = FastText.load_fasttext_format('cc.en.100.bin')

# convert combined column to list so we can build vocabulary using model and train using pre-trained fasttext model 
data = df['combined'].tolist()

#Build vocabulary from a sequence of documents (can be a once-only generator stream), which help whenn we try train custom data using pre-trained model
#build vocabulary of model from our data
model.build_vocab(data, update = True)
print("Hello")
### Train custom data on pre-trained model
#training model on our custom data
#corpus_iterable is the place input to model(data in our case), total_examples tells of the count of the number of entries/rows in the data
model.train(corpus_iterable = data,
            total_examples = model.corpus_count,
            epochs = 5,
            compute_loss = True)

#Save fasttext model after training on custom data
#save model in both formate bin and vec(This will take some time)

model.save("custom_data_model_100_e5.bin")
model.save("custom_data_model_100_e5.vec")