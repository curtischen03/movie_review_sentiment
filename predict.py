import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
print(tf.config.list_physical_devices('GPU'))

data = keras.datasets.imdb
#integer encoded words
word_index = data.get_word_index() 
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

processed_data = pd.read_csv("revised_tomatoes.csv",sep=',')
reviews = processed_data.review
start = 88588
for review in reviews:
    reviewWords = str(review).split(" ")
    for i in reviewWords:
        if i not in word_index:
            word_index[i] = start
            start += 1

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

model = keras.models.load_model("model.h5")
def predict(fileName):
    rating = 0
    with open(fileName,encoding="utf-8") as f:
        for line in f.readlines():
            nline=line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
            words = []
            for i in nline:
                if nline != "":
                    words += [i]
            encode = review_encode(words)
            encode = keras.preprocessing.sequence.pad_sequences([encode],value=word_index["<PAD>"],padding="post",maxlen=250)
            predict = model.predict(encode)
            rating = predict[0]
            #print(line)
            #print(encode)
    print("Filename " + fileName + ": ", rating)
    if rating < .1:
        print("Very likely a bad review")
    elif rating < .4:
        print("Likely a bad review")
    elif rating < .5:
        print("Maybe a bad review?")
    elif rating < .6:
        print("Maybe a good review?")
    elif rating < .9:
        print("Likely a good review")
    else:
        print("Very likely a good review")

#examples:
'''predict("topgun.txt")
predict("morbius.txt")
predict("interstellar.txt")
predict("emojimovie.txt")
predict("spiderman2.txt")'''
      