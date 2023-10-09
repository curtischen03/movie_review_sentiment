import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import metrics


print(tf.config.list_physical_devices('GPU'))

data = keras.datasets.imdb
#only take 88588 most frequent words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=100000)

#integer encoded words
word_index = data.get_word_index() 
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#decode from integers to words
def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

#encode from words to integers
def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

#rotten tomatoes data
processed_data = pd.read_csv("revised_tomatoes.csv",sep=',')
reviews = processed_data.review

#add rotten tomatoes words to dictionary
start = 88588
for review in reviews:
    reviewWords = str(review).split(" ")
    for i in reviewWords:
        if i not in word_index:
            word_index[i] = start
            start += 1

reviews_encoded = []
for i in range(len(reviews)):
    reviews_encoded += [review_encode(str(reviews[i]).split(" "))]
ratings = np.array(processed_data.rating)

#adds padding to rotten tomatoes reviews
def pad(arr):
    result = []
    for i in arr:
        if len(i) < 500:
            padding = [1] * (500 - len(i))
            result += [i + padding]
    return result

reviews_encoded = pad(reviews_encoded)

#add padding to imdb data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=500)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=500)

#split rotten tomatoes data
train_data2,test_data2,train_labels2,test_labels2 = sklearn.model_selection.train_test_split(reviews_encoded,ratings,test_size = 0.2)

#convert imdb data to list of list so we can add with rotten tomatoes data
train_data = list(train_data)
train_data = [list(train_data[i]) for i in range(len(train_data))]
test_data = list(test_data)
test_data = [list(test_data[i]) for i in range(len(test_data))]

train_data = train_data + train_data2
test_data = test_data + test_data2
train_labels = np.append(train_labels,train_labels2)
test_labels = np.append(test_labels,test_labels2)

#convert added data back to np array of np arrays
train_data = np.array([np.array(train_data[i]) for i in range(len(train_data))])
test_data = np.array([np.array(test_data[i]) for i in range(len(test_data))])


model = keras.Sequential()
model.add(keras.layers.Embedding(300000,16))

#makes it a lower dimension
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(30,activation="relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss = "binary_crossentropy",metrics=["accuracy"])
x_val=train_data[:20000]
x_train=train_data[20000:]
y_val=train_labels[:20000]
y_train=train_labels[20000:]


#batch size: how many movie reviews to load in at once
fitModel=model.fit(x_train,y_train,epochs=40,batch_size=512,validation_data=(x_val,y_val),verbose=1)
results =model.evaluate(test_data,test_labels)
print("Current model accuracy: ",results[1])

prev_accuracy = pd.read_csv('accuracy.csv').Accuracy
best_accuracy = max(prev_accuracy)
print("Best Previous Accuracy: ", best_accuracy)
prev_accuracy = list(prev_accuracy)
prev_accuracy += [results[1]]
header = ['Accuracy']
accuracy_data = [[prev_accuracy[i]] for i in range(len(prev_accuracy))]
accuracy_data = pd.DataFrame(accuracy_data,columns=header)
accuracy_data.to_csv('accuracy.csv',index=False)

if results[1] > best_accuracy:
    model.save("model.h5")


