import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

#process data
data = pd.read_csv("rotten_tomatoes_critic_reviews.csv",sep=",")
#print(data.columns)
reviews = data.review_content
ratings = data.review_type
finalReviews = []
finalRatings = []
special_chars = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '-', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', '?', ',', '.', '/']
for i in range(len(reviews)):
    if isinstance(reviews[i],str) and isinstance(ratings[i],str) and reviews[i] != "" and (ratings[i].lower() == "fresh" or ratings[i].lower() == "rotten"):
        review = ''
        for letter in reviews[i]:
            if letter not in special_chars:
                review += letter
        if ratings[i].lower() == 'fresh':
            finalRatings.append(np.int64(1))
            finalReviews.append(review)
        else:
            finalRatings.append(np.int64(0))
            finalReviews.append(review)
header = ['review','rating']
data = [[finalReviews[i],finalRatings[i]] for i in range(len(finalRatings))]
data = pd.DataFrame(data,columns=header)
data.to_csv('revised_tomatoes.csv',index=False)

#create accuracy file
header = ['Accuracy']
accuracy_data = [[0]]
accuracy_data = pd.DataFrame(accuracy_data,columns=header)
accuracy_data.to_csv('accuracy.csv',index=False)





                                                                                        