This is a machine learning project that uses neural networks to determine the sentiment of a movie, which is either good (1) or bad (0).

Datasets used: 
Kaggle - Rotten 'Tomatoes movies and critic reviews dataset' (used data in 'review_type' and 'review_content')\
https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/\
Keras - IMDB (used data regarding the rating and review)\

Packages used:\
tensorflow\
numpy\
pandas\
sklearn\

Steps to replicate: 
1. Download the Kaggle 'Rotten Tomatoes movies and critic reviews dataset' and extract the rotten_tomatoes_critic_reviews.csv file.
2. Run the manipulateData.py script. The script modifys the rotten_tomatoes_critic_reviews.csv file and puts the processed data in revised_tomatoes.csv. It also creates a new accuracy.csv file to keep track of runs. 
3. Run the runModel.py script. This file creates the neural network, trains the neural network, and saves the model as 'model.h5' if the model's accuracy is better than the previous runs.

Testing your own data:
1. Import your own textfiles from different movie reviews.
2. Edit the predict.py script and add function calls for predict.
```
predict("your_textfile_name.txt")
```
3. Run the predict.py script.