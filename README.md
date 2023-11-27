# Sentiment-Analysis-on-IMDB-large-movie-review-dataset

This project focuses on developing a machine learning model for sentiment analysis, specifically classifying movie reviews into positive or negative categories. The process encompasses data preprocessing, model development and training, and evaluation of the model's performance.

Dataset: https://ai.stanford.edu/~amaas/data/sentiment/

**Data Preprocessing** <br>
Data Loading: Utilized OS library functions to load and merge positive and negative reviews.
Dataset Preparation: Created labeled datasets for both training (train_NLP.py) and testing (test_NLP.py).
Text Cleaning: Removed special characters and HTML tags using a custom preprocessing function.
Stopword Removal: Employed NLTK's English Stopwords Collection for cleaner data.
Tokenization: Applied Keras tokenizer for word tokenization, saving the model for later use.
Data Split: Divided the data into an 80:20 split for training and validation.

![image](https://github.com/bhupeshdod/Sentiment-Analysis-on-IMDB-large-movie-review-dataset/assets/141383468/7cf0a0ff-8e4a-4c94-ba2e-3a08cfd68ae5)

**Model Development and Training** <br>
Architecture: The model includes an embedding layer, a Conv1D layer with Leaky ReLU activation, followed by a GlobalAverageMaxPooling1D layer.
Dense Layers: Implemented three dense layers with decreasing neurons (512, 256, 64), each with a 10% dropout rate and Leaky ReLU activation.
Binary Classification: The final layer uses a sigmoid function for binary classification of reviews.
Optimization: Employed binary_crossentropy for loss and the Adam optimizer with a learning rate of 0.001.
Training Process: Trained over 10 epochs with a batch size of 100, focusing on accuracy as the key metric.
Model Storage: The trained model is saved in .h5 format for future evaluation.

![image](https://github.com/bhupeshdod/Sentiment-Analysis-on-IMDB-large-movie-review-dataset/assets/141383468/99dd28ea-590c-4bce-9def-115ec23173a1)

**Results and Evaluation** <br>
Test Accuracy: Achieved an accuracy of 85.868% on the test dataset with a loss of 85.37%.
Training Insights: The model shows signs of overfitting, with a high accuracy (99.82%) and low loss (0.61%) on the training set compared to the validation set.

**Conclusion** <br>
The model demonstrates effective classification of movie reviews but needs improvements to address overfitting. This project lays a solid foundation for further exploration and optimization in the field of sentiment analysis.
