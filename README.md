# Sentiment-Analysis-With-RNN
Our sentiment analysis RNN model predicts the sentiment (positive/negative) of text data. Trained on customer reviews, it captures sequential dependencies for accurate classification. Evaluate performance using accuracy metrics and visualize training plots. Gain insights into customer opinions and brand perception.
This project implements a sentiment analysis model using a recurrent neural network (RNN) to analyze the sentiment of text data. Sentiment analysis, also known as opinion mining, is the task of determining the sentiment or emotion expressed in a given piece of text.

The model is trained on a dataset of customer reviews or social media posts and can predict whether a given text expresses positive or negative sentiment. By analyzing the sentiment of text data, businesses and organizations can gain valuable insights into customer opinions, public sentiment, and brand perception.

## Features

- **Deep Learning Model**: The sentiment analysis model is built using a recurrent neural network (RNN) architecture. The RNN model is capable of capturing sequential dependencies in the text data, enabling it to understand the sentiment expressed in a given sequence of words.

- **Text Preprocessing**: The project includes text preprocessing steps such as tokenization, converting text sequences to integers, and padding sequences to ensure equal length. These preprocessing steps are crucial for preparing the text data to be fed into the RNN model.

- **Model Training and Evaluation**: The model is trained using the IMDb movie review dataset, a popular dataset for sentiment analysis. The dataset consists of labeled movie reviews, allowing the model to learn the sentiment patterns from both positive and negative examples. The model's performance is evaluated on a separate test dataset to assess its accuracy and effectiveness.

- **Visualization**: The project provides visualizations of the model's accuracy during training. The accuracy plot displays the training and validation accuracy across epochs, allowing for easy interpretation of the model's learning progress.

## Dataset

The sentiment analysis model is trained and evaluated using the IMDb movie review dataset. The dataset contains a large collection of movie reviews, labeled as either positive or negative sentiment. Each review is accompanied by the corresponding sentiment label, allowing the model to learn from labeled examples during training.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.0 or higher)
- NumPy
- pandas
- matplotlib

## Usage

1. Download the IMDb movie review dataset from the provided link.
2. Extract the dataset and locate the train and test folders.
3. Update the paths to the train and test folders in the code, as specified in the comments.
4. Run the `sentiment_analysis_model.py` script.

```shell
python sentiment_analysis_model.py
```

## Results

Upon executing the script, the sentiment analysis model will be trained and evaluated on the IMDb movie review dataset. The script will output the following results:

- Accuracy: The accuracy of the model on the test dataset.
- Confusion Matrix: A matrix showing the true positive, true negative, false positive, and false negative predictions.

Additionally, an accuracy plot will be displayed, showing the training and validation accuracy of the model across epochs.

## Acknowledgements

- The IMDb movie review dataset is provided by Stanford University and can be accessed at [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/).
- The model implementation is based on the TensorFlow and Keras frameworks.
