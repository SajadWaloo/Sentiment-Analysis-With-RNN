import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  # Add this line
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Set the file paths
train_dir = '/Users/mac/Desktop/Projects/sentiment analysis model/aclImdb/train'
test_dir = '/Users/mac/Desktop/Projects/sentiment analysis model/aclImdb/test'

# Rest of the code remains the same...


# Load the training dataset
train_data = pd.DataFrame(columns=['Review', 'Sentiment'])

for label in ['pos', 'neg']:
    folder = train_dir + '/' + label
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as file:
            review = file.read()
            train_data = train_data.append({'Review': review, 'Sentiment': label}, ignore_index=True)

# Load the testing dataset
test_data = pd.DataFrame(columns=['Review', 'Sentiment'])

for label in ['pos', 'neg']:
    folder = test_dir + '/' + label
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r') as file:
            review = file.read()
            test_data = test_data.append({'Review': review, 'Sentiment': label}, ignore_index=True)

# Split the data into features and labels
X_train = train_data['Review']
y_train = train_data['Sentiment']
X_test = test_data['Review']
y_test = test_data['Sentiment']

# Tokenize the text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text sequences to sequences of integers
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure equal length
max_len = 100  # Maximum sequence length
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Build the RNN model
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model's performance
y_pred = model.predict_classes(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mtx = confusion_matrix(y_test, y_pred)

print('Accuracy:', accuracy)
print('Confusion Matrix:')
print(confusion_mtx)

# Generate an accuracy plot
epochs = range(1, 11)
plt.plot(epochs, model.history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, model.history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
