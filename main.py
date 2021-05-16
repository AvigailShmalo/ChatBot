# to get a random response
import random

#to packege the code
import pickle

import numpy as np

#tp access the Json fie
import json
import nltk
import ssl

# i have absolutly no ideal what this is it just helped me fix an error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')

#to find simular words
from nltk.stem import WordNetLemmatizer

#for AI
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lammatizer= WordNetLemmatizer()

#read the contets of the intents.json file
intents= json.loads(open("intents.json").read())

words= []
classes= []
docs= []
# list of letters in the question to ignore
ignored_letters= ["!", "?", ".", ","]

#iterate over intents
for intent in intents["intents"]:
    for pattern in intent['patterns']:
        #splits question into single words
        word_list= nltk.word_tokenize(pattern)
        #add to list of words
        words.extend(word_list)
        #add to docs
        docs.append((word_list, intent['tag']))
        #check if the intent is already in the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lamitize words 
words= [lammatizer.lemmatize(word) for word in words if word not in ignored_letters]
#get rid of duplicates
words = sorted(set(words))

classes = sorted(set(classes))

#save to file as writing binaries
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open ('classes.pk1', 'wb'))

#turning the words into numerical values
training = []
output_empty = [0] *len(classes)

for doc in docs:
    bag= []
    word_pattern = doc[0]
    word_pattern = [lammatizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        if word in word_pattern:
            bag.append(1) 
        else:
            bag.append(0)
   
    #copy list
    output_row= list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#shuffle data
random.shuffle(training)
#turn into a numpy array
training= np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

#create sequential model
model = Sequential()
#add densitiy layer
model.add(Dense(128, input_shape=(len(train_x[0]), ), activation= 'relu'))
#pernevts overfitting
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#scales results in the output layer so they add up to 1 persentage of how likely it is to have a certain output
model.add(Dense(len(train_y[0]), activation= 'softmax'))
#define a stochastic gradient decent optimiser 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#training and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done!')
