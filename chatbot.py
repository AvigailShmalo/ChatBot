import random
import json
import pickle
import numpy as np

import nltk
from nltk import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer= WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
#open words file
words= pickle.load(open('words.pk1', 'rb'))

#open classes file
classes= pickle.load(open('classes.pk1', 'rb'))
model= load_model('chatbot_model.h5')

#functions to work with the numerical data in terms of words
#clean up sentances
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#convert a sentance into a bag of words indicating if a word is there or not using 0s and 1s
def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag= [0]* len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]= 1 #otherwise 0
    return np.array(bag)

#predict class based on sentence
def predict_class(sentence):
    bow= bag_of_words(sentence)
    #predict results based on the bag of words
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = .25 #error threshold is 25%
    results= [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]#allows for uncertanty but if too high not take into result

    #sort results by probablility in revers order
    results.sort(key= lambda x: x[1], reverse= True)
    return_list= []
    #appened intents and probabilities to retrun_list
    for r in results:
       return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    result = None ####### Initialise before hand
    tag = ints[0]["intent"]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result #### Changed here
    return result

print('This is a ChatBot that knows a bunch of things about Avigail. Ask away!!!')

while True:
    #user input
    message = input("")
    ints = predict_class(message)
    res= get_response(ints, intents)
    print (res)