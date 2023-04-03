import numpy as  np 
import tensorflow as tf 
import tflearn
import random



import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()



# other
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

print('processing the Intents.......')
with open('intents.json') as json_data:
    intents = json.load(json_data)







words = []
classes = []
documents = []
ignore_words = ['?']


print('Looping through intents to convert them to words, classes,documents, and ignore_words....')

for intent in intents['intents']:
    for pattern in intent['patterns']:


        w = nltk.word_tokenize(pattern)

        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


print('Stemming,Lowering and Removing duplicates......')

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))


classes = sorted(list(set(classes)))




print('creating the data for our model')

training = []
output = []

print('creating a list ,,, empty for output')


output_empty = [0]*len(classes)

print('creating training set, bag of words for our model')

for doc in documents:

    bag = []
    pattern_words = doc[0]

    pattern_words = [stemmer.stem(word.lower())for word in words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)



        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag,output_row])
        
training[1][1]


print('shuffling randomly and converting into numpy array for fasrer processing')

random.shuffle(training)

training = np.array(training)

print('creating train and test list')

train_x = list(training[:,0])
train_y = list(training[:,1])
print('building neural network')




net = tflearn.input_data(shape=[None,len(train_x[0])])
net =  tflearn.fully_connected(net, 8)
net =  tflearn.fully_connected(net, 8)
net =  tflearn.fully_connected(net, len(train_y[0]), activation= 'softmax')
net =  tflearn.regression(net)

print('training')


model = tflearn.DNN(net,tensorboard_dir='tflearn_logs')


print('training')

model.fit(train_x,train_y,n_epoch=1000,batch_size=8,show_metric=True)
print('saving the model')
model.save('model.tflearn')


print('pickle is also saved')
pickle.dump({'words':words, 'classes': classes,'train_x':train_x,'train_y':train_y},open('training_data','wb'));



print('loading pickle')

data = pickle.load(open('training_data','rb'))

words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']


with open('intents.json') as json_data:
    intents = json.load(json_data)



print('loading model')


model.load('./model.tflearn')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower())for word in sentence_words]
    return sentence_words


def bow(sentence,words,show_details = False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]* len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[1] = 1
                if show_details:
                    print('found in bag: '%w)

    return(np.array(bag))

ERROR_THRESHOLD = 0.25

def classify(sentence):
    results = model.predict([bow(sentence,words)])[0]

    results = [[i,r] for i,r in enumerate(results) if r> ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1],reverse = True)
    return_list=[]

    for r in results:
        return_list.append((classes[r[0]],r[1]))
    return return_list

def response(sentence, userID = '369',show_details= False):
    results = classify(sentence)

    if results:
        while results:
            for i in intents['intents']:
                if i['tag']==results[0][0]:
                    return print(random.choice(i['responses']))

            results.pop(0)


while True:
    input_data = input('you:  ')
    print(input_data)
    answer = response(input_data)
    print(answer)







def chat():
    #user input
    print("Start talking with bot!(type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        #All this is going to give us a matrix of numbers where the numbers are probabilities of each class
        results = model.predict([bag_of_words(inp, words)])
        #Argmax will grab the index of highest probability in the matrix
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))


chat()




print('completed')















