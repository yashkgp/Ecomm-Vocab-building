import pandas as pd
import numpy as np
import json
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import re
from nltk.stem.snowball import SnowballStemmer
import keras
from  sklearn_crfsuite import metrics
import tqdm

data = pd.read_csv("total.csv", encoding="utf-8")
#data2 = pd.read_csv("test_2704_fastext.csv", encoding="utf-8")
#data = pd.concat([data,data2])
# with open('total_text.txt') as of :
#     text = of.readlines()
data = data.fillna(method="ffill")
words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words)
tags = list(set(data["Tag"].values))
n_tags = len(tags)
print "ntags :" + str(n_tags)
print tags
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby(["Sentence #"]).apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["{}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
getter = SentenceGetter(data)
sent = getter.get_next()
print (sent)
sentences = getter.sentences
# print sentences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def clean_text(text):
    
    ## Remove puncuation
    # text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

tok_sent = []
for x in range(len(sentences)):
    sen =[]
    for y in range(len(sentences[x])):
        sen.append(sentences[x][y][0])
    st = " ".join(sen)
    tok_sent.append(st)
max_len = 40 # map back

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
from keras.preprocessing.sequence import pad_sequences
# X = [[word2idx[w[0]] for w in s] for s in sentences]
# X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tok_sent)
word_index = tokenizer.word_index
encoded_docs = tokenizer.texts_to_sequences(tok_sent)
print(encoded_docs[0])
# pad documents to a max length of 4 word
padded_docs = pad_sequences(encoded_docs, maxlen=max_len, padding='post',value= len(word_index))
print("paddeddoc: ",padded_docs[1])
index_word = {v: k for k, v in word_index.items()}

print('Found %s unique tokens.' % len(word_index))
X = padded_docs
y = [[tag2idx[w[2]] for w in s] for s in sentences]

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
print (X[1],y[1],sentences[1])
# X.reshape((1787,100,1))


from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]
print "To categorical",y[1:5]
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report


embeddings_index = {}
f = open('wiki.en.vec')
for line in f:
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
print "X :",X.shape,"Y : ", np.array(y).shape
embedding_matrix = np.zeros((len(word_index) + 1, 300))
embedding_matrix_unk = embedding_matrix[1]
embedding_matrix_unk[1]=1
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else :
        embedding_matrix[i] = embedding_matrix_unk
print "X_train  :",X_tr[1:5],y_tr[1:5]
input = Input(shape=((max_len,)))
model = Embedding(input_dim=len(word_index) + 1, output_dim=300,weights=[embedding_matrix],input_length=max_len, mask_zero=True,trainable=False)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=60, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(10, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
print (model.summary())
history = model.fit(X_tr, np.array(y_tr),validation_split=0.1, epochs= 500 , verbose=1)
hist = pd.DataFrame(history.history)
# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# plt.figure(figsize=(12,12))
# plt.plot(hist["acc"])
# plt.plot(hist["val_acc"])
# plt.show()
#model.save("mymodel.h5")
#model = keras.models.load_model("mymodel.h5")
test_pred = model.predict(X_te, verbose=1)
idx2tag = {i: w for w, i in tag2idx.items()}
# print  "test_pred: ", test_pred[1:5]
# print "labels : " ,y_te[1:5] 
# print "idx2tag",idx2tag

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
def extractedTerms(y_test,X_te,actualTokens,beg,ins):
    numTestData= len(y_test)
    actualTerms={}
    for i in range(numTestData):
        sentLength= len(y_test[i])
        for sl in range(sentLength -1) :
            
            if y_test[i][sl]==beg:
                if (X_te[i][sl]!= len(word_index)):
                    word = index_word.get(X_te[i][sl])+" "+index_word.get(X_te[i][sl+1])
                    actualTerms[word.strip()]=word.strip()


        if y_test[i][-1]==beg:
            word = index_word.get(X_te[i][-1])
            actualTerms[word.strip()]=word.strip()
    return actualTerms

pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)
print "After transformation :"
print  "test_pred: ", test_pred[1:5]
print "labels : " ,y_te[1:5] 
print "idx2tag",idx2tag 
labels ={}
labels["Pred"]=pred_labels

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))
actualTokens =[]
for se in X_te :
    for wor in se :
        if (wor!=len(word_index)):
            actualTokens.append(index_word.get(wor))
trainTokens =[]
for se in X_tr :
    for wor in se :
        if (wor!=len(word_index)):
            trainTokens.append(index_word.get(wor))

predicted_wrong_all ={}
Predicted_Terms ={}
Test_terms={}
extract_train ={}
extract_test ={}
extract_pred ={}
new_terms = {}
new_terms_new={}
multikey={}
y_test = test_labels
y_pred = pred_labels
y_train=pred2label(y_tr)
class_names = ['B-ART','I-ART','B-ATTR','I-ATTR',
                   'B-FSN','I-FSN','B-FBR','I-FBR']

for i in range(len(class_names)/2):
    print class_names[i*2],class_names[2*i+1]
    B = class_names[i*2]
    I = class_names[2*i+1]
    actualTerms = extractedTerms(y_test,X_te,actualTokens,B,I) #Test
    predictedTerms = extractedTerms(y_pred,X_te,actualTokens,B,I)
    trainTerms = extractedTerms(y_train,X_tr,trainTokens,B,I)
    Test_terms[B] = actualTerms
    Predicted_Terms[B] = predictedTerms
    actualTerms=set([x.lower() for x in actualTerms])
    predictedTerms=set([x.lower() for x in predictedTerms])
    trainTerms=set([x.lower() for x in trainTerms])
    predicted_wrong = list(set(actualTerms) - set(actualTerms)&set(predictedTerms))
    pred ={}
    tes ={}
    tra ={}
    for i in trainTerms :
        tra[i] = i 
    for i in actualTerms :
        tes[i] = i
    for i in predictedTerms :
        pred[i]=i
    extract_pred[B]=pred
    extract_test[B]=tes
    extract_train[B]=tra

    new_terms_in_test=list(set(predictedTerms) - set(trainTerms))
    new_terms_in_train =list(set(predictedTerms)- set(actualTerms)-set(trainTerms))
    new_terms[B]=new_terms_in_test
    new_terms_new[B]=new_terms_in_train
    print 'Terms it testData = '+str(len(actualTerms))
    print 'Terms it prediction = '+str(len(predictedTerms))
    print 'Terms it training = '+str(len(trainTerms))
    #print 'Common in test and train = '+str(len(set(actualTerms)&set(trainTerms)))
    print 'new in test =' +str(len(new_terms_in_test))
    print 'Term correctly predicted' + str(len(set(predictedTerms)&set(actualTerms)))
    print ' new Terms predicted correctly'+ str(len(set(predictedTerms)&set(new_terms_in_test)))

with open('extract_train_lstmfastext.json','w') as outfile :
    json.dump(extract_train, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('extract_test_lstmfastext.json','w') as outfile :
    json.dump(extract_test, outfile, sort_keys=True, indent=4, separators=(',', ': '))  
with open('extract_pred_lstmfastext.json','w') as outfile :
    json.dump(extract_pred, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('NewTerms_lstmfastext.json','w') as outfile :
    json.dump(new_terms, outfile, sort_keys=True, indent=4, separators=(',', ': '))
with open('NewTermsNew_lstmfastext.json','w') as outfile :
    json.dump(new_terms_new, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    acc = metrics.flat_accuracy_score(y_test, y_pred)
precision =metrics.flat_precision_score(y_test, y_pred, average='macro', labels=labels)
recall=metrics.flat_recall_score(y_test, y_pred, average='macro', labels=labels)
f1= metrics.flat_f1_score(y_test, y_pred, average='macro', labels=labels)

print 'macro averaged result:'
print 'Precision \t= '+str(precision)
print 'Recall \t\t= '+str(recall)
print 'F1 Score \t= '+str(f1)




with open('pred_labels.json', 'w') as outfile:
    json.dump(labels, outfile, sort_keys=True, indent=4, separators=(',', ': '))
