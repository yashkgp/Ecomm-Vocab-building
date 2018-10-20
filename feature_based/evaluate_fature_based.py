import os
import pickle
import json
import sklearn_crfsuite  
from  sklearn_crfsuite import metrics
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from train_feature_based import load_dataset, fashionTerms, load_terms
from train_feature_based import annotate_with_dep_parsing, word2features, sent2features, sent2labels, sent2tokens

#######################################
#
# Testing and evaluation
#
#######################################

def extractedTerms(y_test,actualTokens,beg,ins):
    numTestData= len(actualTokens)
    actualTerms={}
    for i in range(numTestData):
        sentLength= len(actualTokens[i])
        for sl in range(sentLength -1) :
            
            if y_test[i][sl]==beg:
                word = actualTokens[i][sl]+" "+actualTokens[i][sl+1]
                actualTerms[word.strip()]=word.strip()
            # elif y_test[i][sl].startswith(beg) and y_test[i][sl+1].startswith(ins) and y_test[i][sl][2:]==y_test[i][sl+1][2:]:
            #     nxt=sl+1
            #     word = actualTokens[i][sl]
            #     while(y_test[i][nxt].startswith(ins) and nxt <(sentLength-1)):
            #         word= word +' '+ actualTokens[i][nxt]
            #         nxt+=1    
            #     actualTerms[word.strip()]=word.strip()
            #     sl=nxt

        if y_test[i][-1]==beg:
            word = actualTokens[i][-1]
            actualTerms[word.strip()]=word.strip()
    return actualTerms

def evaluate(test_sents, modelfile, mode):
    # load the model
    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]
    loaded_model = pickle.load(open(modelfile, mode))
    labels = list(loaded_model.classes_)
    labels.remove('O')
    labels
    y_pred = loaded_model.predict(X_test)
    sorted_labels = sorted(labels,
        key=lambda name: (name[1:], name[0]))

    print(metrics.flat_classification_report( y_test, y_pred, labels=sorted_labels, digits=3 ))
    acc = metrics.flat_accuracy_score(y_test, y_pred)
    precision =metrics.flat_precision_score(y_test, y_pred, average='macro', labels=labels)
    recall=metrics.flat_recall_score(y_test, y_pred, average='macro', labels=labels)
    f1= metrics.flat_f1_score(y_test, y_pred, average='macro', labels=labels)

    print 'macro averaged result'
    print 'Accuracy \t= '+ str(acc)
    print 'Precision \t= '+str(precision)
    print 'Recall \t\t= '+str(recall)
    print 'F1 Score \t= '+str(f1)

    return y_test, y_pred



if __name__ == '__main__':
    train_FBR, train_FSN, train_ART, train_ATTR = load_terms("train_terms.json")
    train_keys = dict(train_FBR.items() + train_FSN.items() + train_ART.items() + train_ATTR.items())
    test_sents  = load_dataset("test_2704_zalando.txt")
    #test_sents = annotate_with_dep_parsing(test_sents,train_keys)
    y_test, y_pred = evaluate(test_sents, "CRF_model", "rb")
    

    class_names = ['B-ART','I-ART','B-ATTR','I-ATTR',
                   'B-FSN','I-FSN','B-FBR','I-FBR']
    debug ={}
    debug['y_test'] = y_test
    debug['y_pred'] = y_pred
    actualTokens=[sent2tokens(s) for s in test_sents]
    train_sents  = load_dataset("train_2704_zalando.txt")
    trainTokens=[sent2tokens(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    debug['actualTokens'] = actualTokens
    del train_sents
    predicted_wrong_all ={}
    Predicted_Terms ={}
    Test_terms={}
    extract_train ={}
    extract_test ={}
    extract_pred ={}
    new_terms = {}
    new_terms_new={}
    multikey={}
    for i in range(len(class_names)/2):
        print class_names[i*2],class_names[2*i+1]
        B = class_names[i*2]
        I = class_names[2*i+1]
        actualTerms = extractedTerms(y_test,actualTokens,B,I) #Test
        predictedTerms = extractedTerms(y_pred,actualTokens,B,I)
        trainTerms = extractedTerms(y_train,trainTokens,B,I)
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

        # print 'percentage of correctly predicted terms : ' +str(float(len(set(predictedTerms)&set(actualTerms)))/float(len(predictedTerms))*100)
        # print '******************************'

        #print diff & set(predictedTerms) # terms not in training but correctly identified
        #terms_not_in_gloss=set(predictedTerms) - diff -set(trainTerms) #new term
        # print '******************************'
        # termtype=B[2:]
        # terms_not_in_gloss = {e:e for e in terms_not_in_gloss}
        # multikey[termtype]=terms_not_in_gloss

    # with open('newTerm100k.json','w') as outfile :
    #     json.dump(multikey, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    with open('extract_train_zalando.json','w') as outfile :
        json.dump(extract_train, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    with open('extract_test_zalando.json','w') as outfile :
        json.dump(extract_test, outfile, sort_keys=True, indent=4, separators=(',', ': '))  
    with open('extract_pred_zalando.json','w') as outfile :
        json.dump(extract_pred, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    with open('NewTerms_zalando.json','w') as outfile :
        json.dump(new_terms, outfile, sort_keys=True, indent=4, separators=(',', ': '))
    with open('NewTermsNew_zalando.json','w') as outfile :
        json.dump(new_terms_new, outfile, sort_keys=True, indent=4, separators=(',', ': ')) 


