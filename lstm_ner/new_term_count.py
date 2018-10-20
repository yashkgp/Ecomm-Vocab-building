from collections import defaultdict
import argparse
import pickle
import train_tag_word 
import test_new_word 
import json
import csv

import sys
reload(sys)
sys.setdefaultencoding('utf8')


def load_file(infile,mode):
    with open(infile,mode) as f :
        return pickle.load(f)

def fashionTerms(lists,key):
    for items in json_data[key]:
        getItem=(items.encode('ascii','ignore')).replace('_',' ')
        lists[getItem]=getItem
 
def setToDict(S):
    return {e:e for e in S}

def writeToCSV(mydict, outfile):
    with open(outfile, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])


def load_dataset(filename):
    with open(filename) as fp:
        tup =()
        sent, doc=[], []
        for line in fp:
            line = line.strip()
            if (len(line) == 0):
                # Empty line and also end of a sentence
                if len(sent) != 0 or line.startswith("-DOCSTART"):
                    doc.append(sent)
                sent =[]
            else :
                if len(line) < 2:
                    continue
                line_content = line.split("\t")
                tup = (line_content[0],line_content[1],line_content[2])
                sent.append(tup)
    print "Completed reading of {} lines of the dataset from file :{} ".format(len(doc),filename )
    return doc


def writeJSON(ART_dict, ATTR_dict, FBR_dict, FSN_dict,filename):
    multikey={}
    multikey['article_types'] =setToDict(ART_dict.viewkeys())
    multikey['attribute_types'] = setToDict(ATTR_dict.viewkeys())
    multikey['fashion_strings'] = setToDict(FSN_dict.viewkeys())
    multikey['fabric'] = setToDict(FBR_dict.viewkeys())

    with open(filename, 'w') as outfile:
        json.dump(multikey, outfile, sort_keys=True, indent=4, separators=(',', ': '))

#python new_term_count.py --label lab_.pkl --pred pred_.pkl --train_sent trainBIO --test_sent test_2704.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, dest="label", help="name of pickled original label file")
    parser.add_argument("--pred", required=True, dest="prediction", help="name of pickled prediction file")
    parser.add_argument("--train_sent", required=True, dest="train_sent", help="name of training file")
    parser.add_argument("--test_sent", required=True, dest="test_sent", help="name of testing file")

    options = parser.parse_args() 

    lab = load_file(options.label,"rb")
    pred_lab = load_file(options.prediction,"rb")
    train_sent = load_dataset(options.train_sent)
    test_sent = load_dataset(options.test_sent)
    print len(lab)
    print len(pred_lab)
    print len(train_sent)
    print len(test_sent)
    #testSent = load_file("test3010Ann.txt","r")
    #with open("train3010Ann.txt","r") as f :
    tART_dict, tATTR_dict, tFBR_dict, tFSN_dict  = train_tag_word.get_terms(train_sent) #train set
    ART_dict, ATTR_dict, FBR_dict, FSN_dict = train_tag_word.get_terms(test_sent)#test set       
    pART_dict, pATTR_dict, pFBR_dict, pFSN_dict = test_new_word.get_word(pred_lab,test_sent)    #prediction set

    print "count of dict terms"
    print len(tART_dict)+len(tATTR_dict)+len(tFBR_dict)+len(tFSN_dict) #536 1260 229 396
    print len(pART_dict)+len(pATTR_dict)+len(pFBR_dict)+len(pFSN_dict)
    print len(ART_dict)+len(ATTR_dict)+len(FBR_dict)+len(FSN_dict) # 587 1335 271 391

    print "test-train term count"
    print len( ART_dict.viewkeys() - tART_dict.viewkeys())
    print len( ATTR_dict.viewkeys() - tATTR_dict.viewkeys())
    print len( FBR_dict.viewkeys() - tFBR_dict.viewkeys())
    print len( FSN_dict.viewkeys() - tFSN_dict.viewkeys())

    print "Terms identified in test-train"
    print len( (ART_dict.viewkeys() - tART_dict.viewkeys()) & pART_dict.viewkeys())
    print len( (ATTR_dict.viewkeys() - tATTR_dict.viewkeys()) & pATTR_dict.viewkeys())
    print len( (FBR_dict.viewkeys() - tFBR_dict.viewkeys()) & pFBR_dict.viewkeys())
    print len( (FSN_dict.viewkeys() - tFSN_dict.viewkeys()) & pFSN_dict.viewkeys())

    ################################
    # New identified terms
    ################################
    print "New identified terms"
    print len((pART_dict.viewkeys() - tART_dict.viewkeys()) - ART_dict.viewkeys())
    print len((pATTR_dict.viewkeys() - tATTR_dict.viewkeys()) - ATTR_dict.viewkeys())
    print len((pFBR_dict.viewkeys() - tFBR_dict.viewkeys()) - FBR_dict.viewkeys())
    print len((pFSN_dict.viewkeys() - tFSN_dict.viewkeys()) - FSN_dict.viewkeys())

    writeJSON(setToDict((pART_dict.viewkeys() - tART_dict.viewkeys()) - ART_dict.viewkeys()),
     setToDict((pATTR_dict.viewkeys() - tATTR_dict.viewkeys()) - ATTR_dict.viewkeys()), 
     setToDict((pFBR_dict.viewkeys() - tFBR_dict.viewkeys()) - FBR_dict.viewkeys()), 
     setToDict((pFSN_dict.viewkeys() - tFSN_dict.viewkeys()) - FSN_dict.viewkeys()),
     "new_in_predict.json")

    writeJSON(tART_dict, tATTR_dict, tFBR_dict, tFSN_dict, "terms_in_train_sent.json")
    writeJSON(ART_dict, ATTR_dict, FBR_dict, FSN_dict, "terms_in_test_sent.json")
    writeJSON(pART_dict, pATTR_dict, pFBR_dict, pFSN_dict, "terms_in_pred.json")

'''
    writeToCSV(pART_dict, "predART.csv")
    writeToCSV(pATTR_dict, "predATTR.csv")
    writeToCSV(pFSN_dict, "predFSN.csv")
    writeToCSV(pFBR_dict, "predFBR.csv")

    writeToCSV(ART_dict, "testART.csv")
    writeToCSV(ATTR_dict, "testATTR.csv")
    writeToCSV(FSN_dict, "testFSN.csv")
    writeToCSV(FBR_dict, "testFBR.csv")

    writeToCSV(tART_dict, "trainART.csv")
    writeToCSV(tATTR_dict, "trainATTR.csv")
    writeToCSV(tFSN_dict, "trainFSN.csv")
    writeToCSV(tFBR_dict, "trainFBR.csv")
'''
