#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 21:45:47 2018

@author: mi0240
"""
import os
import json
import random
import pickle
import math

TESTDATA_FRACTION = 0.25

with open("gloss_formatted_ngrams.txt") as json_file:
    json_data=json.load(json_file)

def fashionTerms(termdict,keys):
    for key,values in json_data[keys].iteritems():
        termdict[key.encode('ascii','ignore')]= values
    return delete_amiguous_keys(termdict)


def delete_amiguous_keys(termdict):
    #anti, beach, ring, body, classic, dark, desirable, double, drop, dry, 
    #fit, glass, large, neutral, shade, tank, walking, print ,pu, rag, 
    # small, medium ,large
    delKeys=['alb','attire','brace','hose', 'lid', 'massage','mule', 
            'overall','platform','pump', 'rig' ,'separate', 'switch', 'trial',
            'washing','wash','abs','animal','auto','back','balm',
            'bar','beat', 'both','case','cement', 'cleaning','cold',
            'cut','daily','deep','emotional','essential','formulation', 'free',
            'fresh','full', 'heavy', 'high', 'hot', 'humid', 'iron',
            'jet','king','lean','light','length', 'long' ,'loose' ,'low',
            'media','medium','modern','motion','multiple','natural', 'none',
            'norma','normal','number' ,'oak','oat', 'office', 'oil','open', 'original',
            'outdoor', 'placement','plain','plastic', 'pop', 'powder', 'primer',
            'quality','round','sand','screen','set','short','single','small', 
            'so good','stick','straight', 'strong','touch', 'type' ,'vent',
            'work','aba', 'cord','flash', 'hair','other', 'rep', 'tag', 'web',
            'ball','bike', 'box', 'change', 'corn', 'court', 'dinner','feed',
            'flight','fly','morning','monkey','pea','ski' ,'street',
            'swim','tail','tails']
    for i in delKeys :
        try :
            del termdict[i]
            #print i
        except :
            pass
    return termdict



def split_terms(term_dict):
    totalTerms = len(set(term_dict))         # a sequence or set will work here.
    #testSize = int(len(totalTerms)*0.25)# set the number to select here.
    key_max = max(term_dict.keys(), key=(lambda k: term_dict[k]))
    key_min = min(term_dict.keys(), key=(lambda k: term_dict[k]))
    max_val = term_dict[key_max]
    min_val = term_dict[key_min]
    num_bins = int(math.sqrt(totalTerms))
    bin_width = int(max_val-min_val)/num_bins
    start = min_val
    test_split=set()
    train_split=set()
    for nb in range(num_bins-1):
        #get datas of [min,max)
        tmp={}
        for key,value in term_dict.iteritems():
            if term_dict[key]>=start and term_dict[key]< start+bin_width:
                tmp[key]=value
        test_split.update(random.sample(set(tmp), int(len(tmp)*TESTDATA_FRACTION)))
        train_split.update( set(tmp)-set(test_split)) 
        start = start + bin_width
    remaining_set = set(set(term_dict.keys()) -train_split-test_split)
    test_split.update(random.sample(remaining_set, int(len(remaining_set)*TESTDATA_FRACTION)))
    train_split.update( remaining_set-set(test_split)) 
    return train_split, test_split 



def count_terms(recalldict,lists):
    tmp={}
    for items in lists:
        tmp[items]=recalldict[items]
    return tmp


def save_in_json(allkeys, FBR, FSN, ART, ATTR, jsonfile ):
    multikey = {}
    multikey['fabric'] = count_terms(allkeys, set(allkeys.keys()) & FBR)
    multikey['fashion_strings'] = count_terms(allkeys, set(allkeys.keys()) & FSN)
    multikey['article_types'] = count_terms(allkeys, set(allkeys.keys()) & ART)
    multikey['attribute_types'] = count_terms(allkeys, set(allkeys.keys()) & ATTR)

    with open(jsonfile, 'w') as outfile:
        json.dump(multikey, outfile, sort_keys=True, indent=4, separators=(',', ': '))



if __name__== '__main__':
    fabric ={}
    fashion_strings={}
    article_types={}
    attribute_types={}

    fabric = json_data['fabric']
    fashion_strings = json_data['fashion_strings']
    article_types = json_data['article_types']
    attribute_types = json_data['attribute_types']


    allkeys= dict ( fabric.items() + article_types.items() + 
            fashion_strings.items() + attribute_types.items())

    train_fabric, test_fabric = split_terms(fabric)
    train_fashion_strings, test_fashion_strings = split_terms(fashion_strings)
    train_article_types, test_article_types = split_terms(article_types)
    train_attribute_types, test_attribute_types = split_terms(attribute_types)
    test_fabric = test_fabric -train_fabric&test_fabric
    test_fashion_strings = test_fashion_strings - train_fashion_strings&test_fashion_strings
    test_article_types =test_article_types -train_article_types&test_article_types
    test_attribute_types =test_attribute_types - train_attribute_types&test_attribute_types


    print train_fabric & test_fabric
    print train_fashion_strings & test_fashion_strings
    print train_article_types & test_article_types
    print train_attribute_types & test_attribute_types

    print "FBR\t{}\t{}".format(len(train_fabric), len(test_fabric))
    print "FSN\t{}\t{}".format(len(train_fashion_strings), len(test_fashion_strings))
    print "ART\t{}\t{}".format(len(train_article_types), len(test_article_types))
    print "ATTR\t{}\t{}".format(len(train_attribute_types), len(test_attribute_types)) 

    save_in_json(allkeys, train_fabric, train_fashion_strings, 
            train_article_types, train_attribute_types, 'train_terms.json')
    save_in_json(allkeys, test_fabric, test_fashion_strings, 
            test_article_types, test_attribute_types,'test_terms.json')

