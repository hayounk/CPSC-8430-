import json
import os
import numpy as np
import re
from collections import Counter
import argparse

def count(fileD) :
    counter = Counter()
    
    with open(fileD, 'r') as f :
        data = json.load(f)
    
    for captions in data :
        for sentence in captions['caption'] :
            sentence = sentence.lower()
            sentence = re.sub('[^a-z0-9_]', ' ', sentence)
            word = sentence.split()
            counter.update(word)
            
    counter.update(['<BOS>'])
    counter.update(['<EOS>'])
    
    return counter

def dictionary(fildD = None, counter = None) :
    if counter is None :
        counter = count(fildD)
    
    dic = {}
    
    for index, voca in enumerate(counter.most_common()) :
        dic[voca[0]] = index
    return dic, counter

def word2index(fileD, dic) :
    with open(fileD, 'r') as f :
        data = json.load(f)
    
    newD = []
    
    for captions in data :
        trans_cap = {}
        trans_cap['caption'] = []
        trans_cap['id'] = captions['id']
        trans_word = []
        
        for sentence in captions['caption'] :
            sentence = sentence.lower()
            sentence = re.sub('[^a-z0-9_]', ' ', sentence)
            word = sentence.split()
            
            for vocab in word :
                if vocab in dic :
                    trans_word.append(dic[vocab])
                else :
                    trans_word.append(dic['<UNK>'])
            
            trans_cap['caption'].append(trans_word)
        newD.append(trans_cap)
    return newD

def write_dic(counter, fname) :
    with open(fname, 'w') as f :
        for i, voca in enumerate(counter.most_common()) :
            f.write(str(i) + ' ' + voca[0] + ' ' + str(voca[1]) + '\n')
    return

def train_ex(data, fname) :
    with open(fname, 'w') as f :
        json.dump(data, f, sort_keys = True, indent = 4) 
        #<PAD> <BOS> <EOS> <UNK> total 4
    return

def count_read(fname) :
    if filename[-4:] == '.npy' :
        counter = np.load(fname)
    else : 
        counter = {}
        with open(fname, 'r') as f :
            for line in f :
                sent = line.split()
                counter[sent[1]] = int(s[2])
        
        counter = Counter(counter)
        return counter
    
def dic_read(fname) :
    if fname[-4:] == '.npy' :
        dic = np.load(fname) 
    else : 
        dic = {}
        with open(fname, 'r') as f :
            for line in f :
                sent = line.split()
                dic[sent[1]] = int(sent[0])
    return dic

def train_read(fname) :
    if fname[-4:] =='.npy' :
        data = np.load(fname)
    elif fname[-5:] == '.json' :
        with open(fname, 'r') as f :
            data = json.load(f)
    else :
        data = []
    return data 

def trim(counter, freq=None, num=None):
    cut = 0
    if freq is not None:
        sort_counter = counter.most_common()
        for i in range(len(sort_counter)):
            if sort_counter[i][1] < freq :
                cut = i
                break
    elif num is not None:
        sort_counter = counter.most_common()
        freq = sort_counter[num - 1][1]
        for i in range(len(sort_counter)):
            if sort_counter[i][1] < freq :
                cut = i
                break
    else:
        return counter, data

    new_count = counter.most_common(cut)
    new_count = Counter(dict(new_count))

    if '<EOS>' not in new_count:
        new_count.update(['<EOS>'])
    if '<BOS>' not in new_count:
        new_count.update(['<BOS>'])
    
    new_count.update(['<UNK>'])

    return new_count

def ent_write(fileD, dict_file, train_file) :
    dic, counter = dictionary(fileD) 
    counter = trim(counter, freq = 3)
    write_dic(counter, dict_file)
    
    dic, _ = dictionary(counter=counter)
    train = word2index(fileD, dic)
    train_ex(train, train_file)
    return

def ent_read(dict_file, train_file) :
    dic = read_dic(dict_file)
    train = train_read(train_file)
    return dic, train

def parse_arg() :
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_file')
    return parser.parse_args()

if __name__ == "__main__" :
    arg = parse_arg()
    
    dict_file = 'dictionary.txt'
    train_file = 'train_label.json'
    
    ent_write(arg.train_data_file, dict_file, train_file)