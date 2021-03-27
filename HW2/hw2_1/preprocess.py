import numpy as np
import os
import sys
from os import listdir
import json
import pickle
import io

max_decoder_steps = 15

# Referenced from https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py.
def pad_sequences(sequences, max_len = None, dtype = 'int32',
                  padding = 'pre', truncating = 'pre', value = 0.) :
    if not hasattr(sequences, '__len__') :
        raise ValueError('ERROR!')
    length = []
    for x in sequences :
        if not hasattr(x, '__len__') :
            raise ValueError('ERROR!')
        length.append(len(x))

    num_samples = len(sequences)
    if max_len is None :
        max_len = np.max(length)

    sample_shape = tuple()
    for s in sequences :
        if len(s) > 0 :
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, max_len) + sample_shape) * value).astype(dtype)
    for index, s in enumerate(sequences) :
        if not len(s) :
            continue  
        if truncating == 'pre' :
            trunc = s[-max_len:]
        elif truncating == 'post' :
            trunc = s[:max_len]
        else:
            raise ValueError('ERROR!')

        trunc = np.asarray(trunc, dtype = dtype)
        if trunc.shape[1:] != sample_shape :
            raise ValueError('ERROR!')

        if padding == 'post' :
            x[index, :len(trunc)] = trunc
        elif padding == 'pre' :
            x[index, -len(trunc):] = trunc
        else :
            raise ValueError('ERROR!')
    return x

def build_dictionary(sentences, min_count) :
    word_counts = {}
    senences_count = 0
    for sentence in sentences :
        senences_count += 1
        for word in sentence.lower().split(' ') :
            word_counts[word] = word_counts.get(word, 0) + 1
    
    dictionary = [word for word in word_counts if word_counts[word] >= min_count]
    print ('Filtered words from %d to %d with min_count [%d]' % (len(word_counts), len(dictionary), min_count))

    index2word = {}
    index2word[0] = '<PAD>'
    index2word[1] = '<BOS>'
    index2word[2] = '<EOS>'
    index2word[3] = '<UNK>'

    word2index = {}
    word2index['<PAD>'] = 0
    word2index['<BOS>'] = 1
    word2index['<EOS>'] = 2
    word2index['<UNK>'] = 3

    for index, word in enumerate(dictionary) : 
        word2index[word] = index + 4
        index2word[index + 4] = word

    word_counts['<PAD>'] = senences_count
    word_counts['<BOS>'] = senences_count
    word_counts['<EOS>'] = senences_count
    word_counts['<UNK>'] = senences_count

    return word2index, index2word, dictionary

def filter_token(string) :
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    for c in filters :
        string = string.replace(c, '')
    return string

if __name__ == "__main__" :
    np.random.seed(9487)

    video_feat_folder = sys.argv[1]
    training_label_json_file = sys.argv[2]

    video_feat_filenames = listdir(video_feat_folder)
    video_feat_filepaths = [(video_feat_folder + filename) for filename in video_feat_filenames]

    video_IDs = [filename[:-4] for filename in video_feat_filenames]

    video_feat_dict = {}
    for filepath in video_feat_filepaths :
        video_feat = np.load(filepath)
        video_ID = filepath[: -4].replace(video_feat_folder, "")
        video_feat_dict[video_ID] = video_feat
    
    video_cap = json.load(open(training_label_json_file, 'r'))
    video_cap_dict = {}
    cap_corpus = []
    for video in video_cap :
        filtered_cap = [filter_token(sentence) for sentence in video["caption"]]
        video_cap_dict[video["id"]] = filtered_cap
        cap_corpus += filtered_cap

    word2index, index2word, dictionary = build_dictionary(cap_corpus, min_count = 3)
    
    pickle.dump(word2index, open('./word2index.obj', 'wb'))
    pickle.dump(index2word, open('./index2word.obj', 'wb'))

    ID_cap = []
    cap_words = []

    words_list = []
    for ID in video_IDs :
        for caption in video_cap_dict[ID] :
            ID_cap.append((video_feat_dict[ID], caption))
            words = caption.split()
            cap_words.append(words)
            for word in words :
                words_list.append(word)

    cap_words_set = np.unique(words_list, return_counts = True)[0]
    max_cap_len = max([len(words) for words in cap_words])
    avg_cap_len = np.mean([len(words) for words in cap_words])
    num_unique_tokens_cap = len(cap_words_set)

    print("np.shape(ID_caption): ", np.shape(ID_cap))
    print("Max. length of captions: ", max_cap_len)
    print("Avg. length of captions: ", avg_cap_len)
    print("Number of unique tokens of captions: ", num_unique_tokens_cap)

    print("Shape of features of first video: ", ID_cap[0][0].shape)
    print("ID of first video: ", video_IDs[0])
    print("Caption of first video: ", ID_cap[0][1])

    pickle.dump(video_IDs, open('video_IDs.obj', 'wb'))
    pickle.dump(video_caption_dict, open('video_caption_dict.obj', 'wb'))
    pickle.dump(video_feat_dict, open('video_feat_dict.obj', 'wb'))