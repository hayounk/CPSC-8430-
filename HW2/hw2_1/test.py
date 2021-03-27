import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import json
import random
import pickle

from seq2seq import Seq2Seq_Model
from bleu_eval import BLEU

def inference() :
    np.random.seed(10000)
    random.seed(10000)
    tf.set_random_seed(10000)

    test_folder = sys.argv[1]
    output_testset_filename = sys.argv[2]

    test_id_filename = test_folder + 'id.txt'
    test_video_feat_folder = test_folder + '/feat/'

    tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('dim_video_feat', 4096, 'Feature dimensions of each video frame')
    tf.app.flags.DEFINE_integer('embed_size', 1024, 'Embed dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')
    
    tf.app.flags.DEFINE_boolean('use_attention', True, 'Enable attention')

    tf.app.flags.DEFINE_boolean('beam_search', True, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 1, 'Size of beam search')
    
    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 1450, 'Sampled data size of training epochs')
    tf.app.flags.DEFINE_integer('dim_video_frame', 80, 'Number of frame in each video')

    tf.app.flags.DEFINE_integer('num_epochs', 64, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_video_IDs = []
    with open(test_id_filename, 'r') as f :
        for line in f :
            line = line.rstrip()
            test_video_IDs.append(line)

    test_video_feat_filenames = os.listdir(test_video_feat_folder)
    test_video_feat_filepaths = [(test_video_feat_folder + filename) for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths :
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat

    with tf.Session() as sess :
        model = Seq2Seq_Model(
            rnn_size = FLAGS.rnn_size, 
            num_layers = FLAGS.num_layers, 
            dim_video_feat = FLAGS.dim_video_feat, 
            embed_size = FLAGS.embed_size, 
            learning_rate = FLAGS.learning_rate, 
            word_to_idx = word2index, 
            mode = 'decode', 
            max_gradient_norm = FLAGS.max_gradient_norm, 
            use_attention = FLAGS.use_attention, 
            beam_search = FLAGS.beam_search, 
            beam_size = FLAGS.beam_size,
            max_encoder_steps = FLAGS.max_encoder_steps, 
            max_decoder_steps = FLAGS.max_decoder_steps
        )
        
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else :
            raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

        test_cap = []
        for ID in test_video_IDs :
            test_video_feat = test_video_feat_dict[ID].reshape(1, FLAGS.max_encoder_steps, FLAGS.dim_video_feat)
            test_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size

            test_cap_words_index, logits = model.infer(
                sess, 
                test_video_feat, 
                test_video_frame)

            if FLAGS.beam_search :
                logits = np.array(logits).reshape(-1, FLAGS.beam_size)
                max_logits_index = np.argmax(np.sum(logits, axis = 0))
                predict_list = np.ndarray.tolist(test_cap_words_index[0, :, max_logits_index])
                predict_seq = [index2word[index] for index in predict_list]
                test_cap_words = predict_seq
            else :
                test_cap_words_index = np.array(test_cap_words_index).reshape(-1)
                test_cap_words = index2word_series[test_cap_words_index]
                test_cap = ' '.join(test_cap_words) 

            test_cap = ' '.join(test_cap_words)
            test_cap = test_cap.replace('<BOS> ', '')
            test_cap = test_cap.replace('<EOS>', '')
            test_cap = test_cap.replace(' <EOS>', '')
            test_cap = test_cap.replace('<PAD> ', '')
            test_cap = test_cap.replace(' <PAD>', '')
            test_cap = test_cap.replace(' <UNK>', '')
            test_cap = test_cap.replace('<UNK> ', '')

            if (test_cap == "") :
                test_cap = '.'

            if ID in ["klteYv1Uv9A_27_33.avi", "UbmZAe5u5FI_132_141.avi", "wkgGxsuNVSg_34_41.avi", "JntMAcTlOF0_50_70.avi", "tJHUH9tpqPg_113_118.avi"] :
                print(ID, test_cap)

            test_cap.append(test_cap)
        
        df = pd.DataFrame(np.array([test_video_IDs, test_cap]).T)
        df.to_csv(output_testset_filename, index = False, header = False)
    

if __name__ == "__main__" :
    inference()