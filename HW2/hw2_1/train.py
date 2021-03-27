import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import random
import pickle

from preprocess import pad_sequences
from seq2seq import Seq2Seq_Model
from bleu_eval import BLEU

if __name__ == "__main__" :
    np.random.seed(10000)
    random.seed(10000)
    tf.set_random_seed(10000)

    test_video_feat_folder = sys.argv[1]
    testing_label_json_file = sys.argv[2]
    output_testset_filename = sys.argv[3]

    tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
    tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
    tf.app.flags.DEFINE_integer('dim_video_feat', 4096, 'Feature dimensions of each video frame')
    tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    tf.app.flags.DEFINE_integer('batch_size', 29, 'Batch size')
    tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Max. global gradient norm to clip')

    tf.app.flags.DEFINE_boolean('use_attention', True, 'Enable attention')
    
    tf.app.flags.DEFINE_boolean('beam_search', False, 'Enable beam search')
    tf.app.flags.DEFINE_integer('beam_size', 5, 'Size of beam search')

    tf.app.flags.DEFINE_integer('max_encoder_steps', 64, 'Max. steps of encoder')
    tf.app.flags.DEFINE_integer('max_decoder_steps', 15, 'Max. steps of decoder')

    tf.app.flags.DEFINE_integer('sample_size', 1450, 'Sampled data size of training epochs')
    tf.app.flags.DEFINE_integer('dim_video_frame', 80, 'Number of frame in each video')

    tf.app.flags.DEFINE_integer('num_epochs', 203, 'Maximum # of training epochs')
    tf.app.flags.DEFINE_string('model_dir', 'models/', 'Path to save model checkpoints')
    tf.app.flags.DEFINE_string('model_name', 's2s.ckpt', 'File name used for model checkpoints')

    FLAGS = tf.app.flags.FLAGS
    
    num_top_BLEU = 10
    top_BLEU = []

    print ('Reading pickle files...')
    word2index = pickle.load(open('word2index.obj', 'rb'))
    index2word = pickle.load(open('index2word.obj', 'rb'))
    
    video_IDs = pickle.load(open('video_IDs.obj', 'rb'))
    video_cap_dict = pickle.load(open('video_caption_dict.obj', 'rb'))
    video_feat_dict = pickle.load(open('video_feat_dict.obj', 'rb'))
    index2word_series = pd.Series(index2word)

    print ('Reading testing files...')
    test_video_feat_filenames = os.listdir(test_video_feat_folder)
    test_video_feat_filepaths = [(test_video_feat_folder + filename) for filename in test_video_feat_filenames]
    
    test_video_IDs = [filename[:-4] for filename in test_video_feat_filenames]

    test_video_feat_dict = {}
    for filepath in test_video_feat_filepaths :
        test_video_feat = np.load(filepath)
        
        sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
        test_video_feat = test_video_feat[sampled_video_frame]

        test_video_ID = filepath[: -4].replace(test_video_feat_folder, "")
        test_video_feat_dict[test_video_ID] = test_video_feat
    
    test_video_cap = json.load(open(testing_label_json_file, 'r'))

    with tf.Session() as sess :
        model = Seq2Seq_Model(
            rnn_size = FLAGS.rnn_size, 
            num_layers = FLAGS.num_layers, 
            dim_video_feat = FLAGS.dim_video_feat, 
            embedding_size = FLAGS.embed_size, 
            learning_rate = FLAGS.learning_rate, 
            word_to_index = word2index, 
            mode = 'train', 
            max_grad_norm = FLAGS.max_grad_norm, 
            use_attention = FLAGS.use_attention, 
            beam_search = FLAGS.beam_search, 
            beam_size = FLAGS.beam_size,
            max_encoder_steps = FLAGS.max_encoder_steps, 
            max_decoder_steps = FLAGS.max_decoder_steps
        )
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else :
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph = sess.graph)

        for epoch in range(FLAGS.num_epochs) :
            start_time = time.time()

            sampled_ID_cap = []
            for ID in video_IDs :
                sampled_cap = random.sample(video_cap_dict[ID], 1)[0]
                sampled_video_frame = sorted(random.sample(range(FLAGS.dim_video_frame), FLAGS.max_encoder_steps))
                sampled_video_feat = video_feat_dict[ID][sampled_video_frame]
                sampled_ID_cap.append((sampled_video_feat, sampled_cap))

            random.shuffle(sampled_ID_cap)

            for batch_start, batch_end in zip(range(0, FLAGS.sample_size, FLAGS.batch_size), range(FLAGS.batch_size, FLAGS.sample_size, FLAGS.batch_size)) :
                batch_sampled_ID_cap = sampled_ID_cap[batch_start : batch_end]
                batch_video_feats = [elements[0] for elements in batch_sampled_ID_cap]
                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size
                batch_cap = np.array(["<BOS> " + elements[1] for elements in batch_sampled_ID_cap])

                for index, caption in enumerate(batch_captions) :
                    cap_words = caption.lower().split(" ")
                    if len(cap_words) < FLAGS.max_decoder_steps :
                        batch_cap[index] = batch_cap[index] + " <EOS>"
                    else :
                        new_cap = ""
                        for i in range(FLAGS.max_decoder_steps - 1) :
                            new_cap = new_cap + cap_words[i] + " "
                        batch_cap[index] = new_cap + "<EOS>"

                batch_cap_words_index = []
                for caption in batch_cap :
                    words_index = []
                    for cap_words in caption.lower().split(' ') :
                        if cap_words in word2index :
                            words_index.append(word2index[cap_words])
                        else :
                            words_index.append(word2index['<UNK>'])
                    batch_cap_words_index.append(words_index)

                batch_capt_matrix = pad_sequences(batch_cap_words_index, padding = 'post', maxlen = FLAGS.max_decoder_steps)
                batch_cap_length = [len(x) for x in batch_cap_matrix]
               
                loss, summary = model.train(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame, 
                    batch_cap_matrix, 
                    batch_cap_length)
            print()
               
            test_captions = []
            
            for batch_start, batch_end in zip(range(0, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size), range(FLAGS.batch_size, len(test_video_IDs) + FLAGS.batch_size, FLAGS.batch_size)) :
                if batch_end < len(test_video_IDs) :
                    batch_sampled_ID = np.array(test_video_IDs[batch_start : batch_end])
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]
                else :
                    batch_sampled_ID = test_video_IDs[batch_start : batch_end]
                    for _ in range(batch_end - len(test_video_IDs)) :
                        batch_sampled_ID.append(test_video_IDs[-1])
                    batch_sampled_ID = np.array(batch_sampled_ID)
                    batch_video_feats = [test_video_feat_dict[x] for x in batch_sampled_ID]

                batch_video_frame = [FLAGS.max_decoder_steps] * FLAGS.batch_size 

                batch_cap_words_index, logits = model.infer(
                    sess, 
                    batch_video_feats, 
                    batch_video_frame) 

                if batch_end < len(test_video_IDs) :
                    batch_cap_words_index = batch_cap_words_index
                else :
                    batch_cap_words_index = batch_cap_words_index[:len(test_video_IDs) - batch_start]

                for index, test_cap_words_index in enumerate(batch_cap_words_index) :

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

                        
            df = pd.DataFrame(np.array([test_video_IDs, test_captions]).T)
            df.to_csv(output_testset_filename, index = False, header = False)

            result = {}
            with open(output_testset_filename, 'r') as f :
                for line in f :
                    line = line.rstrip()
                    test_id, caption = line.split(',')
                    result[test_id] = caption
                    
            bleu = []
            for item in test_video_cap :
                score_per_video = []
                captions = [x.rstrip('.') for x in item['caption']]
                score_per_video.append(BLEU(result[item['id']],captions,True))
                bleu.append(score_per_video[0])
            average = sum(bleu) / len(bleu)

            if (len(top_BLEU) < num_top_BLEU) :
                top_BLEU.append(average)
                print ("Saving model with BLEU@1: %.4f ..." %(average))
                model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            else :
                if (average > min(top_BLEU)) :
                    top_BLEU.remove(min(top_BLEU))
                    top_BLEU.append(average)
                    print ("Saving model with BLEU@1: %.4f ..." %(average))
                    model.saver.save(sess, './models/model' + str(average)[2:6], global_step=epoch)
            top_BLEU.sort(reverse=True)
            print ("Top [%d] BLEU: " %(num_top_BLEU), ["%.4f" % x for x in top_BLEU])

            print ("Epoch %d/%d, loss: %.6f, Avg. BLEU@1: %.6f, Elapsed time: %.2fs" %(epoch, FLAGS.num_epochs, loss, average, (time.time() - start_time)))