import tensorflow as tf
import numpy as np
import os
import sys
import random

class Seq2Seq_Model() :
    def __init__(self, rnn_size, num_layers, dim_video_feat, embed_size, 
                    learning_rate, word_to_idx, mode, max_gradient_norm, 
                    use_attention, beam_search, beam_size, 
                    max_encoder_steps, max_decoder_steps) :
        
        tf.set_random_seed(9487)
        np.random.seed(9487)
        random.seed(9487)

        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.dim_video_feat = dim_video_feat
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.word_to_idx = word_to_idx
        self.mode = mode
        self.max_gradient_norm = max_gradient_norm
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.max_encoder_steps = max_encoder_steps
        self.max_decoder_steps = max_decoder_steps
        self.vocab_size = len(self.word_to_idx)

        self.build_model()

    def _create_rnn_cell(self) :
        def single_rnn_cell() :
            single_cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob = self.keep_prob_placeholder, seed = 9487)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self) :
        tf.set_random_seed(9487)
        np.random.seed(9487)
        random.seed(9487)

        self.encoder_inputs = tf.placeholder(tf.float32, [None, None, None], name = 'encoder_inputs')
        self.encoder_inputs_len = tf.placeholder(tf.int32, [None], name = 'encoder_inputs_len')

        self.batch_size = tf.placeholder(tf.int32, [], name = 'batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name = 'keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name = 'decoder_targets')
        self.decoder_targets_len = tf.placeholder(tf.int32, [None], name = 'decoder_targets_len')

        self.max_target_sequence_len = tf.reduce_max(self.decoder_targets_len, name = 'max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_len, self.max_target_sequence_len, dtype = tf.float32, name = 'masks')


        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE) :
            encoder_inputs_flat = tf.reshape(self.encoder_inputs, [-1, self.dim_video_feat])
            encoder_inputs_embed = tf.layers.dense(encoder_inputs_flat, self.embed_size, use_bias = True)
            encoder_inputs_embed = tf.reshape(encoder_inputs_embed, [self.batch_size, self.max_encoder_steps, self.rnn_size])

            encoder_cell = self._create_rnn_cell()

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, encoder_inputs_embed, 
                sequence_len = self.encoder_inputs_len, 
                dtype = tf.float32)
    
        with tf.variable_scope('decoder', reuse = tf.AUTO_REUSE) :
            encoder_inputs_len = self.encoder_inputs_len

            if self.beam_search :
                print("Using beamsearch")
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier = self.beam_size)
                encoder_state = tf.contrib.framework.nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_len, multiplier = self.beam_size)

            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            projection_layer = tf.layers.Dense(units = self.vocab_size, kernel_init = tf.truncated_normal_init(mean = 0.0, stddev = 0.1, seed = 9487))
            
            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, self.rnn_size], -0.1, 0.1, seed = 9487), name = 'embedding_decoder')


            decoder_cell = self._create_rnn_cell()

            if self.use_attention :
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units = self.rnn_size, 
                    memory = encoder_outputs, 
                    normalize = True,
                    memory_sequence_len = encoder_inputs_len)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell = decoder_cell, 
                    attention_mechanism = attention_mechanism, 
                    attention_layer_size = self.rnn_size, 
                    name = 'Attention_Wrapper')

                decoder_initial_state = decoder_cell.zero_state(batch_size = batch_size, dtype = tf.float32).clone(cell_state = encoder_state)
            else :
                decoder_initial_state = encoder_state

            output_layer = tf.layers.Dense(self.vocab_size, kernel_init = tf.truncated_normal_init(mean = 0.0, stddev = 0.1, seed = 9487))

            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], self.word_to_index['<BOS>']), ending], 1)
            
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)

            training_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs = decoder_inputs_embedded, 
                sequence_len = self.decoder_targets_len, 
                time_major = False, name = 'training_helper')
            
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell = decoder_cell, helper = training_helper, 
                initial_state = decoder_initial_state, 
                output_layer = output_layer)
            
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = training_decoder, 
                impute_finished = True, 
                maximum_iterations = self.max_target_sequence_len)

            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis = -1, name = 'decoder_pred_train')

            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits = self.decoder_logits_train, 
                targets = self.decoder_targets, 
                weights = self.mask)

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            trainable_params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_index['<BOS>']
            end_token = self.word_to_index['<EOS>']
            
            if self.beam_search :
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell = decoder_cell, 
                    embedding = embedding_decoder,
                    start_tokens = start_tokens, 
                    end_token = end_token,
                    initial_state = decoder_initial_state,
                    beam_width = self.beam_size,
                    output_layer = output_layer)
            else :
                inference_decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding = embedding_decoder, 
                    start_tokens = start_tokens, 
                    end_token = end_token)
                
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell = decoder_cell, 
                    helper = inference_decoding_helper, 
                    initial_state = decoder_initial_state, 
                    output_layer = output_layer)

            inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder = inference_decoder, 
                maximum_iterations = self.max_decoder_steps)

            if self.beam_search :
                self.decoder_predict_decode = inference_decoder_outputs.predicted_ids
                self.decoder_predict_logits = inference_decoder_outputs.beam_search_decoder_output
            else :
                self.decoder_predict_decode = tf.expand_dims(inference_decoder_outputs.sample_id, -1)
                self.decoder_predict_logits = inference_decoder_outputs.rnn_output

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 50)

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_len) :
        feed_dict = {self.encoder_inputs : encoder_inputs,
                      self.encoder_inputs_len : encoder_inputs_len,
                      self.decoder_targets : decoder_targets,
                      self.decoder_targets_len : decoder_targets_len,
                      self.keep_prob_placeholder : 0.8,
                      self.batch_size : len(encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict = feed_dict)
        return loss, summary

    def eval(self, sess, encoder_inputs, encoder_inputs_length, decoder_targets, decoder_targets_len) :
        feed_dict = {self.encoder_inputs : encoder_inputs,
                      self.encoder_inputs_len : encoder_inputs_len,
                      self.decoder_targets : decoder_targets,
                      self.decoder_targets_len : decoder_targets_len,
                      self.keep_prob_placeholder : 1.0,
                      self.batch_size : len(encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict = feed_dict)
        return loss, summary

    def infer(self, sess, encoder_inputs, encoder_inputs_len) :
        feed_dict = {self.encoder_inputs : encoder_inputs,
                      self.encoder_inputs_len : encoder_inputs_len,
                      self.keep_prob_placeholder : 1.0,
                      self.batch_size : len(encoder_inputs)}
        predict, logits = sess.run([self.decoder_predict_decode, self.decoder_predict_logits], feed_dict = feed_dict)
        return predict, logits