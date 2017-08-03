# encoding: utf-8
#!/usr/bin/env python
import os
import time
import numpy as np
import tensorflow as tf
from util import FLAGS

from model import Model
from load_data_v1 import get_vocab_dict, LoadTrainData, LoadTestData, get_word_vector
from tester import test

vocab_dict = get_vocab_dict()
word_vectors = get_word_vector()
vocab_size = len(vocab_dict)
#print("vocab_size: ",vocab_size)
#print("word_vector: ", len(word_vectors))

training_set = LoadTrainData(vocab_dict,
                             data_path=FLAGS.training_set,
                             query_len_threshold=FLAGS.query_len_threshold,
                             doc_len_threshold=FLAGS.doc_len_threshold,
                             batch_size=FLAGS.batch_size)

dev_set = LoadTestData(vocab_dict, FLAGS.dev_set,
                       query_len_threshold=FLAGS.query_len_threshold,
                       doc_len_threshold=FLAGS.doc_len_threshold,
                       batch_size= FLAGS.batch_size)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = FLAGS.GPU_rate

with tf.Session(config=config) as sess:
    timestamp = str(int(time.time()))
    print("timestamp: ", timestamp)
    model_name = "{}_lr{}_bz{}_poolsize{}_{}".format(FLAGS.flag,
                                            FLAGS.learning_rate,
                                            FLAGS.batch_size,
                                            FLAGS.pooling_size,
                                            timestamp)

    model = Model(max_query_word=FLAGS.query_len_threshold,
                  max_doc_word=FLAGS.doc_len_threshold,
                  word_vec_initializer=word_vectors,
                  batch_size=FLAGS.batch_size,
                  vocab_size=vocab_size,
                  embedding_size=FLAGS.embedding_dim,
                  learning_rate=FLAGS.learning_rate,
                  filter_size=FLAGS.filter_size,
                  keep_prob=FLAGS.keep_prob)

    log_dir = "../logs/" + model_name
    model_path = os.path.join(log_dir, "model.ckpt")
    os.mkdir(log_dir)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    steps = []
    train_DCG_3 = []
    train_DCG_5 = []
    train_DCG_full = []
    val_DCG_3 = []
    val_DCG_5 = []
    val_DCG_full = []
    test_DCG_3 = []
    test_DCG_5 = []
    test_DCG_full = []

    step = 0
    #total_steps = FLAGS.total_training_num // FLAGS.batch_size
    #print("total steps number: ", total_steps)
    num_epochs = FLAGS.num_epochs
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        #for i in range(10):
        #    features_local, queries, docs = sess.run([features_local_batch, query_batch, docs_batch])
        for batch_data in training_set.next_batch():
            features_local, (_, queries), (_, docs, labels)= batch_data
            feed_dict = {model.feature_local: features_local,
                         model.query: queries,
                         model.doc: docs,
                         model.label: labels}
            #_, loss, summary = sess.run([model.train_op, model.loss, model.merged_summary_op], feed_dict)
            _, loss, score_pos, score_neg, subs, summary =\
                sess.run([model.optimize_op, model.loss, model.score_pos,
                          model.score_neg, model.sub, model.merged_summary_op],
                         feed_dict)


            if step % FLAGS.validation_steps == 0:
                print(step, " - loss:", loss)
                train_set = LoadTestData(vocab_dict, FLAGS.train_set,
                                         query_len_threshold=FLAGS.query_len_threshold,
                                         doc_len_threshold=FLAGS.doc_len_threshold, batch_size=-1)

                print("On training set:\n")
                dcg_3, dcg_5, dcg_full = test(sess, model, train_set, filename=None)
                train_DCG_3.append(dcg_3)
                train_DCG_3.append(dcg_5)
                train_DCG_full.append(dcg_full)

                dev_set = LoadTestData(vocab_dict, FLAGS.dev_set, query_len_threshold=FLAGS.query_len_threshold,
                                       doc_len_threshold=FLAGS.doc_len_threshold, batch_size=FLAGS.batch_size)
                print("On validation set:\n")

                dcg_3, dcg_5, dcg_full = test(sess, model, dev_set, filename=None)
                val_DCG_3.append(dcg_3)
                val_DCG_5.append(dcg_5)
                val_DCG_full.append(dcg_full)

            step += 1
            if step == 4000 or step == 5000:
                saver = tf.train.Saver(tf.global_variables())
                saver_path = saver.save(sess, os.path.join(log_dir, "model.ckpt"), step)

    coord.request_stop()
    coord.join(threads)
