# encoding: utf-8
import tensorflow as tf


flags = tf.app.flags

# Model parameters
flags.DEFINE_integer("filter_size", 128, "the num of filters of CNN")
flags.DEFINE_integer("embedding_dim", 100, "words embedding size")
flags.DEFINE_float("keep_prob", 0.8, "dropout keep prob")

# change each runing
flags.DEFINE_string("flag", "word_1", "word/char/drmm")
flags.DEFINE_string("save_dir", "../logs/char_4000_4_lr0.0005_bz128_poolsize80_1497961044", "save dir")
flags.DEFINE_string("predict_dir", "predict_word_1.csv", "predict result dir")


# Training / test parameters
flags.DEFINE_integer("query_len_threshold", 20, "threshold value of query length")
flags.DEFINE_integer("doc_len_threshold", 200, "threshold value of document length")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("num_epochs", 1, "number of epochs")
flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
flags.DEFINE_float("margin", 1.0, "cos margin")
flags.DEFINE_integer("pooling_size", 80, "pooling size")
'''
flags.DEFINE_string("training_set", "../../data/data_char/train.csv", "training set path")
flags.DEFINE_string("train_set", "../../data/data_char/train.json", "train set path")
flags.DEFINE_string("dev_set", "../../data/data_char/dev.json", "dev set path")
flags.DEFINE_string("vocab_path", "../../data/data_char/char_dict.txt", "vocab path")
flags.DEFINE_string("vectors_path", "../../data/data_char/vectors_char.txt", "vectors path")
'''
flags.DEFINE_float("validation_steps", 1000, "steps between validations")
flags.DEFINE_float("GPU_rate", 0.9, "steps between validations")

flags.DEFINE_string("training_set", "../../data/data_word/train.csv", "training set path")
flags.DEFINE_string("train_set", "../../data/data_word/train.test.json", "train set path")
flags.DEFINE_string("dev_set", "../../data/data_word/dev.json", "dev set path")
flags.DEFINE_string("vocab_path", "../../data/data_word/word_dict.txt", "vocab path")
flags.DEFINE_string("vectors_path", "../../data/data_word/vectors_word.txt", "vectors path")






FLAGS = flags.FLAGS

