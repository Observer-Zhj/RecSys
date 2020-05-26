# RecSys
推荐系统

对于多标签特征，
fm_tf_1.py中用-1补全，然后使用tf.nn.embedding_lookup，在tensorflow-gpu==0.12.0下是可行的；
但部分tensorflow版本的embedding_lookup不支持ids为-1，在fm_tf_2.py中使用tf.nn.embedding_lookup_sparse