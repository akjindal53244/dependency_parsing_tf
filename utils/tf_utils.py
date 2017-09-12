import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def visualize_sample_embeddings(sess, log_dir, words, word2idx, embeddings):   # embedding -> tf.get_variable()
    list_idx = map(lambda word: word2idx[word], words)
    # sample_embeddings = tf.gather(embeddings, list_idx, name="my_embeddings")
    # sample_embeddings = embeddings[list_idx]

    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()

    metadata_path = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata_path, "w") as f:
        [f.write(word + "\n") for word in words]

    embedding_conf.tensor_name = embeddings.name  # embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    projector.visualize_embeddings(summary_writer, config)
    # summary_writer.close()



