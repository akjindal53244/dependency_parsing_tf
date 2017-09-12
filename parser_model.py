import os
import time
import tensorflow as tf
import numpy as np
from base_model import Model
from params_init import xavier_weight_init
from utils.general_utils import Progbar
from utils.general_utils import get_minibatches, dump_pickle
from utils.feature_extraction import load_datasets, DataConfig, Flags
from utils.tf_utils import visualize_sample_embeddings


class ParserModel(Model):
    def __init__(self, config, word_embeddings, pos_embeddings):
        self.word_embeddings = word_embeddings
        self.pos_embeddings = pos_embeddings
        self.config = config
        self.build()


    def add_placeholders(self):

        self.word_input_placeholder = tf.placeholder(shape=[None, self.config.word_features_types],
                                                     dtype=tf.int32, name="inputs")
        self.pos_input_placeholder = tf.placeholder(shape=[None, self.config.pos_features_types],
                                                    dtype=tf.int32, name="inputs")
        self.labels_placeholder = tf.placeholder(shape=[None, self.config.num_classes],
                                                 dtype=tf.float32, name="labels")
        self.dropout_placeholder = tf.placeholder(shape=(), dtype=tf.float32, name="dropout")


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):

        feed_dict = {
            self.word_input_placeholder: inputs_batch[0],
            self.pos_input_placeholder: inputs_batch[1],
            self.dropout_placeholder: dropout
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict


    def add_embedding(self):
        self.word_embedding_matrix = tf.get_variable(shape=self.word_embeddings.shape, dtype=tf.float32,
                                                     initializer=tf.random_uniform_initializer(
                                                         minval=-0.1, maxval=0.1, dtype=tf.float32), trainable=False,
                                                     name="word_embedding_matrix")
        self.pos_embedding_matrix = tf.get_variable(shape=self.pos_embeddings.shape, dtype=tf.float32,
                                                    initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1,
                                                                                              dtype=tf.float32),
                                                    trainable=True, name="pos_embedding_matrix")

        word_context_embeddings = tf.nn.embedding_lookup(self.word_embedding_matrix, self.word_input_placeholder)
        pos_context_embeddings = tf.nn.embedding_lookup(self.pos_embedding_matrix, self.pos_input_placeholder)

        word_embeddings = tf.reshape(word_context_embeddings,
                                     [-1, self.config.word_features_types * self.config.embedding_dim],
                                     name="word_context_embeddings")
        pos_embeddings = tf.reshape(pos_context_embeddings,
                                    [-1, self.config.pos_features_types * self.config.embedding_dim],
                                    name="pos_context_embeddings")

        embeddings = tf.concat(1, [word_embeddings, pos_embeddings])

        return embeddings


    def add_prediction_op(self):
        x = self.add_embedding()
        xavier_initializer = xavier_weight_init()
        w1 = xavier_initializer((self.config.num_features_types * self.config.embedding_dim,
                                 self.config.hidden_size), "w1")
        b1 = xavier_initializer((self.config.hidden_size,), "b1")

        w2 = xavier_initializer((self.config.hidden_size, self.config.num_classes), "w2")
        b2 = xavier_initializer((self.config.num_classes,), "b2")

        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x, w1) + b1),
                           keep_prob=self.dropout_placeholder, name="layer_1_activations")
        predictions = tf.matmul(h1, w2) + b2

        return predictions


    def add_loss_op(self, pred):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            pred, self.labels_placeholder), name="avg_batch_loss")
        return loss


    def add_training_op(self, loss):
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op


    def get_word_pos_inputs(self, inputs_batch):  # inputs_batch : list([list(word_id), list(pos_id)])
        # inputs_batch: [ [[1,2], [3,4]], [[5,6],[7,8]], [[9,10],[11,12]] ]
        inputs_batch = np.asarray(inputs_batch)
        word_inputs_batch, pos_inputs_batch = np.split(inputs_batch, 2, 1)
        word_inputs_batch = np.squeeze(word_inputs_batch)  # removes extra dimenstion -> convert 3-d to 2-d matrix
        pos_inputs_batch = np.squeeze(pos_inputs_batch)
        return word_inputs_batch, pos_inputs_batch


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        word_inputs_batch, pos_inputs_batch = inputs_batch
        feed = self.create_feed_dict([word_inputs_batch, pos_inputs_batch], labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss


    def run_valid_epoch(self, sess, data, dataset):
        sentences = data
        rem_sentences = [sentence for sentence in sentences]
        [sentence.clear_prediction_dependencies() for sentence in sentences]
        [sentence.clear_children_info() for sentence in sentences]

        while len(rem_sentences) != 0:
            curr_batch_size = min(dataset.model_config.batch_size, len(rem_sentences))
            batch_sentences = rem_sentences[:curr_batch_size]

            enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                               batch_sentences]
            enable_count = np.count_nonzero(enable_features)

            while enable_count > 0:
                curr_sentences = [sentence for i, sentence in enumerate(batch_sentences) if enable_features[i] == 1]

                # get feature for each sentence
                # call predictions -> argmax
                # store dependency and left/right child
                # update state
                # repeat

                curr_inputs = [
                    dataset.feature_extractor.extract_for_current_state(sentence, dataset.word2idx, dataset.pos2idx) for
                    sentence in curr_sentences]
                word_inputs_batch = [curr_inputs[i][0] for i in range(len(curr_inputs))]
                pos_inputs_batch = [curr_inputs[i][1] for i in range(len(curr_inputs))]

                predictions = sess.run(self.pred,
                                       feed_dict=self.create_feed_dict([word_inputs_batch, pos_inputs_batch]))
                legal_labels = np.asarray([sentence.get_legal_labels() for sentence in curr_sentences],
                                          dtype=np.float32)
                legal_transitions = np.argmax(predictions + 1000 * legal_labels, axis=1)

                # update left/right children so can be used for next feature vector
                [sentence.update_child_dependencies(transition) for (sentence, transition) in
                 zip(curr_sentences, legal_transitions) if transition != 2]

                # update state
                [sentence.update_state_by_transition(legal_transition, gold=False) for (sentence, legal_transition) in
                 zip(curr_sentences, legal_transitions)]

                enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                                   batch_sentences]
                enable_count = np.count_nonzero(enable_features)

            # Reset stack and buffer
            [sentence.reset_to_initial_state() for sentence in batch_sentences]
            rem_sentences = rem_sentences[curr_batch_size:]


    def get_UAS(self, data):
        correct_tokens = 0
        all_tokens = 0
        for sentence in data:
            # reset each predicted head before evaluation
            [token.reset_predicted_head_id() for token in sentence.tokens]

            head = [-2] * len(sentence.tokens)
            # assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
            for h, t, in sentence.predicted_dependencies:
                head[t.token_id] = h.token_id
            correct_tokens += sum([1 if token.head_id == head[i] else 0 for (i, token) in enumerate(sentence.tokens)])
            all_tokens += len(sentence.tokens)

        UAS = correct_tokens / float(all_tokens)
        return UAS


    def run_epoch(self, sess, config, dataset):
        prog = Progbar(target=1 + len(dataset.train_inputs[0]) / config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs, dataset.train_targets],
                                                               config.batch_size, is_multi_feature_input=True)):
            # print "input, outout: {}, {}".format(np.array(train_x).shape, np.array(train_y).shape)
            # np.savetxt("./data/my_model_input", train_x, fmt='%.0f')
            # np.savetxt("./data/my_model_output", train_y, fmt='%.0f')
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        self.run_valid_epoch(sess, dataset.valid_data, dataset)
        valid_UAS = self.get_UAS(dataset.valid_data)
        print "- dev UAS: {:.2f}".format(valid_UAS * 100.0)
        return valid_UAS


    def fit(self, sess, saver, config, dataset):
        best_valid_UAS = 0
        for epoch in range(config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            valid_UAS = self.run_epoch(sess, config, dataset)
            if valid_UAS > best_valid_UAS:
                best_valid_UAS = valid_UAS
                if saver:
                    print "New best dev UAS! Saving model.."
                    saver.save(sess,
                               os.path.join(DataConfig.data_dir_path, DataConfig.model_dir, DataConfig.model_name))
            print


def highlight_string(temp):
    print 80 * "="
    print temp
    print 80 * "="


def main(flag, load_existing_vocab=False):
    highlight_string("INITIALIZING")
    print "loading data.."

    dataset = load_datasets(load_existing_vocab)
    config = dataset.model_config

    print "word vocab Size: {}".format(len(dataset.word2idx))
    print "pos vocab Size: {}".format(len(dataset.pos2idx))
    print "Training Size: {}".format(len(dataset.train_inputs[0]))
    print "valid data Size: {}".format(len(dataset.valid_data))

    if not os.path.exists(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir)):
        os.makedirs(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))

    with tf.Graph().as_default() as graph:
        print "Building network...",
        start = time.time()
        with tf.variable_scope("model") as model_scope:
            model = ParserModel(config, dataset.word_embedding_matrix, dataset.pos_embedding_matrix)
            """
            model_scope.reuse_variables()
                -> no need to call tf.variable_scope(model_scope, reuse = True) again
                -> directly access variables & call functions inside this block itself.
            """
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # np.savetxt("./data/my_model_emb.txt", model.pretrained_embeddings, fmt='%.2f')

            sess.run(init)
            if flag == Flags.TRAIN:

                # call 'assignment' after 'init' only, else 'assignment' will get reset by 'init'
                sess.run(tf.assign(model.word_embedding_matrix, model.word_embeddings))
                sess.run(tf.assign(model.pos_embedding_matrix, model.pos_embeddings))

                highlight_string("TRAINING")
                model.print_trainable_varibles()

                model.fit(sess, saver, config, dataset)

                # visualize trained embeddings after complete training (not after each epoch)
                with tf.variable_scope(model_scope, reuse=True):
                    pos_emb = tf.get_variable("pos_embedding_matrix",
                                              [len(dataset.pos2idx.keys()), dataset.model_config.embedding_dim])
                    visualize_sample_embeddings(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir),
                                                dataset.pos2idx.keys(), dataset.pos2idx, pos_emb)

                # Testing
                highlight_string("Testing")
                print "Restoring best found parameters on dev set"
                saver.restore(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                                                 DataConfig.model_name))
                model.run_valid_epoch(sess, dataset.test_data, dataset)
                test_UAS = model.get_UAS(dataset.test_data)
                print "test UAS: {}".format(test_UAS * 100)
            else:
                # saver = tf.train.import_meta_graph(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir,
                #                                                 DataConfig.model_name) + ".meta")

                ckpt_path = tf.train.latest_checkpoint(os.path.join(DataConfig.data_dir_path,
                                                               DataConfig.model_dir))
                if ckpt_path is not None:
                    saver.restore(sess, ckpt_path)
                    print "Found checkpoint!"
                else:
                    print "No checkpoint found!"
                model.run_valid_epoch(sess, dataset.test_data, dataset)
                test_UAS = model.get_UAS(dataset.test_data)
                print "test UAS: {}".format(test_UAS * 100)

if __name__ == '__main__':
    main(Flags.TEST, load_existing_vocab=True)
