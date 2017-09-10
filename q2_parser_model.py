import os
import time
import tensorflow as tf
import cPickle
import numpy as np

from model import Model
from q2_initialization import xavier_weight_init
from utils.general_utils import Progbar
from utils.general_utils import get_minibatches
from utils.feature_extraction import load_datasets, DataConfig
from utils.parser_utils import minibatches, load_and_preprocess_data

class Config(object):

    num_features_types = 36
    num_classes = 3
    dropout = 0.5
    embedding_dim = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 1
    lr = 0.001


class ParserModel(Model):
    """
    Implements a feedforward neural network with an embedding layer and single hidden layer.
    This network will predict which transition should be applied to a given partial parse
    configuration.
    """

    def __init__(self, config, pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.config = config
        self.build()

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, n_classes), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.dropout_placeholder

        (Don't change the variable names)
        """
        self.input_placeholder = tf.placeholder(shape=[None, self.config.num_features_types], dtype=tf.int32, name="inputs")
        self.labels_placeholder = tf.placeholder(shape = [None, self.config.num_classes], dtype=tf.float32, name= "labels")
        self.dropout_placeholder = tf.placeholder(shape=(), dtype=tf.float32, name = "dropout")

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }


        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """

        feed_dict = {
            self.input_placeholder: inputs_batch,
            self.dropout_placeholder: dropout
        }

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch

        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
            - Creates an embedding tensor and initializes it with self.pretrained_embeddings.
            - Uses the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, n_features, embedding_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, n_features * embedding_size).

        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You can use tf.reshape to concatenate the vectors. See following link to understand
            what -1 in a shape means.
            https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, n_features*embed_size)
        """

        # self.embedding_matrix = tf.stack(self.pretrained_embeddings)

        self.embedding_matrix = tf.get_variable(shape=self.pretrained_embeddings.shape, dtype=tf.float32,
                                                initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1,
                                                                                          dtype=tf.float32),
                                                trainable=False, name="embedding_matrix")
        context_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.input_placeholder)

        embeddings = tf.reshape(context_embeddings, [-1, self.config.num_features_types * self.config.embedding_dim],
                                name="context_embeddings")

        return embeddings

    def add_prediction_op(self):
        """Adds the 1-hidden-layer NN:
            h = Relu(xW + b1)
            h_drop = Dropout(h, dropout_rate)
            pred = h_dropU + b2

        Note that we are not applying a softmax to pred. The softmax will instead be done in
        the add_loss_op function, which improves efficiency because we can use
        tf.nn.softmax_cross_entropy_with_logits

        Use the initializer from q2_initialization.py to initialize W and U (you can initialize b1
        and b2 with zeros)

        Hint: Here are the dimensions of the various variables you will need to create
                    W:  (n_features*embed_size, hidden_size)
                    b1: (hidden_size,)
                    U:  (hidden_size, n_classes)
                    b2: (n_classes)
        Hint: Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument. 
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, n_classes)
        """
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
        """Adds Ops for the loss function to the computational graph.
        In this case we are using cross entropy loss.
        The loss should be averaged over all examples in the current minibatch.

        Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            pred, self.labels_placeholder), name="avg_batch_loss")
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(loss)
        return train_op


    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss



    def fit_org(self, sess, saver, parser, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch_org(sess, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if saver:
                    print "New best dev UAS! Saving model in ./data/weights/parser.weights"
                    saver.save(sess, './data/weights/parser.weights')
            print

    def run_valid_epoch(self, sess, dataset):
        sentences = dataset.valid_data
        rem_sentences = [sentence for sentence in sentences]
        [sentence.clear_prediction_dependencies() for sentence in sentences]
        [sentence.clear_children_info() for sentence in sentences]

        while len(rem_sentences) != 0:
            curr_batch_size = min(dataset.model_config.batch_size, len(rem_sentences))
            batch_sentences = rem_sentences[:curr_batch_size]

            enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in batch_sentences]
            enable_count = np.count_nonzero(enable_features)

            while enable_count > 0:
                curr_sentences = [sentence for i, sentence in enumerate(batch_sentences) if enable_features[i] == 1]

                # get feature for each sentence
                # call predictions -> argmax
                # store dependency and left/right child
                # update state
                # repeat

                curr_inputs = [dataset.feature_extractor.extract_for_current_state(sentence, dataset.word2idx) for sentence in curr_sentences]
                predictions = sess.run(self.pred, feed_dict = self.create_feed_dict(curr_inputs))
                legal_labels = np.asarray([sentence.get_legal_labels() for sentence in curr_sentences], dtype=np.float32)
                legal_transitions = np.argmax(predictions + 1000 * legal_labels, axis=1)

                # update left/right children so can be used for next feature vector
                [sentence.update_child_dependencies(transition) for (sentence, transition) in zip(curr_sentences, legal_transitions) if transition != 2]

                # update state
                [sentence.update_state_by_transition(legal_transition, gold=False) for (sentence, legal_transition) in zip(curr_sentences, legal_transitions)]

                enable_features = [0 if len(sentence.stack) == 1 and len(sentence.buff) == 0 else 1 for sentence in
                                   batch_sentences]
                enable_count = np.count_nonzero(enable_features)

            # Reset stack and buffer
            [sentence.reset_to_initial_state() for sentence in batch_sentences]
            rem_sentences = rem_sentences[curr_batch_size:]

    def run_epoch_org(self, sess, parser, train_examples, dev_set):
        prog = Progbar(target=1 + len(train_examples) / self.config.batch_size)
        for i, (train_x, train_y) in enumerate(minibatches(train_examples, self.config.batch_size)):
            # np.savetxt("./data/their_model_input", train_x, fmt='%.0f')
            # np.savetxt("./data/their_model_output", train_y, fmt='%.0f')
            # print "input, outout: {}, {}".format(train_x.shape, train_y.shape)
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS

    def run_epoch(self, sess, config, dataset):
        prog = Progbar(target=1 + len(dataset.train_inputs) / config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatches([dataset.train_inputs, dataset.train_targets],
                                                               config.batch_size)):
            # print "input, outout: {}, {}".format(np.array(train_x).shape, np.array(train_y).shape)
            # np.savetxt("./data/my_model_input", train_x, fmt='%.0f')
            # np.savetxt("./data/my_model_output", train_y, fmt='%.0f')
            loss = self.train_on_batch(sess, train_x, train_y)
            prog.update(i + 1, [("train loss", loss)])

        print "Evaluating on dev set",
        self.run_valid_epoch(sess, dataset)

        correct_tokens = 0
        all_tokens = 0
        for sentence in dataset.valid_data:

            # reset each predicted head before evaluation
            [token.reset_predicted_head_id() for token in sentence.tokens]

            head = [-2] * len(sentence.tokens)
            # assert len(sentence.dependencies) == len(sentence.predicted_dependencies)
            for h, t, in sentence.predicted_dependencies:
                head[t.token_id] = h.token_id
            correct_tokens += sum([1 if token.head_id == head[i] else 0 for (i, token) in enumerate(sentence.tokens)])
            all_tokens += len(sentence.tokens)

            # for pred_h, gold_h, gold_l, pos in \
            #         zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
            #     assert self.id2tok[pos].startswith(P_PREFIX)
            #     pos_str = self.id2tok[pos][len(P_PREFIX):]
            #     if (self.with_punct) or (not punct(self.language, pos_str)):
            #         UAS += 1 if pred_h == gold_h else 0
            #         all_tokens += 1



            # true_pairs = [(head.token_id, dependent.token_id) for (head, dependent) in sentence.dependencies]
            # for (head, dependent) in sentence.predicted_dependencies:
            #     dependent.predicted_head_id = head.token_id
            # predicted_pairs = [(head.token_id, dependent.token_id) for (head, dependent) in sentence.predicted_dependencies]
            #
            # correct_tokens += len(set(true_pairs).intersection(predicted_pairs))
            # all_tokens += len(sentence.tokens)


            # if len(sentence.dependencies) != len(sentence.predicted_dependencies):
            #     print "A"
            #
            # correct_tokens += sum([1 if token.head_id == token.predicted_head_id else 0 for token in sentence.tokens])
            # all_tokens += len(sentence.tokens)


        valid_UAS = correct_tokens / float(all_tokens)
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
                    saver.save(sess, os.path.join(DataConfig.data_dir_path, DataConfig.model_dir, DataConfig.model_name))
            print


def highlight_string(temp):
    print 80 * "="
    print temp
    print 80 * "="


def main_org(debug=True):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug, load_existing_vocab = True)
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    print "vocab Size: {}".format(len(embeddings))
    print "Training Size: {}".format(len(train_examples))
    print "valid data Size: {}".format(len(dev_set))


    with tf.Graph().as_default():
        print "Building model...",
        start = time.time()
        model = ParserModel(config, embeddings)
        parser.model = model
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        # If you are using an old version of TensorFlow, you may have to use
        # this initializer instead.
        # init = tf.initialize_all_variables()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as session:
            parser.session = session
            session.run(init)

            print 80 * "="
            print "TRAINING"
            print 80 * "="
            session.run(tf.assign(model.embedding_matrix, model.pretrained_embeddings))
            # np.savetxt("./data/their_model_emb.txt", model.pretrained_embeddings, fmt='%.2f')
            model.fit_org(session, saver, parser, train_examples, dev_set)

            if not debug:
                print 80 * "="
                print "TESTING"
                print 80 * "="
                print "Restoring the best model weights found on the dev set"
                saver.restore(session, './data/weights/parser.weights')
                print "Final evaluation on test set",
                UAS, dependencies = parser.parse(test_set)
                print "- test UAS: {:.2f}".format(UAS * 100.0)
                print "Writing predictions"
                with open('q2_test.predicted.pkl', 'w') as f:
                    cPickle.dump(dependencies, f, -1)
                print "Done!"


def main(debug=True, load_existing_vocab = False):
    highlight_string("INITIALIZING")
    print "loading data.."

    dataset = load_datasets(load_existing_vocab)
    config = dataset.model_config

    print dataset.model_config.vocab_size
    print sorted(dataset.word2idx.keys())
    print len(dataset.train_inputs)

    print "vocab Size: {}".format(len(dataset.word2idx))
    print "Training Size: {}".format(len(dataset.train_inputs))
    print "valid data Size: {}".format(len(dataset.valid_data))

    if not os.path.exists(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir)):
        os.makedirs(os.path.join(DataConfig.data_dir_path, DataConfig.model_dir))

    with tf.Graph().as_default():
        print "Building network...",
        start = time.time()
        model = ParserModel(config, dataset.embedding_matrix)
        print "took {:.2f} seconds\n".format(time.time() - start)

        init = tf.global_variables_initializer()
        saver = None if debug else tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # np.savetxt("./data/my_model_emb.txt", model.pretrained_embeddings, fmt='%.2f')
            sess.run(tf.assign(model.embedding_matrix, model.pretrained_embeddings))
            highlight_string("TRAINING")

            model.fit(sess, saver, config, dataset)

            # if not debug:
            #     highlight_string("TRAINING")
            #     print "Restoring the best model weights found on the dev set"
            #     saver.restore(sess, './data/weights/parser.weights')
            #     print "Final evaluation on test set",
            #     UAS, dependencies = parser.parse(test_set)
            #     print "- test UAS: {:.2f}".format(UAS * 100.0)
            #     print "Writing predictions"
            #     with open('q2_test.predicted.pkl', 'w') as f:
            #         cPickle.dump(dependencies, f, -1)
            #     print "Done!"


if __name__ == '__main__':
    # main(debug=False, load_existing_vocab = True)
    main_org(debug=False)
