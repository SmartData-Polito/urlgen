# define the discriminator model
import itertools
import json
import random
import string
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import keras
import Levenshtein
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
# import sklearn
import tensorflow as tf
from tensorflow.keras.layers import (LSTM, Convolution1D, Dense, Dropout,
                                     Embedding, Flatten, Input, LeakyReLU,
                                     MaxPooling1D, Reshape, concatenate)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm

# print("Keras version:", tf.keras.__version__)
# print("Tensorflow version:", tf.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(False)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print("is_built_with_cuda:", tf.test.is_built_with_cuda())
# print("gpu_device_name:", tf.test.gpu_device_name())

physical_devices = tf.config.experimental.list_physical_devices('GPU')


class Data(object):

    name = None
    dataset = None
    tokenizer = None
    char_dictionary = None
    reversed_dictionary = None
    alphabet = string.ascii_lowercase + string.digits + "/:._-()=;?&%*,"
    embedding = None
    embedding_size = None
    embedding_weights = None

    def __init__(self, input_file, alphabet=None, embedding_path=None, embedding_size=70, name='default', apply_embedding=True):
        self.name = name
        self.dataset = pd.read_csv(input_file, sep="\t")
        self.token()

        assert embedding_path is not None
        self.embedding_size = embedding_size
        self.load_embedding(embedding_path)
        self.build_embedding()

    def token(self, alphabet=None):
        if alphabet is not None:
            self.alphabet = alphabet

        self.tokenizer = Tokenizer(
            num_words=None, filters="", lower=True, char_level=True)
        self.tokenizer.fit_on_texts(self.alphabet)
        self.tokenizer.word_index[" "] = 0
        self.char_dictionary = self.tokenizer.word_index
        self.reversed_dictionary = {
            num: char for char, num in self.char_dictionary.items()}
        return

    def get_alphabet_size(self):
        return len(self.alphabet)

    def get_dataset(self, size_word=200, split_data=0.8):
        data = []
        for line in self.dataset.values.tolist():
            data.append(line[0].lower()[:size_word])

        X = np.array(data)
        X = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=size_word, padding='post', value=0)

        train_len = int(len(data)*split_data)
        X_train = X[:train_len]
        X_test = X[train_len:]

        return X_train, X_test

    def get_real_samples(self, data):
        # random indexes
        samples = np.random.randint(0, len(data), len(data))

        # get samples
        X = data[samples]

        # generate class labels
        y = np.ones((len(data), 1))
        return iter(X), iter(y)

    def load_embedding(self, filename):
        with open(filename, 'r') as fp:
            dict_vector = json.load(fp)

        vocabulary_size = len(self.char_dictionary.keys())
        self.embedding_matrix = np.zeros(
            (vocabulary_size, self.embedding_size))

        for char, idx in self.char_dictionary.items():
            embedding_vector = dict_vector.get(char)
            if embedding_vector is not None:
                self.embedding_matrix[idx] = np.array(embedding_vector)

        return self.embedding_matrix

    def build_embedding(self):
        alphabet_size = len(self.char_dictionary.keys())
        self.embedding = Sequential()
        self.embedding.add(Embedding(alphabet_size, self.embedding_size,
                           name='embedding', weights=[self.embedding_matrix]))
        self.embedding.compile('rmsprop', 'mse')

    def save(self):
        pass


class Discriminator(object):

    model = None

    def __init__(self, input_size=200, embedding_size=70, kernel_values=[2, 3, 4, 5, 6, 7],
                 n_filters=[64, 64], pool_sizes=[2, 2], dense_layers=[400, 200, 50],
                 leaning_rate=0.0005, initializer_stddev=0.02, apply_embedding=True, alphabet_size=200):

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.kernel_values = kernel_values
        self.n_filters = n_filters
        self.pool_sizes = pool_sizes
        self.dense_layers = dense_layers
        self.leaning_rate = leaning_rate
        self.initializer_stddev = initializer_stddev
        self.apply_embedding = apply_embedding
        self.alphabet_size = alphabet_size
        self.build()

    def fit(self, x, y, batch_size, epochs):
        self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def build(self):
        # weight initialization
        init = keras.initializers.RandomNormal(stddev=self.initializer_stddev)

        input_size_tuple = None
        if self.apply_embedding:
            input_size_tuple = (self.input_size, self.embedding_size)
        else:
            input_size_tuple = (self.input_size, self.alphabet_size)

        visible = Input((input_size_tuple[0], input_size_tuple[1]))

        flat = []
        for kernel_size in self.kernel_values:
            layer = Reshape(
                (input_size_tuple[0], input_size_tuple[1]))(visible)

            for n_filter, pool_size in zip(self.n_filters, self.pool_sizes):
                layer = Convolution1D(filters=n_filter, kernel_size=kernel_size,
                                      kernel_initializer=init,
                                      padding="same")(layer)

                layer = LeakyReLU()(layer)
                layer = MaxPooling1D(pool_size=pool_size)(layer)

            flat.append(Flatten()(layer))

        li = concatenate(flat)

        for dense in self.dense_layers:
            li = Dense(dense)(li)

        output = Dense(1, activation='sigmoid')(li)

        opt = keras.optimizers.RMSprop(lr=self.leaning_rate)
        self.model = Model(inputs=visible, outputs=output)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt, metrics=['accuracy'])

        return self.model

    @staticmethod
    def load(self, path):
        self.model = keras.models.load_model(path)

    def save(self):
        pass


class Generator(object):

    model = None

    def __init__(self, latent_dim=100, lstm_units=1000, output_shape=(200, 70), max_size_word=200, dropout=0.6):
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.output_shape = output_shape
        self.max_size_word = max_size_word
        self.dropout = dropout

        self.build()

    def build(self):
        visible = Input((self.latent_dim, 1))
        li = LSTM(self.lstm_units)(visible)
        li = Dropout(self.dropout)(li)
        li = Dense(
            self.output_shape[0]*self.output_shape[1], activation='linear')(li)
        output = Reshape(self.output_shape)(li)

        self.model = Model(inputs=visible, outputs=output)
        return self.model

    # generate points in latent space as input for the generator
    def generate_latent_points(self,  n):
        # generate points in the latent space
        # x_input = np.random.normal(0, 1, self.latent_dim * n)
        x_input = np.random.normal(0, 5, self.latent_dim * n)

        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n, self.latent_dim, 1)

        return x_input

    # use the generator to generate n fake examples and plot the results
    def generate_fake_samples(self, n):
        # generate points in latent space
        x_input = self.generate_latent_points(n)

        # predict outputs
        X = self.model.predict(x_input, verbose=0)

        # create class labels
        y = np.zeros((n, 1))

        return X, y

    def save(self):
        pass


class GAN(object):

    model = None
    discriminator = None
    generator = None

    def __init__(self, discriminator: Discriminator, generator: Generator, dataset: Data, leaning_rate=0.0005, max_size_word=200, latent_dim=100, apply_embedding=True):
        self.dataset = dataset
        self.discriminator = discriminator
        self.generator = generator
        self.leaning_rate = leaning_rate
        self.max_size_word = max_size_word
        self.latent_dim = latent_dim
        self.apply_embedding = apply_embedding
        self.history = {'d_loss_real': [], 'd_loss_fake': [], 'gan_loss': []}
        self.checkpoints = []

        self.best_gen = {'step': None, 'metric': None, 'model': None}
        self.best_disc = {'step': None, 'metric': None, 'model': None}

        self.build()

    def build(self):
        # make weights in the discriminator not trainable
        self.discriminator.model.trainable = False

        # connect them
        self.model = Sequential()

        # add generator
        self.model.add(self.generator.model)

        # add the discriminator
        self.model.add(self.discriminator.model)

        # compile model
        opt = keras.optimizers.RMSprop(lr=self.leaning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=opt)

        return self.model

    def get_train_state(self, x_input):
        # generate a encoded string from input
        X = self.generator.model.predict(x_input, verbose=0)

        return self.get_generate_string(X)

    def get_generate_string(self, x_data, real_data=False):
        generate_strings = []

        if real_data:
            for vector in x_data:
                generate_strings.append(
                    "".join([self.dataset.reversed_dictionary[n] for n in vector]))
        else:
            if self.apply_embedding:
                for embedded_vector in x_data:
                    one_hot = tf.linalg.matmul(embedded_vector, tf.linalg.pinv(
                        self.dataset.embedding.weights[0]))
                    encoded_url = np.argmax(one_hot, axis=1)
                    generate_strings.append(
                        "".join([self.dataset.reversed_dictionary[n] for n in encoded_url]))
            else:
                for encoded_url in x_data:
                    one_hot = np.argmax(encoded_url, axis=1)
                    generate_strings.append(
                        "".join([self.dataset.reversed_dictionary[n] for n in one_hot]))

        return generate_strings

    def save_best(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f'Best generator is at step {self.best_gen["step"]}')
        self.best_gen['model'].save(f'{path}/best_generator.h5')

        print(f'Best discriminator is at step {self.best_disc["step"]}')
        self.best_disc['model'].save(f'{path}/best_discriminator.h5')

    def compute_score_function(self, fake_data):

        distances_fake_fake, hist_fake_fake, cdf_fake_fake = self.compute_cdf(
            fake_data, fake_data)

        distance = tf.keras.losses.KLDivergence()(
            self.cdf_real_samples, cdf_fake_fake).numpy()

        distance_hist = tf.keras.losses.KLDivergence()(
            self.histogram_real_samples, hist_fake_fake).numpy()

        distance_shannon = scipy.spatial.distance.jensenshannon(
            self.histogram_real_samples, hist_fake_fake)

        self.checkpoints.append({
            'cdf_fake_fake': cdf_fake_fake,
            'hist_fake_fake': hist_fake_fake,
            'kl_div': distance,
            'kl_div_hist': distance_hist,
            'js_div_hist': distance_shannon
        })

    def compute_cdf(self, data1, data2):
        distances = []
        for str1 in data1:
            for str2 in data2:
                distances.append(Levenshtein.distance(str1, str2))

        counter = Counter(distances)
        histogram = []

        for i in range(self.max_size_word):
            histogram.append(counter.get(i, 1))

        histogram = np.array(histogram)/np.sum(histogram)
        cdf = np.cumsum(histogram)

        return distances, histogram, cdf

    def checkpoint(self, step, loss, interval, path, n_samples, save=False):
        x_seed = self.generator.generate_latent_points(n=n_samples)
        X = self.generator.model.predict(x_seed)
        generated_output = self.get_generate_string(X)

        Path(path + '/output').mkdir(parents=True, exist_ok=True)

        with open(f'{path}/output/generated-urls-{step}.txt', 'w') as f:
            for item in generated_output:
                f.write("%s\n" % item)

    # train the generator and discriminator
    def train(self, n_epochs=1000, n_batch=256, n_eval=200, n_discriminator=5, 
              n_generator=5, path="", checkpoint_interval=3, evaluation_size=1000, 
              save=False, split_data=0.8):
        
        X_train, X_test = self.dataset.get_dataset(split_data=split_data)

        real_data = self.get_generate_string(
            np.array(list(itertools.islice(X_test, evaluation_size))), real_data=True)
        
        # self.distances_real_samples, self.histogram_real_samples, self.cdf_real_samples = self.compute_cdf(real_data, real_data)

        # calculate the number of batches per epoch
        half_batch = int(n_batch / 2)
        n_steps = int(len(X_train) / (half_batch*n_discriminator))

        # this will be used to follow the evolution of the generate url
        x_seed = self.generator.generate_latent_points(n=1)

        try:
            # manually enumerate epochs
            self.checkpoint(step=0, loss=1, interval=checkpoint_interval,
                            path=path, n_samples=evaluation_size, save=save)
            
            for epoch in range(n_epochs):
                # prepare real samples
                x_real, y_real = self.dataset.get_real_samples(X_train)
                test = tf.random.Generator.from_seed(123)
                
                pbar = tqdm(range(1, n_steps+1), position=0, leave=True)
                for i in pbar:
                    for _ in range(n_discriminator):
                        # update discriminator
                        x_real_samples = np.array(
                            list(itertools.islice(x_real, half_batch)))
                        y_real_samples = np.array(
                            list(itertools.islice(y_real, half_batch)))

                        if self.apply_embedding:
#                             x_input = self.dataset.embedding.predict(
#                                 x_real_samples, verbose=0)
                            x_input = test.normal(shape=(5, 200, 30))

                        else:
                            x_input = tf.keras.utils.to_categorical(
                                y=x_real_samples, num_classes=self.discriminator.alphabet_size, dtype='float32')

                        # train on real samples
                        d_loss_real, _ = self.discriminator.model.train_on_batch(
                            x_input, y_real_samples)

                        # prepare fake examples
                        x_fake, y_fake = self.generator.generate_fake_samples(
                            half_batch)

                        # update discriminator
                        d_loss_fake, _ = self.discriminator.model.train_on_batch(
                            x_fake, y_fake)
                    
                    for _ in range(n_generator):
                        # prepare points in latent space as input for the generator
                        x_gan = self.generator.generate_latent_points(n_batch)

                        # create inverted labels for the fake samples
                        y_gan = np.ones((n_batch, 1))

                        # update the generator via the discriminator's error
                        gan_loss = self.model.train_on_batch(x_gan, y_gan)
                    
#                     pbar.set_description("Epoch {} Step {} l_real: {:.2f} l_fake: {:.2f} g_loss: {:.2f}".format(
#                         epoch + 1,  i, d_loss_real, d_loss_fake, gan_loss))
                    pbar.set_postfix(url=self.get_train_state(x_seed))

                    # self.checkpoint(step=epoch*n_steps + i, loss=(d_loss_real + d_loss_fake)/2, 
                    # interval=checkpoint_interval, path=path, n_samples=evaluation_size, save=save)

        except KeyboardInterrupt:
            # self.save_best(path)
            return self.history

            # self.save_best(path)
        return self.history

    def plot_kl_div(self, ax, name='kl_div'):
        kl_div = [step[name] for step in self.checkpoints]
        sns.lineplot(x=range(len(kl_div)), y=kl_div, ax=ax)

    def plot_history(self, ax):
        _ = pd.DataFrame.from_dict(self.history).plot(ax=ax)

    def plot_histogram(self, ax, cdf=False, epochs=None):
        warnings.filterwarnings("ignore")

        if type(ax) is not list:
            ax = [ax]
        else:
            ax = ax.flatten()

        checkpoints = None
        if epochs is not None:
            checkpoints = []
            for i in epochs:
                checkpoints.append(self.checkpoints[i])
        else:
            checkpoints = self.checkpoints

        for checkpoint, axis, step in zip(checkpoints, ax[0], list(range(len(checkpoints)))):
            if cdf:
                _ = sns.lineplot(x=list(range(self.max_size_word)), y=self.cdf_real_samples,
                                 ax=axis, label='Real').set_title(f'Step {step}')
                _ = sns.lineplot(x=list(range(self.max_size_word)),
                                 y=checkpoint['cdf_fake_fake'], ax=axis, label='Fake')

            else:
                _ = axis.bar(list(range(self.max_size_word)),
                             self.histogram_real_samples, label='Real')
                _ = axis.bar(list(range(self.max_size_word)),
                             checkpoint['hist_fake_fake'], label='Fake')
                axis.legend(scatterpoints=1, bbox_to_anchor=(1, 0.7),
                            loc=2, borderaxespad=1., ncol=1, fontsize=14)


class FCDiscriminator(Discriminator):

    def __init__(self, input_size=200, embedding_size=70, dense_layers=[400, 200, 50], activation='linear', dropout_value=0.5, leaning_rate=0.0005, apply_embedding=True, alphabet_size=200):

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.activation = activation
        self.dense_layers = dense_layers
        self.leaning_rate = leaning_rate
        self.dropout_value = dropout_value
        self.leaning_rate = leaning_rate
        self.apply_embedding = apply_embedding
        self.alphabet_size = alphabet_size

        self.build()

    def build(self):

        input_size_tuple = None
        if self.apply_embedding:
            input_size_tuple = (self.input_size, self.embedding_size)
        else:
            input_size_tuple = (self.input_size, self.alphabet_size)

        self.model = Sequential()
        self.model.add(Input((input_size_tuple[0], input_size_tuple[1])))
        self.model.add(Flatten())

        # Add arbitrary layers
        for size in self.dense_layers:
            size = int(size)
            self.model.add(Dense(size, activation=self.activation))
            self.model.add(Dropout(self.dropout_value))

        # Add the final layer, with a single output
        self.model.add(Dense(1, activation='sigmoid'))

        opt = keras.optimizers.RMSprop(lr=self.leaning_rate)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=opt, metrics=['accuracy'])

        return self.model


class FCGenerator(Generator):

    def __init__(self, latent_dim=100, dense_layers=[8, 4, 8], output_shape=(200, 70), max_size_word=200, activation='tanh', dropout_value=0.5, leaning_rate=0.0005):
        self.latent_dim = latent_dim
        self.dense_layers = dense_layers
        self.output_shape = output_shape
        self.max_size_word = max_size_word
        self.activation = activation
        self.dropout_value = dropout_value
        self.leaning_rate = leaning_rate

        self.build()

    def build(self):

        self.model = Sequential()
        self.model.add(Input((self.latent_dim, 1)))
        self.model.add(Flatten())

        # Add arbitrary layers
        for size in self.dense_layers:
            self.model.add(Dense(size, activation=self.activation))
            self.model.add(Dropout(self.dropout_value))

        # Add the final layer
        self.model.add(Dense(np.prod(self.output_shape),
                       activation=self.activation))
        self.model.add(Dropout(self.dropout_value))
        self.model.add(Reshape(self.output_shape))

        return self.model
