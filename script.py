import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
import sklearn
import seaborn as sns
import GANET as ganet
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


max_size_url = 200
embedding_size = 30
latent_dim = 10
lstm_units = 100

batch_size = 10
n_batch = 10
n_epochs = 10
n_discriminator = 1
n_generator = 1

embedding_path = f'char_embedding_{embedding_size}.json'
dataset_name = sys.argv[1]
number_of_samples = int(sys.argv[2])

base_path = 'generated_samples'
path = sys.argv[1]
_ = Path('./generated_samples').mkdir(parents=True, exist_ok=True)

print('Loading data...')
data = ganet.Data(input_file=dataset_name, embedding_size=embedding_size,
                  embedding_path=embedding_path, name=dataset_name, apply_embedding=True)

print('Building model...')
# CNN Discriminator and LSTM Generator
generator = ganet.Generator(latent_dim=latent_dim, lstm_units=lstm_units,
                            output_shape=(max_size_url, embedding_size))
discriminator = ganet.Discriminator(
    embedding_size=embedding_size, apply_embedding=True, alphabet_size=data.get_alphabet_size()+1)
gan_model = ganet.GAN(discriminator, generator, data,
                      leaning_rate=0.0005, apply_embedding=True)

print('Starting training...')
history = gan_model.train(n_batch=n_batch, n_epochs=n_epochs, n_discriminator=n_discriminator, n_generator=n_generator, path=f"./{base_path}/", evaluation_size=1000, checkpoint_interval=10, save=False)

print('Generate samples:')
x_seed = gan_model.generator.generate_latent_points(n=number_of_samples)
urls = gan_model.get_train_state(x_seed)

for url in urls:
    print(url)
