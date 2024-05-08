import os
import time

import joblib
from tabulate import tabulate

from marketmimic.constants import LATENT_DIM, BATCH_SIZE, EPOCHS
from marketmimic.data import prepare_data, inverse_scale_data, invert_sliding_windows
from marketmimic.metric import *
from marketmimic.model import build_gan, generate_data
from marketmimic.training import train_gan
from marketmimic.utils import generate_market_data_from_func, load_data, join_date_time

if __name__ == '__main__':
    zip_file = '../data/AAPL-Tick-Standard.txt.zip'
    txt_file = 'AAPL-Tick-Standard.txt'

    # Load data
    # df = load_data(zip_file, txt_file)
    # df = join_date_time(df, 'Date', 'Time')
    # df = df.iloc[:100]

    df = generate_market_data_from_func(1000000)

    print(f'Original data shape: {df.shape}')
    print(df.sample(15))

    # Prepare data
    data_scaled, scalers = prepare_data(df)

    # Build GAN
    generator, discriminator, gan, gen_optimizer, disc_optimizer = build_gan()
    generator.summary()
    discriminator.summary()
    gan.summary()

    # Train GAN
    # Calculate time to train
    start = time.time()
    train_gan(generator, discriminator, gen_optimizer, disc_optimizer,
              data_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE)
    end = time.time()
    print(f"Time to train: {end - start:.2f}")

    # Save models
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    path = '../models/v0.1'
    checkpoint_prefix = os.path.join(path, "ckpt")
    checkpoint.save(file_prefix=checkpoint_prefix)
    joblib.dump(scalers, path + '/scalers.pkl')

    new_data = generate_data(generator, 100, LATENT_DIM)

    # Inverse scale the generated data
    inverse_data = invert_sliding_windows(new_data)
    original_data = inverse_scale_data(inverse_data, scalers)

    print(tabulate(original_data, headers='keys', tablefmt='psql'))
