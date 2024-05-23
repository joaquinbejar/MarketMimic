import time

from marketmimic.data import prepare_data
from marketmimic.metric import *
from marketmimic.model import build_gan
from marketmimic.training import train_gan
from marketmimic.utils import load_data, join_date_time

if __name__ == '__main__':
    # zip_file = '../data/AAPL-Tick-Standard.txt.zip'
    # txt_file = 'AAPL-Tick-Standard.txt'
    zip_file = '../data/AMZN-Tick-Standard.txt.zip'
    txt_file = 'AMZN-Tick-Standard.txt'

    # Load data
    df = load_data(zip_file, txt_file)
    df = join_date_time(df, 'Date', 'Time')
    # df = generate_market_data_from_func(100)

    generator, discriminator, gan, gen_optimizer, disc_optimizer = build_gan()
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    path = '../models/v0.1'
    manager = tf.train.CheckpointManager(checkpoint,
                                         directory=path,
                                         max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restaurado de {}".format(manager.latest_checkpoint))
    else:
        print("Inicializando desde cero.")

    # Prepare data
    data_scaled, _ = prepare_data(df)
    start = time.time()
    # Train GAN
    train_gan(generator,
              discriminator,
              gen_optimizer,
              disc_optimizer,
              data_scaled,
              epochs=3,
              batch_size=2,
              reset_weights=False)
    end = time.time()
    print(f"Time to train: {end - start:.2f}")
