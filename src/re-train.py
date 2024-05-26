import time

from marketmimic.constants import BATCH_SIZE, EPOCHS
from marketmimic.data import prepare_data
from marketmimic.metric import *
from marketmimic.model import build_gan
from marketmimic.training import train_gan
from marketmimic.utils import join_date_time, process_zip_files

if __name__ == '__main__':
    for e in range(1, 10):
        epoch: int = int(e * EPOCHS)
        for i in range(BATCH_SIZE, 0, -1):
            current_batch_size = int(BATCH_SIZE // (2 ** (BATCH_SIZE - i)))
            print(f"Current batch size: {current_batch_size}")
            for df in process_zip_files('../data'):
                df = join_date_time(df, 'Date', 'Time')
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
                          epochs=epoch,
                          batch_size=current_batch_size,
                          reset_weights=False)
                end = time.time()
                print(f"Time to train: {end - start:.2f}")

# if __name__ == '__main__':
#     # zip_file = '../data/AAPL-Tick-Standard.txt.zip'
#     # txt_file = 'AAPL-Tick-Standard.txt'
#     zip_file = '../data/AMD-Tick-Standard.txt.zip'
#     txt_file = 'AMD-Tick-Standard.txt'
#
#     # Load data
#     df = load_data(zip_file, txt_file)
#     df = join_date_time(df, 'Date', 'Time')
#     # df = generate_market_data_from_func(100)
#
#     generator, discriminator, gan, gen_optimizer, disc_optimizer = build_gan()
#     checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
#                                      discriminator_optimizer=disc_optimizer,
#                                      generator=generator,
#                                      discriminator=discriminator)
#
#     path = '../models/v0.1'
#     manager = tf.train.CheckpointManager(checkpoint,
#                                          directory=path,
#                                          max_to_keep=5)
#     checkpoint.restore(manager.latest_checkpoint)
#     if manager.latest_checkpoint:
#         print("Restaurado de {}".format(manager.latest_checkpoint))
#     else:
#         print("Inicializando desde cero.")
#
#     # Prepare data
#     data_scaled, _ = prepare_data(df)
#     start = time.time()
#     # Train GAN
#     train_gan(generator,
#               discriminator,
#               gen_optimizer,
#               disc_optimizer,
#               data_scaled,
#               epochs=EPOCHS,
#               batch_size=BATCH_SIZE,
#               reset_weights=False)
#     end = time.time()
#     print(f"Time to train: {end - start:.2f}")
