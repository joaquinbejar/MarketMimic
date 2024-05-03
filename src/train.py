from tabulate import tabulate

from marketmimic.constants import LATENT_DIM
from marketmimic.data import prepare_data, inverse_scale_data, invert_sliding_windows
from marketmimic.file import load_model_from_file, save_model_to_file, generate_filename_with_timestamp
from marketmimic.loss import *
from marketmimic.metric import *
from marketmimic.model import build_gan, generate_data
from marketmimic.training import train_gan
from marketmimic.utils import load_data, join_date_time

if __name__ == '__main__':
    zip_file = '../data/AAPL-Tick-Standard.txt.zip'
    txt_file = 'AAPL-Tick-Standard.txt'

    # Load data
    df = load_data(zip_file, txt_file)
    df = join_date_time(df, 'Date', 'Time')

    # Prepare data
    data_scaled, scalers = prepare_data(df)

    loss_func = wasserstein_loss
    metrics_func = dtw_distance

    print(f"Using Loss function: {loss_func.__name__}")
    print(f"Using Metrics function: {metrics_func.__name__}")

    # Build GAN
    generator, discriminator, gan = build_gan(loss_func=loss_func, metrics=metrics_func)
    generator.summary()
    discriminator.summary()
    gan.summary()

    # calculate time to train
    import time

    start = time.time()
    # Train GAN
    train_gan(generator, discriminator, gan, data_scaled, epochs=2000, batch_size=32)
    end = time.time()
    print(f"Time to train: {end - start:.2f}")

    path = '../models/'
    # Save to file
    generator_filename = generate_filename_with_timestamp('generator')
    save_model_to_file(generator, path + generator_filename)

    gan_filename = generate_filename_with_timestamp('gan')
    save_model_to_file(gan, path + gan_filename)

    # Load from file
    generator = load_model_from_file(path + generator_filename, loss_func)
    gan = load_model_from_file(path + gan_filename, loss_func, metrics_func)

    new_data = generate_data(generator, 100, LATENT_DIM)

    # Inverse scale the generated data
    inverse_data = invert_sliding_windows(new_data)
    original_data = inverse_scale_data(inverse_data, scalers)

    print(tabulate(original_data, headers='keys', tablefmt='psql'))

