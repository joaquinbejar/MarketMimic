from tabulate import tabulate

from marketmimic.constants import LATENT_DIM
from marketmimic.data import prepare_data, inverse_scale_data, invert_sliding_windows
from marketmimic.file import save_models, load_models
from marketmimic.loss import *
from marketmimic.metric import *
from marketmimic.model import build_gan, generate_data
from marketmimic.training import train_gan
from marketmimic.utils import load_data, join_date_time, generate_market_data_from_func

if __name__ == '__main__':
    # zip_file = '../data/AAPL-Tick-Standard.txt.zip'
    # txt_file = 'AAPL-Tick-Standard.txt'
    #
    # # Load data
    # df = load_data(zip_file, txt_file)
    # df = join_date_time(df, 'Date', 'Time')

    df = generate_market_data_from_func(10_000_000)
    print(df.sample(15))

    # Prepare data
    data_scaled, scalers = prepare_data(df)

    loss_func = wasserstein_loss
    metrics_func = rmse

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
    train_gan(generator, discriminator, gan, data_scaled, epochs=200, batch_size=128)
    end = time.time()
    print(f"Time to train: {end - start:.2f}")

    # path = '../models/'
    # generator_filename, discriminator_filename, gan_filename = save_models(generator, discriminator, gan, path)
    # generator, discriminator, gan = load_models(generator_filename, discriminator_filename, gan_filename,
    #                                             loss_func=loss_func,
    #                                             metrics_func=metrics_func,
    #                                             path=path)

    new_data = generate_data(generator, 100, LATENT_DIM)

    # Inverse scale the generated data
    inverse_data = invert_sliding_windows(new_data)
    original_data = inverse_scale_data(inverse_data, scalers)

    print(tabulate(original_data, headers='keys', tablefmt='psql'))
