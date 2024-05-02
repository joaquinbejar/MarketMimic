from tabulate import tabulate

from marketmimic.constants import LATENT_DIM
from marketmimic.data import prepare_data, inverse_scale_data, invert_sliding_windows
from marketmimic.file import load_model_from_file, save_model_to_file, generate_filename_with_timestamp
from marketmimic.loss import least_squares_loss
from marketmimic.metric import dtw_distance
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

    # Build GAN
    generator, discriminator, gan = build_gan(loss_func=least_squares_loss, metrics=dtw_distance)
    generator.summary()
    discriminator.summary()
    gan.summary()

    # calculate time to train
    import time

    start = time.time()
    # Train GAN
    train_gan(generator, discriminator, gan, data_scaled, epochs=2000, batch_size=128)
    end = time.time()
    print(f"Time to train: {end - start:.2f}")

    path = '../models/'
    # Save to file
    generator_filename = generate_filename_with_timestamp('generator')
    save_model_to_file(generator, path + generator_filename)

    gan_filename = generate_filename_with_timestamp('gan')
    save_model_to_file(gan, path + gan_filename)

    # Load from file
    generator = load_model_from_file(path + generator_filename, least_squares_loss)
    gan = load_model_from_file(path + gan_filename, least_squares_loss, dtw_distance)

    new_data = generate_data(generator, 100, LATENT_DIM)

    # Inverse scale the generated data
    inverse_data = invert_sliding_windows(new_data)
    original_data = inverse_scale_data(inverse_data, scalers)

    print(tabulate(original_data, headers='keys', tablefmt='psql'))

