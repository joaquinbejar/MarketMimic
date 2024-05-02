from tabulate import tabulate

from marketmimic.constants import LATENT_DIM
from marketmimic.data import prepare_data, inverse_scale_data, invert_sliding_windows
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
    generator, discriminator, gan = build_gan()
    generator.summary()
    discriminator.summary()
    gan.summary()

    # Train GAN
    train_gan(generator, discriminator, gan, data_scaled, epochs=100, batch_size=64)

    new_data = generate_data(generator, 1000, LATENT_DIM)

    # Inverse scale the generated data
    inverse_data = invert_sliding_windows(new_data)
    original_data = inverse_scale_data(inverse_data, scalers)

    print(tabulate(original_data, headers='keys', tablefmt='psql'))

    # print(original_data)
