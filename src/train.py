from marketmimic.utils import load_data, join_date_time
from marketmimic.model import build_gan, generate_data
from marketmimic.data import prepare_data, inverse_scale_data
from marketmimic.training import train_gan
from marketmimic.constants import LATENT_DIM

if __name__ == '__main__':
    zip_file = '../data/AAPL-Tick-Standard.txt.zip'
    txt_file = 'AAPL-Tick-Standard.txt'

    # Load data
    df = load_data(zip_file, txt_file)
    df = join_date_time(df, 'Date', 'Time')

    # Prepare data
    data_scaled, scaler = prepare_data(df)

    # Build GAN
    generator, discriminator, gan = build_gan()

    # Train GAN
    train_gan(generator, discriminator, gan, data_scaled, epochs=1000, batch_size=64)

    new_data = generate_data(generator, LATENT_DIM, 100)
    print("Generated Data Samples:")
    print(new_data)

    # Inverse scale the generated data
    original_data = inverse_scale_data(new_data, scaler)
    print("Generated Data Samples (Original Scale):")
    print(original_data)