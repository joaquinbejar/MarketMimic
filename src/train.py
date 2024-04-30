from marketmimic.utils import load_data, join_date_time
from marketmimic.model import build_gan
from marketmimic.data import prepare_data
from marketmimic.training import train_gan

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
    train_gan(generator, discriminator, gan, data_scaled, epochs=1000, batch_size=32)

