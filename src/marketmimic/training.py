import numpy as np
from marketmimic.model import build_generator, build_discriminator
def train_gan(gan, dataset, latent_dim, epochs=1000, batch_size=32):
    generator = build_generator()
    discriminator = build_discriminator()
    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminador
        # ---------------------
        # Select a random batch of instances
        idx = np.random.randint(0, dataset.shape[0], batch_size)
        real_data = dataset[idx]

        # Generate fake data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        # Crear las etiquetas
        real_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))

        # Train discriminator (real classified as 1 and fake as 0)
        d_loss_real = discriminator.train_on_batch(real_data, real_y)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generador
        # ---------------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Progreso
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")



