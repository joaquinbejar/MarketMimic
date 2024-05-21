from math import sqrt

import numpy as np
import tensorflow as tf
from fastdtw import fastdtw
from scipy.linalg import sqrtm
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.metrics import Precision, Recall


def pearson_correlation(real_data, generated_data):
    if real_data.shape != generated_data.shape:
        raise ValueError("Shapes of real_data and generated_data must match.")
    return np.corrcoef(real_data, generated_data)[0, 1]


def dtw_distance(series1, series2):
    distance, path = fastdtw(series1, series2, dist=euclidean)
    return distance


def mae(real_data, generated_data):
    return mean_absolute_error(real_data, generated_data)


def mse(real_data, generated_data):
    return mean_squared_error(real_data, generated_data)


def rmse(real_data, generated_data):
    return sqrt(mse(real_data, generated_data))


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    @staticmethod
    def __name__():
        return 'f1_score'


def inception_score(images, batch_size=32, resize=False, splits=1):
    assert images.shape[1:] == (299, 299, 3), "The images should be of shape (299, 299, 3)."
    model = InceptionV3(include_top=True, pooling='avg', input_shape=(299, 299, 3))

    def get_preds(x):
        x = preprocess_input(x)
        _preds = model.predict(x, batch_size=batch_size)
        return _preds

    # If the images are not (299, 299, 3), they need to be resized.
    if resize:
        images = tf.image.resize(images, (299, 299)).numpy()

    preds = get_preds(images)
    split_scores = []

    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        split_scores.append(np.exp(kl))

    return np.mean(split_scores), np.std(split_scores)


def calculate_fid(act1, act2):
    """ Calculate the FID score between two activations """
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def frechet_inception_distance(real_images, generated_images):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    act1 = model.predict(preprocess_input(real_images))
    act2 = model.predict(preprocess_input(generated_images))

    fid_value = calculate_fid(act1, act2)
    return fid_value
