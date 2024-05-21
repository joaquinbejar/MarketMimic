import unittest

import tensorflow as tf


class TestGPU(unittest.TestCase):
    def test_if_available(self):
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        print("Available devices:", tf.config.list_physical_devices())
        tf.debugging.set_log_device_placement(True)

        # Crear algunos tensores y realizar una operación simple en la GPU
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c = tf.matmul(a, tf.transpose(b))

        print(c)


if __name__ == '__main__':
    unittest.main()
