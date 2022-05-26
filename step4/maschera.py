import numpy as np
import tensorflow as tf


def getMask(total_rows, total_cols, radius):
    X, Y = np.ogrid[:total_rows, :total_cols]

    center_row, center_col = total_rows / 2, total_cols / 2
    dist_from_center = (X - center_row) ** 2 + (Y - center_col) ** 2

    circular_mask = (dist_from_center <= radius ** 2)
    maschera = tf.cast(circular_mask, dtype=tf.complex64)

    return maschera
