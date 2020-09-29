import tensorflow as tf
import numpy as np

ff_model = tf.keras.models.load_model('models/ff_model')
print(ff_model.predict(np.array([[-.0520739, -0.0474268]])))
