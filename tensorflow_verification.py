import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_graphics.math.optimizer as tfg


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


BATCH_SIZE = 8000
NUMBER_OF_HIDDEN_NEURONS = 20
MAX_EPOCHS = 250

all_data = np.genfromtxt('all_data.csv', delimiter=',')
alpha = all_data[0]
# alpha = 2*(alpha-min(alpha))/(max(alpha)-min(alpha))-1
beta = all_data[1]
# beta = 2*(beta-min(beta))/(max(beta)-min(beta))-1
Cm_old = all_data[2]
Cm = Cm_old
# Cm = 2*(Cm_old-min(Cm_old))/(max(Cm_old)-min(Cm_old))-1
all_data = np.array([alpha, beta, Cm]).T#.reshape(10000, 3).astype("float32")
validation_split = 0.2
shuffle_dataset = True
random_seed = 1

dataset_size = len(all_data)
indices = list(range(dataset_size))
test_len = int(np.floor(validation_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[test_len:], indices[:test_len]

train_data = all_data[train_indices,:]
test_data = all_data[test_indices, :]
x_train = train_data[:,0:2]
y_train = train_data[:,2]
x_test = test_data[:,0:2]
y_test = test_data[:,2]

ff_model = tf.keras.models.Sequential()
ff_model.add(tf.keras.Input(shape=(2,), batch_size=BATCH_SIZE))
ff_model.add(layers.Dense(units=NUMBER_OF_HIDDEN_NEURONS, activation=tf.keras.activations.sigmoid, use_bias=False))#, kernel_initializer=tf.keras.initializers.RandomNormal))
ff_model.add(layers.Dense(units=1, activation=None, use_bias=False))

ff_model.compile(
    # optimizer=tf.optimizers.Adam(learning_rate=0.0005),
    tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False, name="SGD"),  # Optimizer
    # optimizer=tfg.levenberg_marquardt,
    # optimizer = tf.keras.optimizers.RMSprop(),

    metrics=['accuracy'],

    # loss = tf.losses.mean_squared_error)
    loss= root_mean_squared_error)
    # loss = tf.keras.losses.Huber())

ff_model.summary()

history = ff_model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_test, y_test),
)

plt.figure(dpi=300)
plt.plot()
plt.plot(range(MAX_EPOCHS), history.history['loss'], label='Test error, ' + r'$\eta = 1$')
plt.grid('minor')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('RMSE [-]')
plt.yscale('log')
plt.ylim(0.001, 10)
plt.xlim(0, 250)
plt.show()

print("Evaluate on ALL data")
results = ff_model.evaluate(all_data[:,0:2], all_data[:,2], batch_size=BATCH_SIZE)
print("test loss, test acc:", results)
# results = (ff_model.predict(all_data[:,0:2], batch_size=BATCH_SIZE)+1)/2*(max(Cm_old)-min(Cm_old))+min(Cm_old)
results = ff_model.predict(all_data[:,0:2], batch_size=BATCH_SIZE)
rbf_best = np.array([all_data[:,0], all_data[:,1], list(results)])
np.savetxt('rbf_best.csv', rbf_best, delimiter=',')
np.savetxt('inputs_rbf_best.csv', all_data[:,0:2], delimiter=',')

keras.models.save_model(ff_model, 'models/ff_model')

"""###---###---###---"""
# class RBFLayer(Layer):
#     def __init__(self, units, gamma, **kwargs):
#         super(RBFLayer, self).__init__(**kwargs)
#         self.units = units
#         self.gamma = K.cast_to_floatx(gamma)
#
#     def build(self, input_shape):
#         self.mu = self.add_weight(name='mu',
#                                   shape=(int(input_shape[1]), self.units),
#                                   initializer = 'random_normal',
#                                   trainable=True)
#         #initializer='uniform',
#         super(RBFLayer, self).build(input_shape)
#
#     def call(self, inputs):
#         diff = K.expand_dims(inputs) - self.mu
#         l2 = K.sum(K.pow(diff,2), axis=1)
#         res = K.exp(-1 * self.gamma * l2)
#         return res
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.units)
#
#
# BATCH_SIZE = 16
# NUMBER_OF_HIDDEN_NEURONS = 10
#
# rbf_model = tf.keras.models.Sequential()
# rbf_model.add(tf.keras.Input(shape=(1,), batch_size=BATCH_SIZE))
# rbf_model.add(RBFLayer(NUMBER_OF_HIDDEN_NEURONS, 1))
# rbf_model.output_shape
# x = [1, 2, 1, 3, 1, 5, 2, 1, 8, 5, 4, 7, 3, 1, 2, 3]
# y = rbf_model(x)
# print(y)