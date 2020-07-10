from nn_batches import train_neural_network, NeuralNetwork
import matplotlib.pyplot as plt
from utils.activation_functions import radial_basis_function, diff_radial_basis_function, sigmoid, diff_sigmoid
from utils.optimizers import back_propagation, levenberg_marquardt


def initialize_neural_network():
    return NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
                           number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
                           max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
                           activation_function=sigmoid, optimizer=OPTIMIZER,
                           loss_function=least_squares, input_weights=INPUT_WEIGHTS_INIT,
                           output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
                           damping=LM_DAMPING, diff_activation_function=diff_sigmoid)


# load data
# train_data = np.genfromtxt('train.csv', delimiter=',')
# test_data = np.genfromtxt('test.csv', delimiter=',')
train_data = np.genfromtxt('data/train_short.csv', delimiter=',')
test_data = np.genfromtxt('data/test_short.csv', delimiter=',')
time_data = np.genfromtxt('data/time_sequence.csv', delimiter=',')

"""SENSITIVITY ANALYSIS"""
# learning rate
np.random.seed(1)
MAX_EPOCHS = 200
GOAL = 0.001
MIN_GRADIENT = 0.00001
LEARNING_RATE = 0.001
LM_DAMPING = 1
NUMBER_OF_HIDDEN_NEURONS = 15
BATCH_SIZE = 128
OPTIMIZER = back_propagation

NUMBER_OF_INPUTS = 2
NUMBER_OF_OUTPUTS = 1

# plt.figure()
# for i in [0.1, 0.01, 0.001, 0.0001]:
#     LEARNING_RATE = i
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#
#     plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(i))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, learning rate {}'.format(i))#, c='tab:blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()



# # random weight initialization
# np.random.seed(1)
# MAX_EPOCHS = 100
# GOAL = 0.001
# MIN_GRADIENT = 0.000001
# LEARNING_RATE = 0.001
# LM_DAMPING = 1
# NUMBER_OF_HIDDEN_NEURONS = 15
# BATCH_SIZE = 128
# OPTIMIZER = back_propagation
#
# NUMBER_OF_INPUTS = 2
# NUMBER_OF_OUTPUTS = 1
#
# plt.figure(dpi=300)
# for i in range(1, 3):
#     np.random.seed(i)
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#
#     plt.plot(range(final_epoch), test_error_log, label='Test error, seed {}'.format(i))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, seed {}'.format(i))#, c='tab:blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()

# data initialization (maybe just mention that it has a significant effect)

# # number of neurons
# np.random.seed(1)
# MAX_EPOCHS = 100
# GOAL = 0.0001
# MIN_GRADIENT = 0.0000001
# LEARNING_RATE = 0.01
# # LEARNING_RATE = 0.1
# LM_DAMPING = 1
# NUMBER_OF_HIDDEN_NEURONS = 15
# BATCH_SIZE = 256
# OPTIMIZER = back_propagation
#
# plt.figure(dpi=300)
# for i in [3, 5, 10, 25, 50]:  # with the random init we start with a higher error because of the higher number of neurons
#     NUMBER_OF_HIDDEN_NEURONS = i
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#
#     plt.plot(range(final_epoch), test_error_log, label='Test error, {} hidden nodes'.format(i))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, {} hidden nodes'.format(i))#, c='tab:blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()
#
# backprop vs LM
np.random.seed(1)
MAX_EPOCHS = 50
GOAL = 0.001
MIN_GRADIENT = 0.000001
LEARNING_RATE = 0.0001
LM_DAMPING = 0.1
NUMBER_OF_HIDDEN_NEURONS = 50
BATCH_SIZE = 128
OPTIMIZER = back_propagation

plt.figure(dpi=300)
for i in [levenberg_marquardt, back_propagation]:
    OPTIMIZER = i

    # weights initialization
    INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
    OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
    # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
    CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
    ff_network = initialize_neural_network()
    train_error_log, test_error_log, final_epoch = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)

    output = []
    for j in range(len(time_data[0])):
        output.append(ff_network.evaluate([time_data[0,j], time_data[1,j]]))

    optimizer_names = ['backpropagation', 'Levenberg Marquardt']

    # plt.subplot(1, 2, 1)
    plt.plot(range(final_epoch), test_error_log, label='Test error, ' + OPTIMIZER.__name__)#, c='tab:orange')
    plt.plot(range(final_epoch), train_error_log, label='Train error, ' + OPTIMIZER.__name__)#, c='tab:blue')
    plt.xlabel('Epoch')
    plt.ylabel('RMS error [-]')

    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(time_data[0])), np.array(output)[:, 0], label='Time sequence, ' + OPTIMIZER.__name__)
# plt.subplot(1, 2, 2)
# plt.plot(range(len(time_data[0])), time_data[2,:], label='Time sequence, ' + OPTIMIZER.__name__)
plt.grid()
plt.legend()
plt.show()
# todo; plot results in time; approximation vs real
