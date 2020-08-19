from nn_batches import train_neural_network, NeuralNetwork, squared_error
import matplotlib.pyplot as plt
from utils.activation_functions import radial_basis_function, diff_radial_basis_function, sigmoid, diff_sigmoid
from utils.optimizers import back_propagation, levenberg_marquardt
import numpy as np


def initialize_neural_network():
    return NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
                         number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
                         max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
                         activation_function=sigmoid, optimizer=OPTIMIZER,
                         loss_function=squared_error, input_weights=INPUT_WEIGHTS_INIT,
                         output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
                         damping=LM_DAMPING, diff_activation_function=diff_sigmoid, learning_schedule=LEARNING_SCHEDULE_BOOL)


# load data
train_data = np.genfromtxt('data/train_short.csv', delimiter=',')
test_data = np.genfromtxt('data/test_short.csv', delimiter=',')
# train_data = np.genfromtxt('data/train.csv', delimiter=',')
# test_data = np.genfromtxt('data/test.csv', delimiter=',')
time_data = np.genfromtxt('data/time_sequence.csv', delimiter=',')

"""LEARNING RATE SENSITIVITY"""
# np.random.seed(1)
# MAX_EPOCHS = 1000
# GOAL = 0.0001
# MIN_GRADIENT = 0.000001
# LEARNING_RATE = 0.1
# LM_DAMPING = 1
# NUMBER_OF_HIDDEN_NEURONS = 5
# BATCH_SIZE = 128
# # BATCH_SIZE = 8000
# OPTIMIZER = back_propagation
# LEARNING_SCHEDULE_BOOL = False
#
# NUMBER_OF_INPUTS = 2
# NUMBER_OF_OUTPUTS = 1

# plt.figure(dpi=300)
# for LEARNING_RATE in [1, 0.1, 0.01, 0.001]:
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     # plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:blue')
# plt.xlabel('Epoch')
# plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()

"""BATCH SIZE SENSITIVITY"""
# LEARNING_SCHEDULE_BOOL = False
# plt.figure(dpi=300)
# for BATCH_SIZE in [32, 128, 512, 2048, 8000]:
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     # plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, batch size {}'.format(str(BATCH_SIZE)))#, c='tab:blue')
# plt.xlabel('Epoch')
# plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()

"""WEIGHT SEED INIT SENSITIVITY"""
# np.random.seed(1)
# MAX_EPOCHS = 1000
# GOAL = 0.001
# MIN_GRADIENT = 0.000001
# LEARNING_RATE = 0.01
# LM_DAMPING = 1
# NUMBER_OF_HIDDEN_NEURONS = 5
# BATCH_SIZE = 128
# OPTIMIZER = back_propagation
#
# NUMBER_OF_INPUTS = 2
# NUMBER_OF_OUTPUTS = 1

# LEARNING_SCHEDULE_BOOL = False

# # for i in range(1, 3):
# test_log = []
# for i in [1, 2, 3]:
#     plt.figure(dpi=300)
#     np.random.seed(i)
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print('minimum error is ', min_error)
#     plt.plot(range(final_epoch), test_error_log, label='Test error, seed {}'.format(i))#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, seed {}'.format(i))#, c='tab:blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#     plt.ylim(0, 1)
#     plt.grid()
#     plt.legend()
#     plt.show()
#     test_log.append(train_error_log)
#     test_log.append(test_error_log)
# np.savetxt('latest_results.csv', test_log)
# # data initialization (maybe just mention that it has a significant effect)

"""VARIABLE LEARNING RATE"""
# np.random.seed(1)
# MAX_EPOCHS = 2000
# GOAL = 0.001
# MIN_GRADIENT = 0 #this is lower now because we want to get better results
# LEARNING_RATE = 0.1
# LM_DAMPING = 0.1
# NUMBER_OF_HIDDEN_NEURONS = 5
# BATCH_SIZE = 128
# OPTIMIZER = back_propagation
# LEARNING_SCHEDULE_BOOL = True
#
# plt.figure(dpi=300)
# INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
# OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
# # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
# CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
# ff_network = initialize_neural_network()
# train_error_log, test_error_log, final_epoch, min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
# print(min_error)
# plt.plot(range(final_epoch), test_error_log, label='Test error')#, c='tab:orange')
# plt.plot(range(final_epoch), train_error_log, label='Train error')#, c='tab:blue')
# plt.xlabel('Epoch')
# plt.ylabel('RMS error [-]')
# plt.ylim(0, 0.1)
# plt.grid()
# plt.legend()
# plt.show()

"""OPTIMIZE NUMBER OF HIDDEN NEURONS FOR ALL OTHER PARAMETERS FIXED"""
np.random.seed(1)
MAX_EPOCHS = 1000
GOAL = 0.0001
MIN_GRADIENT = 0.000001
LEARNING_RATE = 0.1
LM_DAMPING = 1
BATCH_SIZE = 128
OPTIMIZER = back_propagation

NUMBER_OF_INPUTS = 2
NUMBER_OF_OUTPUTS = 1

LEARNING_SCHEDULE_BOOL = True

plt.figure(dpi=300)


def optimizer(NUMBER_OF_HIDDEN_NEURONS):
    rbf_network = initialize_neural_network()
    train_error_log, test_error_log, final_epoch, min_error = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
    print(min_error)
    plt.plot(range(final_epoch), train_error_log, label='Train error, {} hidden neurons'.format(str(NUMBER_OF_HIDDEN_NEURONS)))
    # plt.plot(range(final_epoch), test_error_log, label='Test error')
    plt.xlabel('Epoch')
    plt.ylabel('RMS error [-]')
    plt.ylim(0, 0.1)
    plt.grid()
    plt.legend()
    return min_error


error_log = np.zeros(50)
NUMBER_OF_HIDDEN_NEURONS = 15
previous_error = 100
last_increase = False
i = 0
change_rate = 5

while i < 20:
    if i > 5:
        change_rate = 2
    if i > 10:
        change_rate = 1
    if error_log[NUMBER_OF_HIDDEN_NEURONS-1] > 0:
        current_error = error_log[NUMBER_OF_HIDDEN_NEURONS-1]
    else:
        INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
        OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
        CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
        current_error = optimizer(NUMBER_OF_HIDDEN_NEURONS)
    error_log[NUMBER_OF_HIDDEN_NEURONS - 1] = current_error
    if last_increase:
        if current_error < previous_error:
            NUMBER_OF_HIDDEN_NEURONS += change_rate
            last_increase = True
        else:
            NUMBER_OF_HIDDEN_NEURONS -= change_rate # no need to do times two, because, if we had a worse score it will go back to the already calculated one, which it will not recalculate so we will jump it either way
            last_increase = False
    else:
        if current_error < previous_error:
            NUMBER_OF_HIDDEN_NEURONS -= change_rate
            last_increase = False
        else:
            NUMBER_OF_HIDDEN_NEURONS += change_rate
            last_increase = True
    previous_error = current_error
    if NUMBER_OF_HIDDEN_NEURONS < 2:
        NUMBER_OF_HIDDEN_NEURONS = 2
    i+=1

print(error_log)
plt.show()

"""BACKPROP vs LM"""
# np.random.seed(1)
# MAX_EPOCHS = 2000
# GOAL = 0.001
# MIN_GRADIENT = 0.000001
# LEARNING_RATE = 0.0001
# LM_DAMPING = 0.1
# NUMBER_OF_HIDDEN_NEURONS = 15
# BATCH_SIZE = 4
# OPTIMIZER = back_propagation
#
# plt.figure(dpi=300)
# for i in [levenberg_marquardt, back_propagation]:
#     OPTIMIZER = i
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#
#     output = []
#     for j in range(len(time_data[0])):
#         output.append(ff_network.evaluate([time_data[0,j], time_data[1,j]]))
#
#     optimizer_names = ['backpropagation', 'Levenberg Marquardt']
#
#     # plt.subplot(1, 2, 1)
#     plt.plot(range(final_epoch), test_error_log, label='Test error, ' + OPTIMIZER.__name__)#, c='tab:orange')
#     plt.plot(range(final_epoch), train_error_log, label='Train error, ' + OPTIMIZER.__name__)#, c='tab:blue')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#
#     # plt.subplot(1, 2, 2)
#     # plt.plot(range(len(time_data[0])), np.array(output)[:, 0], label='Time sequence, ' + OPTIMIZER.__name__)
# # plt.subplot(1, 2, 2)
# # plt.plot(range(len(time_data[0])), time_data[2,:], label='Time sequence, ' + OPTIMIZER.__name__)
# plt.grid()
# plt.legend()
# plt.show()
# # todo; plot results in time; approximation vs real