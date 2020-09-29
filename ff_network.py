from nn_batches import train_neural_network, NeuralNetwork, squared_error
import matplotlib.pyplot as plt
from utils.activation_functions import radial_basis_function, diff_radial_basis_function, sigmoid, diff_sigmoid
from utils.optimizers import back_propagation, levenberg_marquardt
import numpy as np


def reset_variables():
    global SEED, MAX_EPOCHS, GOAL, MIN_GRADIENT, LEARNING_RATE, LM_DAMPING, NUMBER_OF_HIDDEN_NEURONS, BATCH_SIZE,\
        NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, OPTIMIZER, ACTIVATION_FUNCTION, DIFF_ACTIVATION_FUNCTION
    SEED = 1
    MAX_EPOCHS = 250
    GOAL = 0.0001
    MIN_GRADIENT = 0.000001
    LEARNING_RATE = 1
    LM_DAMPING = 1
    NUMBER_OF_HIDDEN_NEURONS = 10
    # BATCH_SIZE = 128
    # BATCH_SIZE = 330
    BATCH_SIZE = 8000
    # BATCH_SIZE = 32
    NUMBER_OF_INPUTS = 2
    NUMBER_OF_OUTPUTS = 1

    # unset other ones
    OPTIMIZER = 42
    ACTIVATION_FUNCTION = 42
    DIFF_ACTIVATION_FUNCTION = 42


def initialize_neural_network(uniform, scaling):
    if uniform:
        INPUT_WEIGHTS_INIT = scaling * np.ones((NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS))
        OUTPUT_WEIGHTS_INIT = scaling * np.ones((NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS))
        CENTERS_INIT = scaling * np.ones((NUMBER_OF_HIDDEN_NEURONS))
    else:
        INPUT_WEIGHTS_INIT = scaling * np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
        OUTPUT_WEIGHTS_INIT = scaling * np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
        CENTERS_INIT = scaling * np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
    return NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
                         number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
                         max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
                         activation_function=ACTIVATION_FUNCTION, optimizer=OPTIMIZER,
                         loss_function=squared_error, input_weights=INPUT_WEIGHTS_INIT,
                         output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
                         damping=LM_DAMPING, diff_activation_function=DIFF_ACTIVATION_FUNCTION,
                         learning_schedule=LEARNING_SCHEDULE_BOOL)


# load data
# train_data = np.genfromtxt('data/train_short.csv', delimiter=',') # for testing new features
# test_data = np.genfromtxt('data/test_short.csv', delimiter=',') # "
train_data = np.genfromtxt('data/train.csv', delimiter=',')
test_data = np.genfromtxt('data/test.csv', delimiter=',')


# # 1. Varying learning rate
# print('---\n VARYING LEARNING RATE\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = False
#
# NUMBER_OF_INPUTS = 2
# NUMBER_OF_OUTPUTS = 1
#
# plt.figure(dpi=300)
# for LEARNING_RATE in [1, 0.1, 0.01, 0.001]:
#     ff_network = initialize_neural_network(False, 1)
#     train_error_log, test_error_log, final_epoch, min_error, idx_min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:orange')
#     # plt.plot(range(final_epoch), train_error_log, label='Train error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:blue')
#     np.savetxt("results/ff/learning_rate_"+str(LEARNING_RATE)+"_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/learning_rate_"+str(LEARNING_RATE)+"_test_error_log.csv", test_error_log, delimiter=",")
# plt.xlabel('Epoch')
# plt.ylabel('RMS error [-]')
# plt.grid()
# plt.legend()
# plt.show()

# # 2. Variable learning rate
# print('---\n VARIABLE LEARNING RATE\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# LEARNING_RATE = 1
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = True
#
# # plt.figure(dpi=300)
# ff_network = initialize_neural_network(False, 1)
# train_error_log, test_error_log, final_epoch, min_error, idx_min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
# print(min_error, idx_min_error)
# np.savetxt("results/ff/variable_learning_rate_train_error_log.csv", train_error_log, delimiter=",")
# np.savetxt("results/ff/variable_learning_rate_test_error_log.csv", test_error_log, delimiter=",")
# # plt.plot(range(final_epoch), test_error_log, label='Test error')#, c='tab:orange')
# # plt.plot(range(final_epoch), train_error_log, label='Train error')#, c='tab:blue')
# # plt.xlabel('Epoch')
# # plt.ylabel('RMS error [-]')
# # plt.ylim(0, 0.1)
# # plt.grid()
# # plt.legend()
# # plt.show()


# # 4. Batch size
# print('---\n BATCH SIZE\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# LEARNING_RATE = 1
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = True
# # plt.figure(dpi=300)
# for BATCH_SIZE in [2048, 8000]:#[32, 128, 512, 2048, 8000]:
#
#     # weights initialization
#     INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
#     OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
#     # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
#     CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
#     ff_network = initialize_neural_network(False, 1)
#     train_error_log, test_error_log, final_epoch, min_error, min_idx = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error, min_idx)
#     np.savetxt("results/ff/batch_size_"+str(BATCH_SIZE)+"_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/batch_size_"+str(BATCH_SIZE)+"_test_error_log.csv", test_error_log, delimiter=",")
#     # plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:orange')
#     # plt.plot(range(final_epoch), train_error_log, label='Train error, batch size {}'.format(str(BATCH_SIZE)))#, c='tab:blue')
# # plt.xlabel('Epoch')
# # plt.ylabel('RMS error [-]')
# # plt.grid()
# # plt.legend()
# # plt.show()
#
# # 3. Weight init
# print('---\n WEIGHT INIT\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# LEARNING_RATE = 1
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = True
# # plt.figure(dpi=300)
#
# for i in [2, 3]:
#     np.random.seed(i)
#     ff_network = initialize_neural_network(False, 1)
#
#     # plt.figure(dpi=300)
#     train_error_log, test_error_log, final_epoch, min_error, min_idx = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error, min_idx)
#     # plt.plot(range(final_epoch), train_error_log, label='Train error')
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('RMS error [-]')
#     # plt.ylim(0, 0.2)
#     # plt.xlim(0, MAX_EPOCHS)
#     # plt.grid()
#     # plt.legend()
#     # plt.show()
#     np.savetxt("results/ff/seed_"+str(i)+"_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/seed_" + str(i) + "_test_error_log.csv", test_error_log, delimiter=",")

# # 4. Weight init
# print('---\n UNIFORM INIT\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# LEARNING_RATE = 1
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = False
# plt.figure(dpi=300)
# for scaling in [0.5, 0.1, 0.01, 0.001, 0.0001]:
#     rbf_network = initialize_neural_network(True, scaling)
#
#     # plt.figure(dpi=300)
#     train_error_log, test_error_log, final_epoch, min_error, min_idx = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error, min_idx)
#     # plt.plot(range(final_epoch), train_error_log, label='Train error')
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('RMS error [-]')
#     # plt.ylim(0, 0.2)
#     # plt.xlim(0, MAX_EPOCHS)
#     # plt.grid()
#     # plt.legend()
#     # plt.show()
#     np.savetxt("results/ff/uniform_"+str(scaling)+"_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/uniform_" + str(scaling) + "_test_error_log.csv", test_error_log, delimiter=",")


# """OPTIMIZE NUMBER OF HIDDEN NEURONS FOR ALL OTHER PARAMETERS FIXED"""
# # 5. Number of hidden neurons
# print('---\n HIDDEN NEURONS OPTIMIZATION\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# NUMBER_OF_HIDDEN_NEURONS = 5
# np.random.seed(1)
# LEARNING_RATE = 1
# LEARNING_SCHEDULE_BOOL = True
#
#
# def optimizer(NUMBER_OF_HIDDEN_NEURONS):
#     rbf_network = initialize_neural_network(True, 0.01)
#     train_error_log, test_error_log, final_epoch, min_error, min_idx = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error, min_idx)
#     # plt.plot(range(final_epoch), train_error_log, label='Train error, {} hidden neurons'.format(str(NUMBER_OF_HIDDEN_NEURONS)))
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     # plt.xlabel('Epoch')
#     # plt.ylabel('RMS error [-]')
#     # plt.ylim(0, 0.2)
#     # plt.xlim(0, MAX_EPOCHS)
#     # plt.grid()
#     # plt.legend()
#     np.savetxt("results/ff/no_hidden_neurons_" + str(NUMBER_OF_HIDDEN_NEURONS) + "_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/no_hidden_neurons_" + str(NUMBER_OF_HIDDEN_NEURONS) + "_test_error_log.csv", test_error_log, delimiter=",")
#     return min_error
#
#
# error_log = np.zeros(50)
# NUMBER_OF_HIDDEN_NEURONS = 20
# previous_error = 100
# last_increase = True
# i = 0
# change_rate = 5
#
# plt.figure(dpi=300)
# while i < 30:
#     if i > 5:
#         change_rate = 2
#     if i > 10:
#         change_rate = 1
#     if error_log[NUMBER_OF_HIDDEN_NEURONS-1] > 0:
#         current_error = error_log[NUMBER_OF_HIDDEN_NEURONS-1]
#     else:
#         current_error = optimizer(NUMBER_OF_HIDDEN_NEURONS)
#     error_log[NUMBER_OF_HIDDEN_NEURONS - 1] = current_error
#     if last_increase:
#         if current_error < previous_error:
#             NUMBER_OF_HIDDEN_NEURONS += change_rate
#             last_increase = True
#         else:
#             NUMBER_OF_HIDDEN_NEURONS -= change_rate # no need to do times two, because, if we had a worse score it will go back to the already calculated one, which it will not recalculate so we will jump it either way
#             last_increase = False
#     else:
#         if current_error < previous_error:
#             NUMBER_OF_HIDDEN_NEURONS -= change_rate
#             last_increase = False
#         else:
#             NUMBER_OF_HIDDEN_NEURONS += change_rate
#             last_increase = True
#     previous_error = current_error
#     if NUMBER_OF_HIDDEN_NEURONS < 1:
#         NUMBER_OF_HIDDEN_NEURONS = 1
#     i+=1
#


"""BACKPROP vs LM"""
reset_variables()
np.random.seed(3)
LEARNING_RATE = 3
LM_DAMPING = 1
NUMBER_OF_HIDDEN_NEURONS = 10
LEARNING_SCHEDULE_BOOL = True
ACTIVATION_FUNCTION = sigmoid
DIFF_ACTIVATION_FUNCTION = diff_sigmoid

for i in [levenberg_marquardt, back_propagation]:
    OPTIMIZER = i

    # weights initialization
    INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
    OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
    # CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
    CENTERS_INIT = np.zeros(NUMBER_OF_HIDDEN_NEURONS)
    ff_network = initialize_neural_network(True, 0.01)
    train_error_log, test_error_log, final_epoch, min_error, idx_final = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)

    # output = []
    # for j in range(len(time_data[0])):
    #     output.append(ff_network.evaluate([time_data[0,j], time_data[1,j]]))

    optimizer_names = ['backpropagation', 'Levenberg Marquardt']
    np.savetxt("results/ff/compare_" + OPTIMIZER.__name__ + "_train_error_log.csv",
               train_error_log, delimiter=",")
    np.savetxt("results/ff/compare_" + OPTIMIZER.__name__ + "_test_error_log.csv",
               test_error_log, delimiter=",")
    # plt.subplot(1, 2, 1)
    # plt.plot(range(final_epoch), test_error_log, label='Test error, ' + OPTIMIZER.__name__)#, c='tab:orange')
    # plt.plot(range(final_epoch), train_error_log, label='Train error, ' + OPTIMIZER.__name__)#, c='tab:blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('RMS error [-]')

    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(time_data[0])), np.array(output)[:, 0], label='Time sequence, ' + OPTIMIZER.__name__)
# # plt.subplot(1, 2, 2)
# # plt.plot(range(len(time_data[0])), time_data[2,:], label='Time sequence, ' + OPTIMIZER.__name__)
# plt.grid()
# plt.legend()
# plt.show()

# # 1. Best
# print('---\n BEST LEARNING RATE\n ---')
# reset_variables()
# OPTIMIZER = back_propagation
# ACTIVATION_FUNCTION = sigmoid
# DIFF_ACTIVATION_FUNCTION = diff_sigmoid
# np.random.seed(1)
# LEARNING_SCHEDULE_BOOL = True
#
# NUMBER_OF_INPUTS = 2
# NUMBER_OF_OUTPUTS = 1
#
# plt.figure(dpi=300)
# for LEARNING_RATE in [3]:
#     ff_network = initialize_neural_network(True, 0.01)
#     train_error_log, test_error_log, final_epoch, min_error, idx_min_error = train_neural_network(ff_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), test_error_log, label='Test error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:orange')
#     # plt.plot(range(final_epoch), train_error_log, label='Train error, learning rate {}'.format(str(LEARNING_RATE)))#, c='tab:blue')
#     np.savetxt("results/ff/best"+str(LEARNING_RATE)+"_train_error_log.csv", train_error_log, delimiter=",")
#     np.savetxt("results/ff/best"+str(LEARNING_RATE)+"_test_error_log.csv", test_error_log, delimiter=",")