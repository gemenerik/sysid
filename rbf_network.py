from nn_batches import train_neural_network, NeuralNetwork, squared_error
import matplotlib.pyplot as plt
from utils.activation_functions import radial_basis_function, diff_radial_basis_function, sigmoid, diff_sigmoid
from utils.optimizers import back_propagation, levenberg_marquardt
import numpy as np

np.random.seed(1)
MAX_EPOCHS = 10000
GOAL = 0.0001
MIN_GRADIENT = 0.000001
LEARNING_RATE = 0.001
LM_DAMPING = 1
NUMBER_OF_HIDDEN_NEURONS = 5
# BATCH_SIZE = 128
BATCH_SIZE = 8000

NUMBER_OF_INPUTS = 2
NUMBER_OF_OUTPUTS = 1


def initialize_neural_network():
    INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
    OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
    CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)
    return NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
                         number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
                         max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
                         activation_function=radial_basis_function, optimizer=OPTIMIZER,
                         loss_function=squared_error, input_weights=INPUT_WEIGHTS_INIT,
                         output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
                         damping=LM_DAMPING, diff_activation_function=diff_radial_basis_function)


# weights initialization

# rbf_network = NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
#                             number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
#                             max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
#                             activation_function=radial_basis_function, optimizer=levenberg_marquardt,
#                             loss_function=squared_error, input_weights=INPUT_WEIGHTS_INIT,
#                             output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
#                             damping=LM_DAMPING, diff_activation_function=diff_radial_basis_function)

# load data
# train_data = np.genfromtxt('data/train_short.csv', delimiter=',')
# test_data = np.genfromtxt('data/test_short.csv', delimiter=',')
train_data = np.genfromtxt('data/train.csv', delimiter=',')
test_data = np.genfromtxt('data/test.csv', delimiter=',')

# for i in [1, 2, 3]:
#     plt.figure(dpi=300)
#     np.random.seed(i)
#     OPTIMIZER = levenberg_marquardt
#     rbf_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), train_error_log, label='Train error')
#     plt.plot(range(final_epoch), test_error_log, label='Test error')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#     plt.ylim(0, 1)
#     plt.grid()
#     plt.legend()
#     plt.show()

# np.random.seed(1)
# plt.figure(dpi=300)
# for LM_DAMPING in [50]:#[0.01, 0.1, 1, 10, 100]:
#     OPTIMIZER = levenberg_marquardt
#     rbf_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), train_error_log, label='Train error, ' + r'$\mu =$' + str(LM_DAMPING))
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#     plt.ylim(0, 1)
#     plt.grid()
#     plt.legend()
# plt.show()

# LM_DAMPING = 10
# plt.figure(dpi=300)
# for BATCH_SIZE in [32, 64, 128, 512]:#, 2048, 8000]:
#     OPTIMIZER = levenberg_marquardt
#     rbf_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), train_error_log, label='Train error, {} batch size'.format(str(BATCH_SIZE)))
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#     plt.ylim(0, 1)
#     plt.grid()
#     plt.legend()
# plt.show()
#

"""OPTIMIZE NUMBER OF HIDDEN NEURONS FOR ALL OTHER PARAMETERS FIXED"""
BATCH_SIZE = len(train_data[0])
LM_DAMPING = 10
plt.figure(dpi=300)
OPTIMIZER = levenberg_marquardt
# for NUMBER_OF_HIDDEN_NEURONS in [1, 2, 3, 5, 10, 20]:#, 10, 20]:

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
NUMBER_OF_HIDDEN_NEURONS = 10
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
#
# BATCH_SIZE = 64
# plt.figure(dpi=300)
# for LM_DAMPING in [0.01, 0.1, 1, 10, 100]:
#     OPTIMIZER = levenberg_marquardt
#     rbf_network = initialize_neural_network()
#     train_error_log, test_error_log, final_epoch, min_error = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)
#     print(min_error)
#     plt.plot(range(final_epoch), train_error_log, label='Train error, {} batch size'.format(str(BATCH_SIZE)))
#     # plt.plot(range(final_epoch), test_error_log, label='Test error')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMS error [-]')
#     plt.ylim(0, 1)
#     plt.grid()
#     plt.legend()
# plt.show()



# todo; plot results in time; approximation vs realll
# todo; update centers ?
