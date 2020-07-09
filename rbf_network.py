from nn_batches import *
import matplotlib.pyplot as plt

np.random.seed(1)
MAX_EPOCHS = 100
GOAL = 0.001
MIN_GRADIENT = 0
LEARNING_RATE = 0.001
LM_DAMPING = 1
NUMBER_OF_HIDDEN_NEURONS = 5
BATCH_SIZE = 128

NUMBER_OF_INPUTS = 2
NUMBER_OF_OUTPUTS = 1

# weights initialization
INPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS)
OUTPUT_WEIGHTS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS)
CENTERS_INIT = np.random.rand(NUMBER_OF_HIDDEN_NEURONS)

rbf_network = NeuralNetwork(number_of_inputs=NUMBER_OF_INPUTS, number_of_hidden_neurons=NUMBER_OF_HIDDEN_NEURONS,
                            number_of_outputs=NUMBER_OF_OUTPUTS, input_bias_weights=0, output_bias_weights=0, range=0,
                            max_epochs=MAX_EPOCHS, goal=GOAL, min_gradient=MIN_GRADIENT, learning_rate=LEARNING_RATE,
                            activation_function=radial_basis_function, optimizer=levenberg_marquardt,
                            loss_function=least_squares, input_weights=INPUT_WEIGHTS_INIT,
                            output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE,
                            damping=LM_DAMPING, diff_activation_function=diff_radial_basis_function)

# load data
train_data = np.genfromtxt('train.csv', delimiter=',')
test_data = np.genfromtxt('test.csv', delimiter=',')

train_error_log, test_error_log, final_epoch = train_neural_network(rbf_network, [train_data[0], train_data[1]], train_data[2], test_data)

plt.figure()
plt.plot(range(final_epoch), train_error_log, label='Train error')
plt.plot(range(final_epoch), test_error_log, label='Test error')
plt.xlabel('Epoch')
plt.ylabel('RMS error [-]')
plt.grid()
plt.legend()
plt.show()

# todo; plot results in time; approximation vs realll
# todo; update centers ?
