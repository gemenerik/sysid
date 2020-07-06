import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)


def radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2
    return a * np.exp(-v)


def diff_radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2)
    return -2*a*(x-c)*np.exp(-v)


def least_squares(desired_output, current_output):
    return 0.5 * np.square(desired_output - current_output)


def evaluate_error(net, desired_output, current_output):
    if net.loss_function == least_squares:
        error = least_squares(desired_output, current_output)
    else:
        raise Exception("Loss function not defined")
    return error


def levenberg_marquardt(damping, jacobian, input_weights, error, type):
    weight_change = np.linalg.inv(jacobian.T.dot(jacobian) + damping*np.eye(len(jacobian))).dot(jacobian.T*error)
    if type:
        output_weights = input_weights.ravel() - weight_change
        output_weights = np.array(np.split(output_weights, 2))
    else:
        output_weights = (input_weights.T - weight_change).T
    return output_weights


def train_neural_network(net, input, desired_output):
    """Trains neural networks
    input, desired_ouput are arrays of inputs and the ground truth / output
    """
    current_epoch = 0
    current_error = 1E3
    # input_weights = net.input_weights
    # output_weights = net.output_weights

    while current_epoch <= net.max_epochs:
        if current_error > net.goal:
            for i in range(len(input)):
                current_input = input[i]
                current_desired_output = desired_output[i]
                if net.activation_function == radial_basis_function:
                    if net.optimizer == levenberg_marquardt:
                        current_output, current_output_hidden_layer = net.evaluate(current_input)
                        current_error = evaluate_error(net, current_desired_output, current_output)
                        # print('Current input is ', current_input)
                        # print('Current output is ', current_output)
                        # print('Desired output is ', desired_output)
                        # print('Error is ', current_error)
                        input_jacobian, output_jacobian = net.get_jacobian(current_input, current_desired_output)
                        new_input_weights = levenberg_marquardt(LM_DAMPING, input_jacobian, net.input_weights, error=current_error, type = True)
                        new_output_weights = levenberg_marquardt(LM_DAMPING, output_jacobian, net.output_weights, error=current_error, type = False)
                        # print('Inputs ', net.input_weights, new_input_weights)
                        # print('Outputs ', net.output_weights, new_output_weights)
                        net.input_weights = new_input_weights
                        net.output_weights = new_output_weights
                        print('Epoch ', current_epoch, ' done.')
                        print(current_error)
                        current_epoch += 1
        else:
            break


class NeuralNetwork:
    """Defines all parameters of the neural network"""
    def __init__(self, number_of_inputs, number_of_hidden_neurons, number_of_outputs, input_bias_weights,
                 output_bias_weights, range, max_epochs, goal, min_gradient, learning_rate, activation_function,
                 optimizer, loss_function, input_weights, output_weights, centers, batch_size):
        self.number_of_inputs = number_of_inputs
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_outputs = number_of_outputs
        self.input_bias_weights = input_bias_weights
        self.output_bias_weights = output_bias_weights
        self.range = range
        self.max_epochs = max_epochs
        self.goal = goal
        self.min_gradient = min_gradient
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.input_weights = input_weights
        self.output_weights = output_weights
        self.centers = centers
        self.batch_size = batch_size

    def evaluate(self, current_input):
        current_output = np.zeros(self.number_of_outputs)
        current_output_hidden_layer = np.zeros(self.number_of_hidden_neurons)
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_neurons):
                for k in range(self.number_of_outputs):
                    xi = current_input[i]
                    yi = xi  # linear node
                    xj = yi * self.input_weights[i, j]
                    if self.activation_function == radial_basis_function:
                        yj = radial_basis_function(xj, 1, self.centers[j])  # todo; set a, check if centers should be 2D
                    else:
                        raise Exception("Activation function not defined")
                    current_output_hidden_layer += yj  # todo; what is output_hidden_layer used for in the Jacobian?
                    xk = yj * self.output_weights[j, k]
                    yk = xk  # linear node
                    current_output += yk
        return current_output, current_output_hidden_layer

    def get_jacobian(self, current_input, desired_output):
        jacobian_input_temp = np.zeros((len(self.input_weights), len(self.input_weights.T)))
        # jacobian_input=np.zeros((len(self.input_weights) * len(self.input_weights.T)))
        jacobian_output = np.zeros((len(self.output_weights) * len(self.output_weights.T)))
        output, output_hidden_layer = self.evaluate(current_input)
        for i in range(self.number_of_inputs):
            for j in range(self.number_of_hidden_neurons):
                jacobian_output[j] = -(desired_output - output) * output_hidden_layer[j]
                # [w11, w12, w13, w21, w22, w23]
                for k in range(self.number_of_outputs):  # which is just one in our case
                    # 0 through 5
                    temp1 = (-(desired_output - output)[k])
                    temp2 = temp1 * self.output_weights[j, k]
                    temp3 = temp2 * diff_radial_basis_function(current_input[i], 1, self.centers[j])  # todo; set a
                    jacobian_input_temp[i, j] = temp3 * current_input[i]
        jacobian_input = jacobian_input_temp.ravel()
        return jacobian_input, jacobian_output


MAX_EPOCHS = 100
GOAL = 0.001
MIN_GRADIENT = 0
LEARNING_RATE = 0.001
LM_DAMPING = 1
NUMBER_OF_HIDDEN_NEURONS = 3
BATCH_SIZE = 3

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
                            output_weights=OUTPUT_WEIGHTS_INIT, centers=CENTERS_INIT, batch_size=BATCH_SIZE)

# load data
data = np.genfromtxt('data.csv', delimiter=',')
print(data)

train_neural_network(rbf_network, [data[0], data[1]], data[2])
# print(rbf_network.input_weights, rbf_network.output_weights)
# print(rbf_network.evaluate([data[0][2], data[1][2]]), data[2][0])
