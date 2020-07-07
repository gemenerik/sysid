import numpy as np
import sys


def radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2
    return a * np.exp(-v)


def diff_radial_basis_function(x, a, c):
    v = (x - c) ** 2
    # v = np.square(w) * (x - c)**2)
    return -2*a*(x-c)*np.exp(-v)


def least_squares(desired_output, current_output):
    return np.square(desired_output - current_output)


def evaluate_error(net, desired_output, current_output):
    if net.loss_function == least_squares:
        error = least_squares(desired_output, current_output)
    else:
        raise Exception("Loss function not defined")
    return error


def levenberg_marquardt(damping, jacobian, input_weights, error, type):
    # weight_change = np.linalg.inv(jacobian.T.dot(jacobian) + damping*np.eye(len(jacobian[0]))).dot(jacobian.T*error)
    weight_change = np.linalg.inv(jacobian.T.dot(jacobian) + damping * np.eye(len(jacobian[0])))
    weight_change = weight_change.dot(jacobian.T.dot(error))
    if type:
        output_weights = (input_weights.ravel() - weight_change.T)[0]
        output_weights = np.array(np.split(output_weights.T, 2))
    else:
        output_weights = (input_weights.T - weight_change.T).T
    return output_weights


def train_neural_network(net, input, desired_output, test_data):
    """Trains neural networks
    input, desired_ouput are arrays of inputs and the ground truth / output
    """
    current_epoch = 0
    train_error = 1e3
    epoch_steps = int(np.floor(len(input[0]) / net.batch_size))
    unused_data = len(input[0]) % net.batch_size
    if unused_data > 0:
        print(unused_data, 'unused datapoints due to data length & batch size combination.')
    assert net.batch_size <= len(input[0])

    test_error_log = []
    train_error_log = []

    stored_exception = False

    while current_epoch <= net.max_epochs:
        try:
            if train_error > net.goal:
                for i in range(epoch_steps):
                    current_input = np.array(input)[:, i*net.batch_size:i*net.batch_size+net.batch_size]
                    current_desired_output = np.array(desired_output[i*net.batch_size:i*net.batch_size+net.batch_size])
                    if net.activation_function == radial_basis_function:
                        if net.optimizer == levenberg_marquardt:
                            batch_output = []
                            current_error = []
                            for j in range(net.batch_size):
                                current_output, current_output_hidden_layer = net.evaluate(current_input[:, j])
                                batch_output.append(current_output)
                                current_error.append(evaluate_error(net, current_desired_output[j], current_output))

                            input_jacobian, output_jacobian = net.get_jacobian(current_input, current_desired_output)
                            new_input_weights = levenberg_marquardt(net.damping, input_jacobian, net.input_weights, error=current_error, type = True)
                            new_output_weights = levenberg_marquardt(net.damping, output_jacobian, net.output_weights, error=current_error, type = False)
                            net.input_weights = new_input_weights
                            net.output_weights = new_output_weights
                current_epoch += 1

                # todo; evaluate error of training set
                test_output = []
                for z in range(len(test_data[0])):
                    test_output.append(net.evaluate([test_data[0,z], test_data[1,z]])[0])
                desired_test_output = test_data[2,:]
                # test_error = 1 / len(test_data[0]) * np.sum(np.square(np.array(desired_test_output) - test_output))
                test_error = (np.square(desired_test_output - test_output)).mean(axis=None)
                test_error_log.append(test_error)

                train_output = []
                for z in range(len(input[0])):
                    train_output.append(net.evaluate([np.array(input)[0, z], np.array(input)[1, z]])[0])
                desired_train_output = desired_output[:]
                # train_error = 1 / len(input[0]) * np.sum(np.square(np.array(desired_train_output) - train_output))
                train_error = (np.square(desired_train_output - train_output)).mean(axis=None)
                train_error_log.append(train_error)
                print(np.shape(np.array(desired_train_output) - train_output))

                print('Epoch ', current_epoch)
                print('Train error is ', train_error)
                print('Test error is ', test_error)

            if stored_exception:
                return train_error_log, test_error_log, current_epoch
        except KeyboardInterrupt:
            stored_exception = sys.exc_info()
    return train_error_log, test_error_log, current_epoch


class NeuralNetwork:
    """Defines all parameters of the neural network"""
    def __init__(self, number_of_inputs, number_of_hidden_neurons, number_of_outputs, input_bias_weights,
                 output_bias_weights, range, max_epochs, goal, min_gradient, learning_rate, activation_function,
                 optimizer, loss_function, input_weights, output_weights, centers, batch_size, damping):
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
        self.damping = damping

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
        batch_jacobian_input = []
        batch_jacobian_output = []
        for z in range(self.batch_size):
            jacobian_input_temp = np.zeros((len(self.input_weights), len(self.input_weights.T)))
            # jacobian_input=np.zeros((len(self.input_weights) * len(self.input_weights.T)))
            jacobian_output = np.zeros((len(self.output_weights) * len(self.output_weights.T)))
            output, output_hidden_layer = self.evaluate(current_input[:, z])
            for i in range(self.number_of_inputs):
                for j in range(self.number_of_hidden_neurons):
                    jacobian_output[j] = -(desired_output[z] - output) * output_hidden_layer[j]
                    # [w11, w12, w13, w21, w22, w23]
                    for k in range(self.number_of_outputs):  # which is just one in our case
                        # 0 through 5
                        temp1 = (-(desired_output - output)[k])
                        temp2 = temp1 * self.output_weights[j, k]
                        temp3 = temp2 * diff_radial_basis_function(current_input[i, z], 1, self.centers[j])  # todo; set a
                        jacobian_input_temp[i, j] = temp3 * current_input[i, z]
            jacobian_input = jacobian_input_temp.ravel()
            batch_jacobian_input.append(jacobian_input)
            batch_jacobian_output.append(jacobian_output)
        return np.array(batch_jacobian_input), np.array(batch_jacobian_output)
