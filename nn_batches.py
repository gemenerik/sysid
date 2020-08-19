import numpy as np
import sys
import time
from utils.optimizers import levenberg_marquardt, back_propagation


def squared_error(desired_output, current_output):
    return np.square(desired_output - current_output)


def train_neural_network(net, input, desired_output, test_data):
    """Trains neural networks"""
    start_time = time.time()
    current_epoch = 0
    error_improvement = 1e3
    train_error = 1e3
    useless_iterations = 0
    epoch_steps = int(np.floor(len(input[0]) / net.batch_size))
    unused_data = len(input[0]) % net.batch_size
    if unused_data > 0:
        print(unused_data, 'unused datapoints due to data length & batch size combination.')
    assert net.batch_size <= len(input[0])

    test_error_log = []
    train_error_log = []

    stored_exception = False

    while current_epoch < net.max_epochs:
        try:
            if train_error > net.goal:
                if net.learning_schedule:
                    if current_epoch > 0:
                        if current_epoch % 100 == 0:
                            net.learning_rate *= 0.9
                if error_improvement < net.min_gradient:
                    useless_iterations += 1
                else:
                    useless_iterations = 0
                if useless_iterations >= 1000:
                    end_time = time.time()
                    print('Runtime ', end_time - start_time)
                    return train_error_log, test_error_log, current_epoch, min(train_error_log)
                # train_output = []
                train_error = []

                test_output = []
                for z in range(len(test_data[0])):
                    test_output.append(net.evaluate([test_data[0, z], test_data[1, z]])[0])
                desired_test_output = test_data[2, :]
                # test_error = 1 / len(test_data[0]) * np.sum(np.square(np.array(desired_test_output) - test_output))
                test_error = (np.square(desired_test_output - test_output)).mean(axis=None)
                test_error_log.append(test_error)

                # train_output = []
                # for z in range(len(input[0])):
                #     train_output.append(net.evaluate([np.array(input)[0, z], np.array(input)[1, z]])[0])
                # desired_train_output = desired_output[:]
                # # train_error = 1 / len(input[0]) * np.sum(np.square(np.array(desired_train_output) - train_output))
                # train_error = (np.square(desired_train_output - train_output)).mean(axis=None)
                # print(train_error)

                for i in range(epoch_steps):
                    current_input = np.array(input)[:, i*net.batch_size:i*net.batch_size+net.batch_size]
                    current_desired_output = np.array(desired_output[i*net.batch_size:i*net.batch_size+net.batch_size])
                    batch_output, batch_hidden_output = net.evaluate(current_input)
                    current_error = net.evaluate_error(current_desired_output, batch_output).T
                    train_error.append(current_error)
                    input_jacobian, output_jacobian = net.get_jacobian(current_input, current_desired_output)
                    if net.optimizer == levenberg_marquardt:
                        new_input_weights = levenberg_marquardt(net.damping, input_jacobian, net.input_weights,
                                                                error=current_error, type = True)
                        new_output_weights = levenberg_marquardt(net.damping, output_jacobian, net.output_weights,
                                                                 error=current_error, type = False)
                    elif net.optimizer == back_propagation:
                        new_input_weights = back_propagation(net.learning_rate, input_jacobian, net.input_weights, type=True)
                        new_output_weights = back_propagation(net.learning_rate, output_jacobian, net.output_weights, type=False)
                    else:
                        raise Exception('Incorrect optimizer defined')
                    net.input_weights = new_input_weights
                    net.output_weights = new_output_weights

                train_error = (np.array(train_error).mean(axis=None))

                if current_epoch > 0:
                    error_improvement = min(train_error_log) - train_error
                train_error_log.append(train_error)
                current_epoch += 1
                print('Epoch ', current_epoch)
                print('Train error is ', train_error)
                print('Test error is ', test_error)

            if stored_exception:
                end_time = time.time()
                print('Runtime ', end_time - start_time)
                return train_error_log, test_error_log, current_epoch, min(train_error_log)
        except KeyboardInterrupt:
            stored_exception = sys.exc_info()
    end_time = time.time()
    print('Runtime ', end_time - start_time)
    return train_error_log, test_error_log, current_epoch, min(train_error_log)


class NeuralNetwork:
    """Defines all parameters of the neural network"""

    def __init__(self, number_of_inputs, number_of_hidden_neurons, number_of_outputs, input_bias_weights,
                 output_bias_weights, range, max_epochs, goal, min_gradient, learning_rate, activation_function,
                 optimizer, loss_function, input_weights, output_weights, centers, batch_size, damping,
                 diff_activation_function, learning_schedule):
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
        self.diff_activation_function = diff_activation_function
        self.learning_schedule = learning_schedule

    def evaluate(self, input):
        """Feeds forward inputs and return output of network"""
        input = np.array(input)
        if input.ndim > 1:
            # input shape (2, 128)
            output = np.zeros((self.number_of_outputs, np.size(input, 1)))
            output_hidden_layer = np.zeros((self.number_of_hidden_neurons, np.size(input, 1)))
        else:
            output = np.zeros(self.number_of_outputs)
            output_hidden_layer = np.zeros(self.number_of_hidden_neurons)
        for i in range(self.number_of_inputs):
            xi = input[i]
            yi = xi  # linear node
            for j in range(self.number_of_hidden_neurons):
                xj = yi * self.input_weights[i, j]
                yj = self.activation_function(xj, 1, self.centers[j])  # todo; update centers
                output_hidden_layer += yj
                for k in range(self.number_of_outputs):
                    xk = yj * self.output_weights[j, k]
                    yk = xk  # linear node
                    output += yk
        return output, output_hidden_layer

    def evaluate_error(self, desired_output, current_output):
        if self.loss_function == squared_error:
            error = squared_error(desired_output, current_output)
        else:
            raise Exception("Loss function not defined")
        return error

    def get_jacobian(self, current_input, desired_output):
        """Returns Jacobian with respect to both input and output weights"""

        batch_jacobian_input = []
        batch_jacobian_output = []
        output, output_hidden_layer = self.evaluate(current_input)
        for z in range(self.batch_size):
            jacobian_input_temp = np.zeros((len(self.input_weights), len(self.input_weights.T)))
            # jacobian_input=np.zeros((len(self.input_weights) * len(self.input_weights.T)))
            jacobian_output = np.zeros((len(self.output_weights) * len(self.output_weights.T)))
            for i in range(self.number_of_inputs):
                for j in range(self.number_of_hidden_neurons):
                    jacobian_output[j] = -(desired_output[z] - output[:,z]) * output_hidden_layer[j, z]
                    # [w11, w12, w13, w21, w22, w23]
                    for k in range(self.number_of_outputs):  # which is just one in our case
                        # 0 through 5
                        temp1 = (-(desired_output - output)[k])
                        temp2 = temp1 * self.output_weights[j, k]
                        temp3 = temp2 * self.diff_activation_function(current_input[i, z], 1, self.centers[j])  # todo; set a
                        jacobian_input_temp[i, j] = temp3[z] * current_input[i, z]
            jacobian_input = jacobian_input_temp.ravel()
            batch_jacobian_input.append(jacobian_input)
            batch_jacobian_output.append(jacobian_output)
        return np.array(batch_jacobian_input), np.array(batch_jacobian_output)
