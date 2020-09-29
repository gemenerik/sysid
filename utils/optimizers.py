from numpy import array, split, eye
from numpy.linalg import inv

def back_propagation(learning_rate, jacobian, input_weights, type):
    weight_change = learning_rate * jacobian
    if len(weight_change) > 1:
        input_weights = array(input_weights)
        weight_change = array(weight_change.mean(axis=0))
        if type:
            output_weights = (input_weights.ravel() - weight_change.T)
            output_weights = array(split(output_weights.T, 2))
        else:
            output_weights = (input_weights.T - weight_change).T
    else:
        if type:
            output_weights = (input_weights.ravel() - weight_change.T)[0]
            output_weights = array(split(output_weights.T, 2))
        else:
            output_weights = (input_weights.T - weight_change).T
    return output_weights


def levenberg_marquardt(damping, jacobian, input_weights, error, type):
    # if type == True, get new input weights, otherwise; output weights
    weight_change = inv(jacobian.T.dot(jacobian) + damping * eye(len(jacobian[0])))
    weight_change = weight_change.dot(jacobian.T.dot(error))
    if type:
        output_weights = (input_weights.ravel() - weight_change.T)[0]
        output_weights = array(split(output_weights.T, 2))
    else:
        output_weights = (input_weights.T - weight_change.T).T
    return output_weights
