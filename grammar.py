# ------------------------------------------------------------------------------
# Replication of:
#   Learning the Structure of Event Sequences,
#   Axel Cleeremans & James L. McClelland
#   Journal of Experimental Psychology: General 120(3), 1991.
# ------------------------------------------------------------------------------
import numpy as np

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)


def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2


class Elman:
    ''' Elamn network '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []

        # Input layer (+ size of first hidden layer
        #              +1 unit for bias)
        self.layers.append(np.zeros(self.shape[0]+self.shape[1]+1))

        # Hidden layer(s)
        for i in range(1, n-1):
            self.layers.append(np.zeros(self.shape[i]+1))

        # Output layer (no bias)
        self.layers.append(np.zeros(self.shape[-1]))

        # Build weights matrix
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0, ]*len(self.weights)

        # Reset weights
        self.reset_weights()

        # Reset weights
        self.reset_activity()

    def reset_weights(self):
        ''' Reset weights '''
        for i in range(len(self.weights)):
            shape = self.weights[i].shape
            self.weights[i] = np.random.uniform(-0.1, +0.1, shape)

    def reset_activity(self):
        ''' Reset activity '''

        # Reset activity
        for i in range(len(self.layers)):
            self.layers[i][...] = 0

        # Set bias
        for i in range(len(self.layers)-1):
            self.layers[i][-1] = 1

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer with data
        self.layers[0][:self.shape[0]] = data

        # and first hidden layer
        self.layers[0][self.shape[0]:-1] = self.layers[1][:-1]

        # Propagate from layer 0 to layer n-1 using sigmoid
        # as activation function
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][:] = sigmoid(
                np.dot(self.layers[i-1], self.weights[i-1]))

        # Return output
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.005, momentum=0.01):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2, 0, -1):
            delta = (np.dot(deltas[0], self.weights[i].T) *
                     dsigmoid(self.layers[i]))
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


def generate_sequences(n=100, length=20, debug=False):
    grammar = {
        0: [('V', 1), ('P', 3)],
        1: [('S', 2), ('X', 5)],
        2: [('Q', 1)],
        3: [('T', 4), ('X', 6)],
        4: [('V', 3)],
        5: [('Q', 3), ('S', 0)],
        6: [('P', 5), ('T', 0)]
    }

    sequences = []
    for i in range(n):
        index = 0
        sequence = ''
        for j in range(length):
            choices = grammar[index]
            choice = np.random.randint(0, len(choices))
            token, index = choices[choice]
            #print "token : ", token, " - index : ", index
            sequence += token
        if debug:
            print(sequence)
        sequences.append(sequence)
    return sequences


def generate_samples(sequences):
    code = {
        'V': [1, 0, 0, 0, 0, 0],
        'P': [0, 1, 0, 0, 0, 0],
        'Q': [0, 0, 1, 0, 0, 0],
        'X': [0, 0, 0, 1, 0, 0],
        'S': [0, 0, 0, 0, 1, 0],
        'T': [0, 0, 0, 0, 0, 1]
    }

    samples = np.zeros((len(sequences), (len(sequences[0])-1)),
                       dtype=[('input',  float, 6), ('output', float, 6)])
    index = 0
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)-1):
            samples[i, j]["input"] = code[sequence[j]]
            samples[i, j]["output"] = code[sequence[j+1]]
            index += 1

    return samples


# ------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    network = Elman(6, 24, 6)
    sequences = generate_sequences(n=100, length=20, debug=True)
    # print(sequences)
    samples = generate_samples(sequences)
    # print(samples)

    errors = []
    mean_error = []
    for i in range(1000):
        index = i % len(samples)
        sample = samples[index]
        network.reset_activity()
        for j in range(len(sample)):
            network.propagate_forward(sample['input'][j])
            error = network.propagate_backward(sample['output'][j])
            errors.append(error)
            mean_error.append(np.mean(errors[-50:]))

    plt.plot(mean_error)
    plt.show()
    
    for i in range(len(samples)):
        sample = samples[i]
        print("Sample %d" % i)
        network.reset_activity()
        for j in range(len(sample)):
            o = network.propagate_forward(sample['input'][j])
            print('%2d: expected: %s' % (j, sample['output'][j]))
            print('    network : %s' % (o == o.max()).astype(float))
        print()
