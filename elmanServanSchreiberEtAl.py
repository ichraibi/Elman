#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Elman reccurent network
# Copyright (C) 2011  Nicolas P. Rougier modified in 2016 by Ikram Chraibi K.
#
# 
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
# ==> Clearmans & McCelland, 1988 & 1989
# -----------------------------------------------------------------------------
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import random
import sys



def generate_sequences(n=100, debug=False):
    grammar = {
        0: [('T', 1), ('P', 3)],
        1: [('S', 1), ('X', 3)],
        2: [('T', 2), ('V', 4)],
        3: [('X', 2), ('S', 5)],
        4: [('P', 3), ('V', 5)]
    }

    sequences = []
    for i in range(n):
        sequence = 'B'
        index =0
        while index<>5:
            choices = grammar[index]
            choice = np.random.randint(0, len(choices))
            token, index = choices[choice]
            sequence += token

        sequence += 'E'
        if debug:
            print(sequence)

        sequences.append(sequence)

    return sequences


def generateSequence(G, edge_labels):

    aSequence = []
    aSequence.append("B")

    CurrentNode = "0"
    while (CurrentNode<>"5"):
        #we get the neighbors of the currentNode
        ListNeighbors = G.neighbors(CurrentNode)
        UpdatedSequence= False
        while (UpdatedSequence==False):
            #we pick randomly one neighbor
            aNeighbor = random.choice(ListNeighbors)
            #we check if the edge (currentNode-neighbor) respect the grammar
            if (CurrentNode, aNeighbor) in G.edges():
                aSequence.append(edge_labels[(CurrentNode, aNeighbor)])
                CurrentNode= aNeighbor
                UpdatedSequence = True

    if CurrentNode == "5":
        aSequence.append("E")


    return aSequence

def generateRandomSequence(ListNeighbors):

    aSequence = []
    CurrentLetter = "B"
    aSequence.append(CurrentLetter)
    #ListNeighbors = "T", "S", "X", "V", "P", "E"
    while (CurrentLetter<>"E"):
        aNeighbor = random.choice(ListNeighbors)
        #print "aNeighbor : ", aNeighbor
        aSequence.append(aNeighbor)
        CurrentLetter = aNeighbor

    #print "\nrandom Sequence : ",aSequence
    return aSequence


def getPotentialSuccesors(networkOutput,OutputLettersDict):

    successors = []
    output= networkOutput.tolist()
    #print " output : ", output
    for i in range(0,len(output)):
        #print type(output[i]), " - ", output[i]
        if output[i]>0.3:
            #print " - valeur: ", output[i]
            binaryVector = [0] * len(output)
            binaryVector[i]=1
            #print "binaryVectors", binaryVector, "--> ",
            successor = getLetterByBinaryValue(OutputLettersDict,binaryVector)
            #print successor
            successors.append(successor)

    return successors


def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def pstdev(data):
    """Calculates the population standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/n # the population variance
    return pvar**0.5

def analyseGrammaticalStructureOfSequences(network, sequences, LettersDict):

    nbNonGrammaticalSequence = 0
    past = []
    for i, sequence in enumerate(sequences):
        #print "\n\n\n Sequence : ", sequence
        network.reset_activity_ContextUnits()
        past = []
        for j in range(len(sequence) - 1):
            o = network.propagate_forward(LettersDict[sequence[j]])
            past.append(sequence[j])

            #print "\nPotential Succesors of : ", sequence[j]
            #print " Past : ", past
            successors = getPotentialSuccesors(o, LettersDict)
            #print "==> successors : ", successors
            """
            if not successors:
                print "\ni : ", i, " - range(len(sequence) - 1) : ", range(len(sequence) - 1)
                print "  -->j : ", j
                print "input letter : ", sequence[j], " - expected letter : ", sequence[j + 1]
                print "output : ", o
                print " --> successors : ", successors
                print " --> Network Result : ", SRNLetter
            """

            networkOutput = (o == o.max()).astype(float)
            SRNLetter = ""
            SRNLetter = getLetterByBinaryValue(LettersDict, networkOutput.tolist())

            followingLetter = sequence[j + 1]
            if followingLetter not in successors:
                nbNonGrammaticalSequence+=1
                break

            if sequence[j + 1] == "E":
                break

    NonGrammaticalSequencesPercent = float(nbNonGrammaticalSequence) / float(len(sequences))
    GrammaticalSequencesPercent = 1- NonGrammaticalSequencesPercent
    print "\nPourcentage of grammatical sequences: ", (GrammaticalSequencesPercent*100), " %"
    print "Pourcentage of non grammatical sequences : ", (NonGrammaticalSequencesPercent*100), " %"



def getEdgeBylabel(G, edge_label, edges_labels):

    anEdge=()
    for oneEdge in G.edges():
        if edges_labels[oneEdge] == edge_label:
            anEdge = oneEdge
            break


    return anEdge

def getTargetofAnEdge(G, edge):
    
    aTarget = ""
    for e in G.edges():
        if e==edge:
            source,target = e
            aTarget = target
            break

    return aTarget

def getLetterByBinaryValue(LettersDict,ABinaryValue):

    theLetter =""
    for KeyLetter, BinaryLetter in LettersDict.iteritems():
        #print KeyLetter
        #print BinaryLetter
        if (BinaryLetter==ABinaryValue):
            theLetter = KeyLetter
            break

    #print "theLetter : ",theLetter
    return theLetter

def generate_samples(sequences, LettersDict, nbSamples):

    samples = np.zeros(nbSamples,
                       dtype=[('input', float, 7), ('output', float, 7)])

    index = 0
    for i, sequence in enumerate(sequences):
        for j in range(len(sequence)-1):

            if max(LettersDict[sequence[j]]) == 0 or max(LettersDict[sequence[j + 1]]) == 0:
                print "ERROR : i ", i, " - j : ", j
                quit()

            samples[index] = LettersDict[sequence[j]], LettersDict[sequence[j+1]]
            index += 1
            if sequence[j + 1]=="E":
                break

    return samples



def createFiniteStateGrammar89():
    # Creation of the finite state grammar as a networkx directed graph
    print "\n\nCreation of the grammar "
    G = nx.DiGraph()

    G.add_edge("0", "1", name="T")
    G.add_edge("1", "1", name="S")
    G.add_edge("1", "3", name="X")
    G.add_edge("3", "2", name="X")
    G.add_edge("3", "5", name="S")

    G.add_edge("0", "2", name="P")
    G.add_edge("2", "2", name="T")
    G.add_edge("2", "4", name="V")
    G.add_edge("4", "3", name="P")
    G.add_edge("4", "5", name="V")

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="white")

    # edges
    nx.draw_networkx_edges(G, pos,
                           width=6, alpha=0.5, edge_color='black')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    edge_labels = nx.get_edge_attributes(G, 'name')
    #print "edges_label : ", edge_labels
    nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

    plt.axis('off')
    plt.savefig("finiteStateGrammarNov88.png")  # save as png
    # plt.show() # display

    return G


def create_dictLettersWithBinaryValues(LetterList):

    LettersDict = {}
    print "\n\nGeneration of the binary vectors for the letters"
    i = 0
    while (i < len(LetterList)):
        currentLetter = LetterList[i]
        ABinaryLetter = []
        for aletter in LetterList:
            # print " --> : ", aletter
            if aletter == currentLetter:
                ABinaryLetter.append(1)
            else:
                ABinaryLetter.append(0)

        LettersDict[currentLetter] = ABinaryLetter
        #print "currentLetter : ", currentLetter, " => ", ABinaryLetter

        i = i + 1


    for key, value in LettersDict.iteritems():
        foundLetter = ""
        foundLetter = getLetterByBinaryValue(LettersDict,value)

        print key, " - ", value, " --> ", foundLetter

    #quit()
    return LettersDict



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
            #print "len(self.layers[i]) : ",len(self.layers[i])

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
            self.weights[i] = np.random.uniform(-0.5, +0.5, shape)

    def reset_activity_ContextUnits(self):

        #Context unit in Input Layer
        #Structure of the input layer : self.shape[0] input unit + self.shape[1] context units + 1 biais
        #print "valeur des unites de la couche de contexte :"
        for j in range(self.shape[0], (len(self.layers[0])-1)):
           self.layers[0][j] = 0

        #print "Input layer : ",
        #for j in range((len(self.layers[0])-1)):
        #   print self.layers[0][j],


        # Hidden layer(s)
        #print "\nvaleur des unites de la couche cachee:"
        #for i in range(1, len(self.shape) - 1):
        #    for j in range(len(self.layers[i])-1):
        #        print "self.layers[i][j] : ",self.layers[i][j]#," j --> ", j
        #        self.layers[i][j]=0.5


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

    def propagate_backward(self, target, lrate=0.01, momentum=0.01):
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




# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    if len(sys.argv) <=1:
        print "\nParameters into the code:"
        #parametersToModify    
        #SequenceSize = 6 #it need to be an even number (un chiffre pair)
        NumberOfSequences = 100
        window = 10
        #print "Size of sequence: ",SequenceSize
        print "Number of sequences : ", NumberOfSequences
        print "Window of average computation: ", window
         
    else:
        print "\nParameters :"
        #SequenceSize = int(sys.argv[1]) #it need to be an even number (un chiffre pair)
        NumberOfSequences = int(sys.argv[1])
        window = int(sys.argv[2])
        #print "Size of sequence: ",SequenceSize
        print "Number of sequences : ", NumberOfSequences
        print "Window of average computation: ", window


    #G = createFiniteStateGrammar89()

    LettersDict = {}
    LettersList = ["B", "T", "S", "X", "V", "P", "E"]

    LettersDict= create_dictLettersWithBinaryValues(LettersList)
    print "\n Letters list : "
    for aLetter in LettersList:
        print aLetter, "-->", LettersDict[aLetter]


    print "\n\nGeneration of the sequences for TRAINING ..."
    #edge_labels = nx.get_edge_attributes(G,'name')
    
    numSeq=0
    randomSequences = []
    nbSamples=0
    LenSequences = []
    sequences = generate_sequences(n=NumberOfSequences)

    while numSeq<len(sequences):
        sequence = sequences[numSeq]
        LenSequences.append(len(sequence))
        nbSamples += len(sequence)-1
        numSeq += 1

    print "Generation of ", NumberOfSequences, " sequences done !"
    print " --> The average len is : ", mean(LenSequences)
    print " --> The standard deviation is : ", pstdev(LenSequences)


    print "\n\nGeneration of samples & LEARNING ...."

    samples = np.zeros(nbSamples,
                       dtype=[('input', float, 7), ('output', float, 7)])
    network = Elman(7, 3, 7)
    errors = []
    mean_error = []
    index =0
    nbNonGrammaticalSequence=0
    for i in range(len(sequences)):
        #print " i :", i,
        #index = i % len(sequences)

        sequence= sequences[i]
        #print " -sequence : ", sequence
        #print "\nreset_activity_HiddenLayer : "
        network.reset_activity_ContextUnits()

        for j in range(len(sequence)-1):
            #print " - j ", sequence[j], " -> ", sequence[j + 1]
            #print " -> index : ", index
            samples[index] = LettersDict[sequence[j]], LettersDict[sequence[j + 1]]
            sample = samples[index]
            index +=1

            if (sample['input']).max() == 0 or (sample['output']).max() == 0:
                print "ERROR : index ", index
                print  sample
                quit()

            #print "sample : ", sample
            o= network.propagate_forward(sample['input'])
            error = network.propagate_backward(sample['output'])#the actual successor
            # print "error : ", error
            errors.append(error)
            mean_error.append(np.mean(errors[-window:]))


            if sequence[j + 1] == "E":
                break


    print "Total number of samples : ", nbSamples
    #print "len(samples): ", len(samples)
    #print "samples.size : ", samples.size
    #print "Generation of ", nbSamples, " samples done !"

    plt.clf()

    plt.plot(mean_error)
    plt.title('Elman neural network:\nevolution of the error during learning process')

    plt.xlabel('Samples')

    plt.ylabel('error')
    plt.gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text
    plt.figtext(.05, .05, '\nNumber of sequences : ' + str(NumberOfSequences) + '\nNumber of samples : ' + str(nbSamples) + '\nAverage of the error every ' + str(window) + ' results')

    plt.savefig("NetworkRes_NbSeq" + str(NumberOfSequences) + "-Win" + str(window) + ".png")  # save as png
    plt.show()


    listNetworkResult = []
    listGrammaticalSequencesResult = []
    nbNonGrammaticalSequence = 0

    print "\n\nAnalyse the grammatical structure of ",NumberOfSequences," sequences "
    analyseGrammaticalStructureOfSequences(network, sequences, LettersDict)

    #quit()

    print "\n\nNetwork analysis for 20 000 sequences.... "#-------------------------------------------------------------------

    numSeq = 0
    randomSequences = []
    nbSamples = 0
    LenSequences = []
    sequences = generate_sequences(n=20000)

    while numSeq < len(sequences):
        sequence = sequences[numSeq]
        LenSequences.append(len(sequence))
        nbSamples += len(sequence) - 1
        numSeq += 1

    print "Generation of ", 20000, " sequences done !"
    print " --> The average len is : ", mean(LenSequences)
    print " --> The standard deviation is : ", pstdev(LenSequences)

    print "\n\nAnalyse the grammatical structure of 20 000 sequences "
    analyseGrammaticalStructureOfSequences(network, sequences, LettersDict)

    quit()

    """
    i=0
    for i in range(len(samples)):
        print "i : ", i
        sample = samples[i]

        if (sample['input']).max() == 0:
            print "ERROR : i ", i
            print  sample
            quit()

        o = network.propagate_forward(sample['input'])

        networkOutput = (o == o.max()).astype(float)

        foundInputLetter = ""
        foundInputLetter = getLetterByBinaryValue(LettersDict, sample['input'].tolist())
        expectedOutputLetter = ""
        expectedOutputLetter = getLetterByBinaryValue(LettersDict, sample['output'].tolist())
        SRNLetter = ""
        SRNLetter = getLetterByBinaryValue(LettersDict, networkOutput.tolist())

        if (sample['output']==networkOutput).all() :
            listNetworkResult.append(1)
        else:
            listNetworkResult.append(0)


    succesRate = float(listNetworkResult.count(1))/float(len(listNetworkResult))
    print "Pourcentage succes of Network: ", (succesRate*100), " %"
    failureRate =  float(listNetworkResult.count(0))/float(len(listNetworkResult))
    print "Pourcentage echec of Network : ", (failureRate*100), " %"

    print "\n\nAnalyse the grammatical structure of sequences "
    analyseGrammaticalStructureOfSequences(network, sequences, LettersDict)



    """

  