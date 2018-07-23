# refer to https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial5/dqn-mountaincar.py
# import logging
# logging.basicConfig(level=logging.INFO)
import random
import numpy as np
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.models import load_model
import os

# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists
    that get returned as another list of dictionaries with each key corresponding to either
    "state", "action", "reward", "nextState" or "isFinal".
    """
    def __init__(self, size, player_name):
        self.size = size
        self.currentPosition = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.newStates = []
        self.finals = []

    def getMiniBatch(self, size):
        indices = random.sample(np.arange(len(self.states)), min(size, len(self.states)))
        miniBatch = []
        for index in indices:
            miniBatch.append({'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                              'newState': self.newStates[index], 'isFinal': self.finals[index]})
        return miniBatch

    def getCurrentSize(self):
        return len(self.states)

    def getMemory(self, index):
        return {'state': self.states[index], 'action': self.actions[index], 'reward': self.rewards[index],
                'newState': self.newStates[index], 'isFinal': self.finals[index]}

    def addMemory(self, state, action, reward, newState, isFinal):
        if (self.currentPosition >= self.size - 1):
            self.currentPosition = 0
        if (len(self.states) > self.size):
            self.states[self.currentPosition] = state
            self.actions[self.currentPosition] = action
            self.rewards[self.currentPosition] = reward
            self.newStates[self.currentPosition] = newState
            self.finals[self.currentPosition] = isFinal
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.newStates.append(newState)
            self.finals.append(isFinal)

        self.currentPosition += 1


class DeepQ(object):
    """ 
    DQN abstraction.
        Traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s'))
    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart, player_name):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = Memory(memorySize, player_name)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
        self.player_name = player_name

    def initNetworks(self, hiddenLayers):
        model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.model = model

        targetModel = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.targetModel = targetModel

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            if (activationType == "LeakyReLU"):
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                # print("adding layer "+str(index))
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
                if (activationType == "LeakyReLU"):
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        # model.summary()
        return model

    def saveModel(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        self.model.save('model/' + self.player_name + '-dqn_model')
        self.targetModel.save('model/' + self.player_name + '-dqn_target_model')

    def loadModel(self):
        print 'load model from', 'model/' + self.player_name + '-dqn_model'
        self.model = load_model('model/' + self.player_name + '-dqn_model')
        print 'load model from', 'model/' + self.player_name + '-dqn_target_model'
        self.targetModel = load_model('model/' + self.player_name + '-dqn_target_model')

    def modelExisted(self):
        if os.path.exists('model/' + self.player_name + '-dqn_model') and os.path.exists('model/' + self.player_name + '-dqn_target_model'):
            return True
        else:
            return False

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print "layer ", i, ": ", weights
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1
        self.saveModel()
        
    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1, len(state)))
        return predicted[0]

    def getTargetQValues(self, state):
        predicted = self.targetModel.predict(state.reshape(1, len(state)))
        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """ target = reward(s,a) + gamma * max(Q(s') """
        if isFinal:
            return reward
        else:
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0, self.input_size), dtype=np.float64)
            Y_batch = np.empty((0, self.output_size), dtype=np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else:
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward] * self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size=len(miniBatch), epochs=1, verbose=0)


class DeepQTrain(DeepQ):
    def __init__(self, inputSize, outputSize, player_name):
        
        self.stepCounter = 0
        self.updateTargetNetworkCounter = 1000 # 10000
        self.explorationRate = 0.9
        self.minibatch_size = 128
        self.learnStart = 128
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 10000 # 1000000

        super(DeepQTrain, self).__init__(inputSize, outputSize, memorySize, discountFactor, learningRate, \
            self.learnStart, player_name)

        if self.modelExisted():
            self.loadModel()
        else:
            self.initNetworks([30, 30])
        self._reset()

    def _learnOnMinBatch(self):
        if self.stepCounter >= self.learnStart:
            if self.stepCounter <= self.updateTargetNetworkCounter:
                self.learnOnMiniBatch(self.minibatch_size, False)
                # print "[INFO] Learn Start <= UpdateTargetNetwork"
            else:
                self.learnOnMiniBatch(self.minibatch_size, True)
                # print "[INFO] Learn Start > UpdateTargetNetwork"

    def _updateTargetNetwork(self):
        if self.stepCounter % self.updateTargetNetworkCounter == 0:
            self.updateTargetNetwork()
            print "[INFO] Update Target Network"

    def _reset(self):
        self.observation = None
        self.newObservation = None
        self.action = None

    def addMemoryUDQN(self, data):
        # action, reward, state, done
        state, reward, done = data['state'], data['reward'], data['done']        
        
        if self.observation is None:
            self.observation = np.array(state)
        else:
            self.newObservation = np.array(state)
   
            self.addMemory(self.observation, self.action, reward, \
                self.newObservation, done)
            self.observation = self.newObservation

            self._learnOnMinBatch()
            self.stepCounter += 1
            self._updateTargetNetwork()

        self.action = data['action']
        
        if done:
            self._reset()

