import numpy as np
import time

from utils.dqn import DeepQTrain
from utils.montecargo import MonteCargo

from holdem import ACTION, action_table, card_to_normal_str


class CdqnModel:
    stateSize = 10  # [monteCarlo, playerCount, remainChips, investChips, pot, toCall, oneHotRound0, oneHotRound1, oneHotRound2, oneHotRound3]
    actionTrain = {0: 'FOLD', 1: 'CHECK', 2: 'RAISE*1', 3: 'RAISE*2', 4: 'RAISE*4', 5: 'RAISE*8', \
                   6: 'RAISE*16', 7: 'RAISE*32', 8: 'RAISE*48', 9: 'RAISE*64', 10: 'RAISE*100'}

    def __init__(self, model_name_prefix="test", deep_q=None):
        # self.reload_left = 2
        self.model = {"seed": 831}
        self.playerid = None
        self._initDqn(model_name_prefix, deep_q)

    def batchTrainModel(self):
        return

    def onlineTrainModel(self):
        return

    def saveModel(self, path):
        return

    def loadModel(self, path):
        return

    def endCycle(self, states):
        if states.player_states[self.playerid].playing_hand:
            observation = self._getObservation(states)
            stack = states.player_states[self.playerid].stack
            betting = states.player_states[self.playerid].betting
            reward = (stack - betting) / self.BB
            self._addMemory(observation, None, reward, True)

    def showAction(self, actions):
        pass

    def takeAction(self, state, playerid):
        ''' (Predict/ Policy) Select Action under state'''
        self.playerid = playerid
        actionID, amount = self._doAction(state)
        return ACTION(actionID, amount)

    def getReload(self, state):
        pass

    def _initDqn(self, model_name_prefix, deep_q):
        self.monteCarlo = MonteCargo()
        self.montecarloTimes = 1000

        self.winRate = 0
        self.playerCount = 0

        inputSize = self.stateSize
        outputSize = len(self.actionTrain)

        if deep_q is None:
            self.deepQ = DeepQTrain(
                inputSize=inputSize,
                outputSize=outputSize,
                model_name_prefix=model_name_prefix
            )
        else:
            self.deepQ = deep_q

        self._reset()

    def _reset(self):
        self.BB = None

    def _doAction(self, state):
        # temporarily skip: raiseCount, betCount, roundName, actionType
        self.BB = float(state.community_state.bigblind)

        observation = self._getObservation(state)
        qValues = self.deepQ.getQValues(observation)
        actionID = self.deepQ.selectAction(qValues)
        self._addMemory(observation, actionID, 0, False)

        actionName = self._transAction(state.community_state.to_call, actionID)
        if 'RAISE' in actionName:
            betTimes = int(actionName.split('*')[-1])
            if betTimes == 1:
                return action_table.CALL, state.community_state.to_call
            else:
                return action_table.RAISE, self.BB * betTimes
        else:
            return getattr(action_table, actionName), 0

    def _getObservation(self, state):
        # [monteCarlo, remainChips, investChips, pot]        
        cards = [card_to_normal_str(c).upper() for c in state.player_states[self.playerid].hand]
        boards = [card_to_normal_str(c).upper() for c in state.community_card if c != -1]

        self.playerCount = len([p for p in state.player_states if p.playing_hand])
        start = time.time()
        self.winRate = self.monteCarlo.calc_win_rate(cards, boards, self.playerCount, self.montecarloTimes)

        remainChips = state.player_states[self.playerid].stack / self.BB
        investChips = state.player_states[self.playerid].betting / self.BB
        toCallChips = state.community_state.to_call / self.BB
        round = self._toOneHotRound(state.community_state.round)
        pot = state.community_state.totalpot / self.BB

        state = [self.winRate, self.playerCount, remainChips, investChips, pot, toCallChips]
        state.extend(round)

        # print 'time_getObs', (time.time() - start), state, self.playerid
        return np.array(state)

    def _addMemory(self, state, actionID, reward, done):
        data = {'state': state, 'action': actionID, 'reward': reward, 'done': done}
        self.deepQ.addMemoryUDQN(data)

    def _transAction(self, minBet, actionID):
        actionTrain = self.actionTrain[actionID]
        actionName = actionTrain

        if 'FOLD' == actionTrain and minBet == 0:
            actionName = 'CHECK'

        # if actionTrain != actionName:
        #     print '***action:', actionTrain, '->', actionName, ', min:', minBet  # , ',cost:', cost
        return actionName

    def _toOneHotRound(self, round):
        if round == 0:
            return [1, 0, 0, 0]
        elif round == 1:
            return [0, 1, 0, 0]
        elif round == 2:
            return [0, 0, 1, 0]
        else:
            return [0, 0, 0, 1]

# observation, actionID, reward, newObservation, done
# observation: 
#   winRate: ['cards']['board'], ['cards']['hands']     
#   remainChips: ['players'][ID]['chips'], BB
#   investChips: ['players'][ID]['round_bet'], ['players'][ID]['bet'], BB
#   pot: ['bet']['pot'], BB
