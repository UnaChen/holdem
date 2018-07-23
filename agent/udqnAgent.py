import json
import sys
import numpy as np
import time
import pandas as pd
import os


from core.abstract import StatisticModel
from core.dqn import DeepQ
from core.montecargo import MonteCargo


class UDqnModel(StatisticModel):
    def __init__(self, player_name):
        super(UDqnModel, self).__init__(player_name)
        self.monteCarlo = MonteCargo()
        self.montecarloTimes = 10000

        self.needReload = False
        self.winRate = 0
        self.playerCount = 0
        self.actionTrain = {0: 'fold', 1: 'check', 2:'bet*0.5', 3:'bet*1', 4:'bet*2', 5:'bet*4', \
            6:'bet*8', 7:'bet*16', 8:'bet*32', 9:'bet*48', 10:'bet*64', 11:'bet*100'}

        inputSize = 1 # [monteCarlo, remainChips, investChips, pot]
        outputSize = len(self.actionTrain)

        self.deepQ = DeepQ(inputSize, outputSize, 0, 0, 0, 0, player_name)
        self.deepQ.loadModel()

        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "preflop_win_rate.csv")
        self.preflop_win_rate = pd.read_csv(file_path, index_col=0)

        self._reset()

    def action(self, ws, data):
        start = time.time()
        send_data = self._doAction(data, 'ACTION')
        ws.send(json.dumps(send_data))
        print send_data
        print 'time_sendAction',(time.time() - start),'\n'

    def bet(self, ws, data):
        start = time.time()
        send_data = self._doAction(data, 'BET')
        ws.send(json.dumps(send_data))
        print send_data
        print 'time_sendAction',(time.time() - start),'\n'

    def start_reload(self, ws, data):
        if self.needReload:
            send_data = self._getSendDataInit()
            send_data["eventName"] = "__reload"
            ws.send(json.dumps(send_data))
            print send_data, '\n'
            self.needReload = False

    def new_round(self, ws, data):
        self._reset()
        self.BB = data['table']['bigBlind']['amount']

    def show_action(self, ws, data):        
        self_data = filter(lambda x: x['playerName'] == self.player_md5, data['players'])[0]
        remain = self_data['chips']
        remainBB = remain / self.BB 
        if remainBB < 5 and remain != 0:
            self.needReload = True


    def _getSendDataInit(self):
        send_data = {
            "eventName": "__action",
            "data": {} # action, amount
        }
        return send_data

    def _getObservation(self, data):
        # [monteCarlo, remainChips, investChips, pot]
        dataSelf = data['self']
        dataGame = data['game']
        cards, boards = dataSelf['cards'], dataGame['board']
        
        self.playerCount = len( filter(lambda p: p['isSurvive'] and not p['folded'] , \
            [pl for pl in dataGame['players']]))
        start = time.time()
        self.winRate = self.monteCarlo.calc_win_rate(cards, boards, self.playerCount, self.montecarloTimes)
        remainChips = dataSelf['chips'] / self.BB
        investChips = (dataSelf['roundBet'] + dataSelf['bet']) / self.BB
        state = [self.winRate] #, remainChips, investChips, pot]
        print 'time_getObs',(time.time() - start)
        print state
        return np.array(state)

    def _reset(self):
        self.BB = None

    def _transAction(self, remain, minBet, actionID, actionType, betCount, raiseCount):        
        actionTrain = self.actionTrain[actionID]
        actionName = actionTrain
        cost = minBet / float(remain) if remain != 0 else 0
        betBB = int(actionName.split('*')[-1])
        if remain == 0: return 'allin'

        if 'fold' == actionTrain and actionType == 'BET':
            actionName = 'check'
        elif 'check' == actionTrain and actionType == 'ACTION':
            actionName = 'call' if cost < 0.05 else 'fold'
        elif 'bet' in actionTrain:            
            if betBB < minBet:
                if betBB >= 32:
                    actionName = 'allin'
                elif betBB >= 16 and cost < 0.8:
                    actionName = 'call'
                else:
                    actionName = self._transFeeling(betBB, minBet, cost)                    
            else:
                if betCount < 4: # betBB == minBet -> call
                    actionName = actionTrain
                elif raiseCount < 4:
                    actionName = 'raise'
                else:
                    actionName = 'allin' if betBB >= 0.87* remain else 'call'
        
        if actionTrain != actionName:
            print '***action:', actionTrain, '->', actionName, ', min:', minBet, ',cost:', cost
        return actionName
   
    def _transFeeling(self, betBB, minBet, cost): 
        # self.winRate, self.playerCount
        actionName = 'call' if cost < 0.1 else 'fold' 
        allin_threshold = self.preflop_win_rate.quantile(axis=0, q=0.9)[self.playerCount - 2]
        call_threshold = self.preflop_win_rate.quantile(axis=0, q=0.6)[self.playerCount - 2]
        if self.roundName == 'Deal':
            # if self.winRate > 0.2 and betBB*10 > minBet and cost < 0.5:
            if self.winRate > allin_threshold:
                actionName = 'call'
            # elif self.winRate > 0.1 and betBB*5 > minBet and cost < 0.15:
            elif self.winRate > call_threshold and cost < 0.5:
                actionName = 'call'
        elif self.roundName != 'ShowDown':
            if self.winRate >= 0.1 and self.winRate < 0.2 and betBB*2 > minBet and cost < 0.2:
                actionName = 'call'
            elif self.winRate >= 0.2 and self.winRate < 0.4 and betBB*4 > minBet and cost < 0.3:
                actionName = 'call'
            elif self.winRate >= 0.4 and self.winRate < 0.6 and betBB*6 > minBet and cost < 0.4:
                actionName = 'call'
        elif self.roundName == 'ShowDown':                                                                                                                  
            if self.winRate >= 0.6 and cost < 0.6:
                actionName = 'call'
        
        return actionName


    def _doAction(self, data, actionType):
        minBet = data['self']['minBet'] / self.BB
        remain = data['self']['chips'] / self.BB
        raiseCount = data["game"]['raiseCount']
        betCount = data["game"]['betCount']
        self.roundName = data["game"]['roundName']

        observation = self._getObservation(data)
        qValues = self.deepQ.getQValues(observation)
        actionID = self.deepQ.selectAction(qValues, 0) # 0 = explorationRate
        actionName = self._transAction(remain, minBet, actionID, actionType, \
            betCount, raiseCount)

        send_data = self._getSendDataInit()
        send_data["data"]["action"] = actionName if 'bet' not in actionName else 'bet'
        if 'bet' in actionName:
            send_data["data"]["amount"] = self.BB * int(actionName.split('*')[-1])
        return send_data

# observation, actionID, reward, newObservation, done
# observation: 
#   winRate: ['cards']['board'], ['cards']['hands']     
#   remainChips: ['players'][ID]['chips'], BB
#   investChips: ['players'][ID]['round_bet'], ['players'][ID]['bet'], BB
#   pot: ['bet']['pot'], BB
