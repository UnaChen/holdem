#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
from websocket import create_connection
import sys
import os
import hashlib
from .utils import hand_to_str, format_action, PLAYER_STATE, COMMUNITY_STATE, STATE, ACTION, action_table, card_str_to_list
from .player import Player
from treys import Card, Deck, Evaluator

ACCEPTED_ACTION = ["bet", "call", "raise", "check", "fold", "allin"]


def listen_to_state():
    return

def packed_action():
    return


class ClientPlayer():
    def __init__(self, server_uri, name, model, debug=False):
        self._server_uri = server_uri
        self._debug = debug
        self.ws = ""
        self._name = str(hashlib.md5(name.encode('utf-8')).hexdigest())
        self._nick_name = name
        self._model = model
        self._room_id = "217" # tableNumber
        self.cycle = 0

        # community_information
        self._round = 0
        self._button = 0
        self._smallblind = 0
        self._bigblind = 0

        self._tableNumber = ""
        self._player_name_list = list()

        self._cycle = 0
        self._evaluator = Evaluator()

        self.community = []
        self._discard = [] # Not used here, can not get the informatiom

        self._current_sidepot = 0  # index of _side_pots
        self._totalpot = 0
        self._tocall = 0
        self._lastraise = 0
        self._last_player = None
        self._last_actions = None

        #  player_information
        self.n_seats = 10
        self.emptyseats = self.n_seats

        self._side_pots = [0] * self.n_seats
        self._seats = self._seats = [Player(i, stack=0, emptyplayer=True) for i in range(self.n_seats)]
        self._player_dict = {}
        self._current_player = None

    def _add_player(self, seat_id, init_stack, init_name, reloadCount):
        player_id = seat_id
        if player_id not in self._player_dict:
            new_player = Player(player_id, stack=init_stack, emptyplayer=False, playername=init_name, reloadCount=reloadCount)
            if self._seats[player_id].emptyplayer:
                self._seats[player_id] = new_player
                new_player.set_seat(player_id)
            else:
                raise KeyError('Seat already taken.')
            self._player_dict[player_id] = new_player
            self.emptyseats -= 1

    def __getActiveSeats(self):
        return self.n_seats - self.emptyseats

    def __getPlayerSeatByName(self, name):
        for i, p in enumerate(self._seats):
            if name == p.get_name():
                return i
        return -1

    def get_current_state(self):
        player_states = []

        my_seat = self.__getPlayerSeatByName(self._name)
        for player in self._seats:
            player_features = PLAYER_STATE(
                int(player.emptyplayer),
                int(player.get_seat()),
                int(player.stack),
                int(player.playing_hand),
                int(player.handrank),
                int(player.playedthisround),
                int(player.betting),
                int(player.isallin),
                int(player.lastsidepot),
                0,
                self._pad(player.hand, 2, -1)
            )
            player_states.append(player_features)

        community_states = COMMUNITY_STATE(
            int(self._button),
            int(self._smallblind),
            int(self._bigblind),
            int(self._totalpot),
            int(self._lastraise),
            int(max(self._bigblind, self._lastraise + self._tocall)),
            int(self._tocall - self._player_dict[my_seat].currentbet),
            int(my_seat)
        )
        return STATE(tuple(player_states), community_states, self._pad(self.community, 5, -1))

    def render(self, mode='human', close=False):
        print('Cycle {}, total pot: {}'.format(self._cycle, self._totalpot))
        if self._last_actions is not None:
            pid = self._last_player.player_id
            print('last action by player {}:'.format(pid) + '\t' + format_action(self._last_player,
                                                                                 self._last_actions[pid]))

        state = self._get_current_state()

        # (player_infos, player_hands) = zip(*state.player_state)

        print('community:')
        print('-' + hand_to_str(state.community_card))
        print('players:')
        for idx, playerstate in enumerate(state.player_states):
            print('{}{}stack: {}'.format(idx, hand_to_str(playerstate.hand), self._seats[idx].stack))
        # for idx, hand in enumerate(player_hands):
        #  print('{}{}stack: {}'.format(idx, hand_to_str(hand), self._seats[idx].stack))

    def _reset(self):
        self.cycle += 1

    def __roundNameToRound(self, rounName):
        if "Deal":
            return 0
        elif "Flop":
            return 1
        elif "Turn":
            return 2
        elif "River":
            return 3

    def _new_round(self):
        for player in self._player_dict.values():
            player.currentbet = 0
            player.playedthisround = False
        self._round += 1
        self._tocall = 0
        self._lastraise = 0

    def _pad(self, l, n, v):
        if (not l) or (l is None):
            l = []
        return l + [v] * (n - len(l))

    def _send_get_reload(self):
        self.ws.send(json.dumps({
            "eventName": "__reload"
        }))
        return

    def _send_action(self, model_action):
        ''' Not goinf to check whether the action avalble'''
        if model_action.action == action_table.CHECK:
            self.ws.send(json.dumps({
                "eventName": "__action",
                "data": {
                    "action" : "check"
                }
            }))
        elif model_action.action == action_table.RAISE:
            self.ws.send(json.dumps({
                "eventName": "__action",
                "data": {
                    "action": "bet",
                    "amount": int(model_action.amount)
                }
            }))
        elif model_action.action == action_table.FOLD:
            self.ws.send(json.dumps({
                "eventName": "__action",
                "data": {
                    "action": "FOLD"
                }
            }))
        elif model_action.action == action_table.CALL:
            self.ws.send(json.dumps({
                "eventName": "__action",
                "data": {
                    "action": "call"
                }
            }))
        return

    def _handle_event(self, msg, data):
        if self._debug:
            print("Reveice Message:  [{}]".format(msg))

        if msg[:len("__new_peer")] == "__new_peer":
            return False # not interesting
        elif msg == "__join":
            return False # TODO
        elif msg == "__game_prepare":
            return False # not interesting
        elif msg == "__new_round":
            # No avtion, update state
            self._tocall = 0
            self._lastraise = 0

            self._round = self.__roundNameToRound(data["table"]["roundName"])
            self._tableNumber = data["table"]["tableNumber"]

            smallblind_name = data["table"]["smallBlind"]["playerName"]
            smallblind_id = -1
            bigblind_name = data["table"]["bigBlind"]["playerName"]
            bigblind_id = -1

            for i, p in enumerate(data["players"]):
                # Add New Player
                self._add_player(i, p["chips"], p["playerName"], p["reloadCount"])
                if p["playerName"] == bigblind_name:
                    bigblind_id = i
                if p["playerName"] == smallblind_name:
                    smallblind_id = i

            # get button id
            if smallblind_id + 1 == self.__getActiveSeats() and bigblind_id == 0:
                button_id = smallblind_id - 1
            elif smallblind_id == 0 and bigblind_id == 1:
                button_id = self.__getActiveSeats() - 1
            elif smallblind_id + 1 == bigblind_id:
                button_id = smallblind_id - 1
            else:
                button_id = -1
            if bigblind_id == self.__getActiveSeats() - 1:
                current_player_seat = 0
            else:
                current_player_seat = bigblind_id + 1

            self.button = button_id
            self._smallblind = int(data["table"]["smallBlind"]["amount"])
            self._bigblind = int(data["table"]["bigBlind"]["amount"])
            self._tocall = int(data["table"]["bigBlind"]["amount"])
            self._current_player = current_player_seat

            return False

        elif msg == "__start_reload": # Might Action
            for p in data["players"]:
                i = self.__getPlayerSeatByName(p["playerName"])
                player_info = self._player_dict[i]
                player_info.stack = p["chips"]
                player_info.playing_hand = not p["folded"]
                player_info.isallin=p["allIn"]
                player_info.sitting_out=not p["isSurvive"]
                player_info.reloadCount=p["reloadCount"]

            self._tableNumber = data["tableNumber"]
            if self._model.getReload(self.get_current_state()):
                self._send_get_reload()
            return False # not interesting

        elif msg == "__deal": # No Action
            # Update player_states
            self._tocall = 0
            self._lastraise = 0

            for p in data["players"]:
                i = self.__getPlayerSeatByName(p["playerName"])
                player_info = self._player_dict[i]
                player_info.playedthisround = False
                player_info.stack = p["chips"]
                player_info.playing_hand = not p["folded"]
                player_info.isallin=p["allIn"]
                player_info.sitting_out=not p["isSurvive"]
                player_info.reloadCount=p["reloadCount"]
                player_info.betting = p["bet"]
                player_info.currentbet = 0 # p["roundBet"]
                try:
                    if len(p["cards"]) == 2:
                        p_card = [card_str_to_list(p["cards"][0]),card_str_to_list(p["cards"][1])]
                    else:
                        p_card = [-1, -1]
                except KeyError:
                    p_card = [-1, -1]
                    pass
                player_info.hand = p_card

            # Update Community_state
            self._round = self.__roundNameToRound(data["table"]["roundName"])
            self._tableNumber = data["table"]["tableNumber"]

            # Not going to check name twice
            self._smallblind = int(data["table"]["smallBlind"]["amount"])
            self._bigblind = int(data["table"]["bigBlind"]["amount"])

            c_cards = list()
            for c in data["table"]["board"]:
                c_cards.append(card_str_to_list(c))
            self.community = c_cards
            # drop data["table"]["roundCount"]
            # drop data["table"]["raiseCount"]
            # drop data["table"]["betCount"]

            return False # not interesting

        elif msg == "__action" or msg == "__bet": # Must Action
            self._tableNumber = data["tableNumber"]
            my_seat = self.__getPlayerSeatByName(self._name)
            if my_seat == -1:
                print(self._name)
                print(self._player_dict)
            # Update player_states Information
            if len(data["self"]["cards"]) == 2:
                p_card = [card_str_to_list(data["self"]["cards"][0]), card_str_to_list(data["self"]["cards"][1])]
            else:
                print("API Failed in {}".format(msg))

            player_info = self._player_dict[my_seat]
            player_info.playedthisround = True
            player_info.stack = data["self"]["chips"]
            player_info.playing_hand = not data["self"]["folded"]
            player_info.isallin=data["self"]["allIn"]
            player_info.sitting_out=data["self"]["isSurvive"]

            self_bet = data["self"]["bet"] # money already in pot
            self_minbet = data["self"]["minBet"] # minimun to call

            for p in data["game"]["players"]:
                i = self.__getPlayerSeatByName(p["playerName"])
                if i != my_seat:
                    player_info = self._player_dict[i]
                    player_info.stack = p["chips"]
                    player_info.playing_hand = not p["folded"]
                    player_info.isallin = p["allIn"]
                    player_info.sitting_out = not p["isSurvive"]
                    player_info.reloadCount = p["reloadCount"]
                    player_info.betting = p["bet"]

            # Update community_state
            self._tocall = self_minbet
            self._current_player = my_seat
            self._round = self.__roundNameToRound(data["game"]["roundName"])
            self._smallblind = int(data["game"]["smallBlind"]["amount"])
            self._bigblind = int(data["game"]["bigBlind"]["amount"])
            #  (drop) data["table"]["raiseCount"]
            #  (drop) data["table"]["betCount"]
            c_cards = list()
            for c in data["game"]["board"]:
                c_cards.append(card_str_to_list(c))
            self.community = c_cards

            # Action by model
            model_action = self._model.takeAction(self.get_current_state(), my_seat)
            self._send_action(model_action)

            return False # not interesting

        elif msg == "__show_action": # no Action
            actioned_id = self.__getPlayerSeatByName(data["action"]["playerName"])
            cur_round = self.__roundNameToRound(data["table"]["roundName"])
            if self._debug:
                print("seat", actioned_id, "do some action ")
            player_info = self._player_dict[actioned_id]

            player_info.playedthisround = True
            player_info.stack = data["action"]["chips"]
            if data["action"]["action"] == "bet" or data["action"]["action"] == "raise" or data["action"]["action"] == "allin":
                self._lastraise = data["action"]["amount"]
            self._totalpot = data["table"]["totalBet"]
            return False # not interesting

        elif msg == "__round_end":
            # no reply
            if self._debug:
                print(data)
            self._reset()
            return False # not interesting
        elif msg == "__game_over":
            if self._debug:
                print(data)
            return True # not interesting
        else:
            print("Error, Unknown event message [{}]".format(msg))
            return False

    def doListen(self):
        try:
            self.ws = create_connection(self._server_uri)
            self.ws.send(json.dumps({
                "eventName": "__join",
                "data" : {
                    "playerName": self._nick_name
                }
            }))
            if self._debug:
                print("Joined {}, hash: {}".format(self._nick_name, self._name))
            while True:
                print("keep waiting")
                result = self.ws.recv()
                print("recv {} [{}]".format(type(result), result))
                msg = json.loads(result)
                terminal = self._handle_event(msg["eventName"], msg["data"])
                if terminal:
                    break
            print("Game Over")
        #except Exception as e:
        except IOError as e:
            print("Failed to doListern(): ", str(e))
            self.doListen()