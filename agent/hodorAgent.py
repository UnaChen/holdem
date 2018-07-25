import random
import pandas as pd
import os

from utils.montecargo import MonteCargo
from holdem import card_to_normal_str, ACTION, action_table

# roundName = ['Deal', 'Flop', 'Turn', 'River', 'ShowDown']
# bet_actions = ['bet', 'check', 'fold', 'allin']
# action_actions = ['bet', 'check', 'fold', 'allin', 'raise', 'call']
# action_actions: check=call

class WinRateGambler():
    def __init__(self):
        self.card_rank = "23456789TJQKA"
        file_path = os.path.join(os.path.dirname(__file__), "..", "data", "preflop_win_rate.csv")
        self.preflop_win_rate = pd.read_csv(file_path, index_col=0)
        self.mcg = MonteCargo()

    def endCycle(self, states):
        pass

    def showAction(self, actions):
        pass

    def getReload(self, state):
        pass

    def takeAction(self, state, playerid):
        round = state.community_state.round # 0-3
        hands = [card_to_normal_str(c).upper() for c in state.player_states[playerid].hand]
        table_cards = [ card_to_normal_str(c).upper() for c in state.community_card if c != -1]
        min_bet = state.community_state.to_call
        chips = state.player_states[playerid].stack
        bigblind = state.community_state.bigblind
        betting = state.player_states[playerid].betting

        num_in_game_player = len([p for p in state.player_states if p.playing_hand])

        allin_threshold = self.preflop_win_rate.quantile(axis=0, q=0.9)[num_in_game_player - 2]
        call_threshold = self.preflop_win_rate.quantile(axis=0, q=0.6)[num_in_game_player - 2]

        raise_action_amount = min_bet * 2
        allin_action_amount = chips
        call_action_amount = min_bet

        action_amount = 0
        if round == 0:
            win_rate = self._get_preflop_win_rate(hands, num_in_game_player)
            if win_rate >= allin_threshold:
                if min_bet * 4 > chips:
                    action_amount = allin_action_amount
                else:
                    action_amount = max(min_bet, 0.4 * chips * random.random())
            elif win_rate >= call_threshold:
                action_amount = call_action_amount
            elif win_rate >= call_threshold and chips < 3 * bigblind:
                action_amount = allin_action_amount
            elif min_bet <= 0.05 * chips:
                action_amount = call_action_amount
        else:
            win_rate = self.mcg.calc_win_rate(hands, table_cards, num_in_game_player, 1000)
            if win_rate >= 0.9:
                action_amount = allin_action_amount
            elif win_rate >= 0.75:
                if betting > 0.3 * chips:
                    action_amount = call_action_amount
                else:
                    action_amount = max(min_bet, 0.3 * chips * random.random())
            elif win_rate >= 0.6:
                if betting > 0.1 * chips:
                    action_amount = call_action_amount
                else:
                    action_amount = max(min_bet, 0.1 * chips * random.random())

        if action_amount >= allin_action_amount:
            action = ACTION(action_table.RAISE, chips)
        elif abs(action_amount - raise_action_amount) <= 0.5 * bigblind:
            action = ACTION(action_table.RAISE, raise_action_amount)
        elif abs(action_amount - call_action_amount) <= 0.5 * bigblind:
            action = ACTION(action_table.CALL, min_bet)
        elif action_amount >= call_action_amount:
            action = ACTION(action_table.RAISE, action_amount)
        else:
            action = ACTION(action_table.FOLD, action_amount)

        return action

    def _get_preflop_win_rate(self, hand_cards, player_amount):
        number1, suit1 = hand_cards[0]
        number2, suit2 = hand_cards[1]
        rank1 = self.card_rank.index(number1)
        rank2 = self.card_rank.index(number2)
        if number1 == number2:
            avg_win = self.preflop_win_rate[str(player_amount)][number1 + number2]
        else:
            shape = "s" if suit1 == suit2 else "o"
            if rank1 > rank2:
                avg_win = self.preflop_win_rate[str(player_amount)][number1 + number2 + shape]
            else:
                avg_win = self.preflop_win_rate[str(player_amount)][number2 + number1 + shape]
        return avg_win
