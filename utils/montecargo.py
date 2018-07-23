from subprocess import check_output

class MonteCargo(object):

    def calc_win_rate(self, hand_cards, table_cards, player_amount, max_run):
        # print ','.join(hand_cards)
        if len(table_cards) == 0:
            out = check_output(["./utils/poker", "-worker=10", "-round=%d" % max_run, "-hand=%s" % (','.join(hand_cards)), "-player=%d" % player_amount])
        else:
            out = check_output(["./utils/poker", "-worker=10", "-round=%d" % max_run, "-hand=%s" % (','.join(hand_cards)),"-board=%s" %(','.join(table_cards)), "-player=%d" % player_amount])
        # print out
        return float(out)