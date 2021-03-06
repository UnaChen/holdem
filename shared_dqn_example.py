import gym
import holdem
import agent
import traceback
import sys
from utils.dqn import DeepQTrain


def lets_play(env, n_seats, model_list):
    try:
        while True:
            cur_state = env.new_cycle()
            # env.render(mode='human')
            cycle_terminal = False
            if env.episode_end:
                break

            while not cycle_terminal:
                actions = holdem.model_list_action(cur_state=cur_state, n_seats=n_seats, model_list=model_list)

                for m in model_list:
                    m.showAction(actions)

                cur_state, rews, cycle_terminal, info = env.step(actions)

            for m in model_list:
                m.endCycle(cur_state)

            # for s in cur_state.player_states:
            #     print( holdem.utils.hand_to_str(s.hand, "human"))
    except Exception as e:
        traceback.print_exc()
        raise

env = gym.make('TexasHoldem-v2')

model_list = list()

dqn_input_size = agent.CdqnModel.stateSize
dqn_output_size = len(agent.CdqnModel.actionTrain)
model_prefix_name = 'test'
shared_deep_q = DeepQTrain(dqn_input_size, dqn_output_size, model_prefix_name)

env.add_player(0, stack=3000)
model_list.append(agent.CdqnModel(model_prefix_name, shared_deep_q))

env.add_player(1, stack=3000)
model_list.append(agent.CdqnModel(model_prefix_name, shared_deep_q))

env.add_player(2, stack=3000)
model_list.append(agent.CdqnModel(model_prefix_name, shared_deep_q))

env.add_player(3, stack=3000)
model_list.append(agent.CdqnModel(model_prefix_name, shared_deep_q))

env.add_player(4, stack=3000)
model_list.append(agent.CdqnModel(model_prefix_name, shared_deep_q))

env.add_player(5, stack=3000)
model_list.append(agent.WinRateGambler())

env.add_player(6, stack=3000)
model_list.append(agent.WinRateGambler())

env.add_player(7, stack=3000)
model_list.append(agent.WinRateGambler())

env.add_player(8, stack=3000)
model_list.append(agent.WinRateGambler())

env.add_player(9, stack=3000)
model_list.append(agent.WinRateGambler())

# play out a hand
try:
    episode = 0
    while True:
        lets_play(env, env.n_seats, model_list)

        # raw_input("press for next episode...")
        print('episode ', episode)
        sys.stdout.flush()

        env.reset()
        episode += 1
except Exception, e:
    print(e)
