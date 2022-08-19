from model import DDPG
import gym
import numpy as np

def update(n_episode):
    debug = True

    for episode in range(n_episode):
        # initial state
        state = env.reset()
        total_reward = 0

        while True:

            if episode > 380:
                env.render()

            # DDPG choose action based on state
            action = DDPG.choose_action(state)
            if action[0] < env.action_space.low[0] or action[0] > env.action_space.high[0]:
                raise ValueError("unacceptable action", action[0])

            # DDPG take action and get next state and reward
            next_state, reward, done, _ = env.step(action)

            # debug
            if debug:
                print("state={}, action={}, reward={}, next_state={}, done={}".format(
                    state, action, reward, next_state, done))
                debug = False

            total_reward += reward.item()

            # store into memory
            DDPG.memory.push(state, action, reward, next_state, done)

            # swap state
            state = next_state

            # DDPG learn from the samples in memory
            DDPG.learn()

            # break while loop when end of this episode
            if done:
                break

        # return
        print("回合数：{}/{}，奖励{:.1f}".format(episode+1, n_episode, total_reward))
    # end of game
    print('game over')

class NormalizedActions(gym.ActionWrapper):
    ''' 将action范围重定在[0.1]之间
    '''
    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

if __name__ == "__main__":
    env = NormalizedActions(gym.make('Pendulum-v1'))
    # reproducible, general Policy gradient has high variance
    env.reset(seed=1)
    # env = env.unwrapped
    print(env.action_space.low, env.action_space.high)
    print(env.observation_space.shape[0], env.action_space.shape[0])
    DDPG = DDPG(n_states=env.observation_space.shape[0],
                n_actions=env.action_space.shape[0], capacity=8000)

    update(n_episode=400)
