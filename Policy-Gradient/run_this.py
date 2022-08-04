from matplotlib import spines
from model import PG
import gym


def update(n_episode):
    for episode in range(n_episode):
        # initial state
        state = env.reset()
        total_reward = 0

        RL.trajectory = []

        while True:
            # fresh env
            # if episode> 300:
            #     env.render()

            # RL choose action based on state
            action, pr = RL.choose_action(state)

            # RL take action and get next state and reward
            next_state, reward, done, _ = env.step(action)

            RL.trajectory.append(
                {"state": state, "action": action, "pr": pr, "reward": reward})

            total_reward += reward

            # swap state
            state = next_state

            # break while loop when end of this episode
            if done:
                break

        # learn after the end of each episode
        RL.learn()
        print("回合数：{}/{}，奖励{:.1f}".format(episode+1, n_episode, total_reward))
    # end of game
    print('game over')


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # reproducible, general Policy gradient has high variance
    env.reset(seed=1)
    # env = env.unwrapped
    print(env.observation_space.shape[0])
    RL = PG(
        n_states=env.observation_space.shape[0], n_actions=env.action_space.n)

    update(n_episode=400)
