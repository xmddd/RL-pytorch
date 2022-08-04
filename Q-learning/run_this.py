from model import QLearningTable
import gym


def update(n_episode):
    for episode in range(n_episode):
        # initial state
        state = env.reset()
        total_reward = 0

        while True:
            # fresh env
            # env.render()

            # RL choose action based on state
            action = RL.choose_action(str(state))

            # RL take action and get next state and reward
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

            # RL learn from this transition
            RL.learn(str(state), action, reward, str(next_state), done)

            # swap state
            state = next_state

            # break while loop when end of this episode
            if done:
                break

        print("回合数：{}/{}，奖励{:.1f}".format(episode+1, n_episode, total_reward))
    # end of game
    print('game over')


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')
    # reproducible, general Policy gradient has high variance
    env.reset(seed=1)
    # env = env.unwrapped
    RL = QLearningTable(n_actions=env.action_space.n)
    # print(list(range(env.action_space.n)))

    update(n_episode=400)
