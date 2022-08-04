from model import DQN
import gym

def update(n_episode, update_frequency):
    debug = True

    for episode in range(n_episode):
        # initial state
        state = env.reset()
        total_reward = 0

        while True:
            # fresh env
            # env.render()

            # DQN choose action based on state
            action = DQN.choose_action(state)

            # DQN take action and get next state and reward
            next_state, reward, done, _ = env.step(action)

            # debug
            if debug:
                print("state={}, action={}, reward={}, next_state={}, done={}".format(
                    state, action, reward, next_state, done))
                debug = False

            total_reward += reward

            # store into memory
            DQN.memory.push(state, action, reward, next_state, done)

            # swap state
            state = next_state

            # DQN learn from the samples in memory
            DQN.learn()

            # break while loop when end of this episode
            if done:
                break
        
        if (episode + 1) % update_frequency == 0:
            DQN.target_net.load_state_dict(DQN.policy_net.state_dict())

        print("回合数：{}/{}，奖励{:.1f}".format(episode+1, n_episode, total_reward))
    # end of game
    print('game over')


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # reproducible, general Policy gradient has high variance
    env.reset(seed=1)
    # env = env.unwrapped
    DQN = DQN(n_states=env.observation_space.shape[0],
              n_actions=env.action_space.n, capacity=100000)

    # print(list(range(env.action_space.n)))

    update(n_episode=200, update_frequency = 4)
