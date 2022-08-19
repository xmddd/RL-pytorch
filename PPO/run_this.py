from model import PPO
import gym
import torch

def update(n_episode, update_step=128):
    for episode in range(n_episode):
        # initial state
        state = env.reset()
        total_reward = 0
        time_step = 0

        while True:
            # fresh env
            # if episode> 100:
            #     env.render()

            # PPO choose action based on state
            action, log_pr = PPO.choose_action_old(state)
            # print(action, log_pr)
            # PPO take action and get next state and reward
            next_state, reward, done, _ = env.step(action)

            PPO.memory.push(state, action, reward, log_pr)

            total_reward += reward
            time_step += 1

            # swap state
            state = next_state

            # learn
            if (time_step + 1) % update_step == 0 or done:
                PPO.learn(next_state, done)

            # break while loop when end of this episode
            if done:
                break

        print("回合数：{}/{}，奖励{:.1f}".format(episode+1, n_episode, total_reward))
    # end of game
    print('game over')


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    # reproducible, general Policy gradient has high variance
    env.reset(seed=1)
    torch.manual_seed(1)
    # env = env.unwrapped
    PPO = PPO(
        n_states=env.observation_space.shape[0], n_actions=env.action_space.n)
    # print(list(range(env.action_space.n)))

    update(n_episode=400)
