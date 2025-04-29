import gymnasium as gym

def random_run(num_episodes=5, verbose=True):
    env = gym.make('LunarLander-v3', render_mode='human')
    for episode in range(num_episodes):
        state, _ = env.reset()
        random_action_example = env.action_space.sample()
        if verbose:
            print(f"state::: \n{state}\n")
            print(f"action space::: {env.action_space}\n")
            print(f"random action example::: {random_action_example}\n")
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
        if verbose:
            print(f"Episode {episode} finished. Total reward::: {total_reward}\n")
    env.close()

if __name__ == "__main__":
    random_run()