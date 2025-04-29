import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from approxQLearningAgent import *
from lunarLanderEnv import *
from cartPoleEnv import *
from util import *


def fitRbfTransform(env, num_samples, gamma, n_components, seed=42):
    """
    Fit a RBF transformer to the state space of the environment.
    """
    # store states for RBF transformation
    states = []

    # reset the environment
    state, _ = env.reset()

    # get a set number of samples from the environment
    for _ in range(num_samples):
        # get a random action
        action = env.action_space.sample()
        # take the random action in the environment
        state, _, done, truncated, _ = env.step(action)
        # store the state
        states.append(state)
        # reset the environment once a terminal or truncated state is reached
        if done or truncated:
            state, _ = env.reset()

    states = np.array(states)
    # fit the scaler
    scaler = StandardScaler().fit(states)
    # fit the RBF from the sampled states
    rbf = RBFSampler(gamma=gamma, 
                     n_components=n_components,
                    random_state=seed).fit(scaler.transform(states))
    
    return scaler, rbf


def calculate_epsilon_decay(epsilon, epsilon_min, decay_fraction, num_episodes):
    """
    Calculates the epsilon decay rate needed to reach the minimum epsilon value
    passed in over a given percentage (fraction) of the total number of training episodes.
    """
    # get number of episodes to decay over
    decay_episodes  = int(num_episodes * decay_fraction)

    # calculate the decay rate
    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / decay_episodes)

    return epsilon_decay


def trainAgent(agent, env, epsilon_decay, epsilon_min, num_episodes, verbose=False):
    """
    Train the agent in the environment.
    """
    # store rewards for reporting
    training_rewards = []

    # iterate for required number of episodes
    for episode in tqdm(range(num_episodes), desc="Training the RL agent"):
        # count for verbose logging
        count = 0

        # reset the env and get initial state
        state, _ = env.reset()

        # reset the terminal state flags
        agent.isTerminal = False
        terminated = False

        # track the total reward for this episode
        total_reward = 0

        # run the episode
        while not terminated:
            # get the action to take in this state
            action = agent.getAction(state)

            # take the action
            next_state, reward, done, truncated, _ = env.step(action)

            # check if episode is in terminal state or timed out
            if done or truncated:
                agent.isTerminal = True
                terminated = True

            # update the weights
            agent.update(state, action, next_state, reward, done)

            # update the total reward
            total_reward += reward

            # set state to next state
            state = next_state

            # FOR DEBUGGING: log the weights to ensure the gradient isn't exploding or vanishing
            if verbose:
                weights = np.linalg.norm(list(agent.weights.values()))
                if count % 100 == 0:
                    print(f"weights on iteration {count}::: \n{weights}\n")
            
            # increment episode count 
            count += 1
                
        
        # decay epsilon after each episode
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        # keep a record of the total reward for this episode
        training_rewards.append(total_reward)

        if verbose:
            # report the current episode number and total reward
            print(f"Episode {episode} done. Total reward::: {total_reward}")
    
    return training_rewards


def saveAgent(agent, filepath):
    """
    Stores the agents learned weights.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(agent.weights, f)


def runningMean(data, window=100):
    """
    Compute the running average reward over 100 episodes.
    """
    # return empty array if we don't have a full window worth of episodes
    if len(data) < window:
        return np.array([])
    # calculate the cumulative sum of rewards up to the current point
    cumsum = np.cumsum(np.insert(data, 0, 0))
    # return the average cumulative sum overt the size of the window
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plotRewards(rewards, training=True):
    """
    Plot the rewards across all episodes.
    """
    if training:
        title = 'Reward per Training Episode'
    else:
        title = 'Reward per Test Episode'
    
    plt.figure(figsize=(10, 6))

    # plot the rewards
    plt.plot(rewards)

    # plot the 100-episode running average
    running_avg = runningMean(rewards, window=100)
    if running_avg.size > 0:
        plt.plot(np.arange(99, len(rewards)),
                running_avg,
                label="100-episode running avg",
                linewidth=2)

    # add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()




def runCartPole(filepath, verbose=False):
    """
    Entry point to train and run the Epsilon Greedy Approximate Q-learning Agent
    in the OpenAI gymnasium Cart Pole environment.
    """
    # manually set number of training/test episodes and render mode
    NUM_TRAINING_EPISODES = 3_000
    RENDER_MODE = 'rgb_array' # 'human' or 'rgb_array'

    # manually set alpha, epsilon, and gamma values
    ALPHA = 0.0023
    EPSILON = 1.0
    GAMMA = 0.99

    # manually set fraction of episodes to decay over
    EPSILON_MIN = 0.04
    DECAY_FRACTION = 0.16

    # create the Cart Pole environment
    env = gym.make('CartPole-v1', render_mode=RENDER_MODE)

    # build the RBF transform
    scaler, rbf = fitRbfTransform(env=env,
                                  num_samples=10_000,
                                  gamma=1.0,
                                  n_components=500)
    

    # UNCOMMENT desired feature extractor
    feat_extractor = RBFExtractor(scaler, rbf)
    # feat_extractor = CPFeatureExtractor()

    # sanity check: ensure feature vector size is same as number of RBF components
    if verbose:
        test_state, _ = env.reset(seed=42)
        test_features = feat_extractor.getFeatures(np.array(test_state), 0)
        print(f"feature vector elements::: {len(test_features)}")
        print(f"sample entries: {list(test_features.items())[:4]}")


    # Cart Pole action space
        # 0: push cart to left
        # 1: push cart to right
    actions = [0, 1]

    # instantiate the agent
    agent = ApproxQAgent(feat_extractor=feat_extractor,
                         actions=actions,
                         alpha=ALPHA,
                         epsilon=EPSILON,
                         gamma=GAMMA)
    
    # get the epsilon decay rate based on target % of episodes to decay over
    epsilon_decay = calculate_epsilon_decay(epsilon=EPSILON,
                                            epsilon_min=EPSILON_MIN,
                                            decay_fraction=DECAY_FRACTION,
                                            num_episodes=NUM_TRAINING_EPISODES)
    
    # train the agent in the environment
    training_rewards = trainAgent(agent,
                                  env,
                                  epsilon_decay=epsilon_decay,
                                  epsilon_min=EPSILON_MIN,
                                  num_episodes=NUM_TRAINING_EPISODES,
                                  verbose=False)
    
    # plot the training rewards for each episode
    plotRewards(training_rewards)

    # save the agent weights
    saveAgent(agent, filepath)




def runLunarLander(filepath):
    """
    Entry point to train and run the Epsilon Greedy Approximate Q-learning Agent
    in the OpenAI gymnasium Lunar Lander environment.
    """
    # manually set number of training/test episodes and render mode
    NUM_TRAINING_EPISODES = 1_000
    RENDER_MODE = 'rgb_array' # 'human' or 'rgb_array'

    # manually set alpha, epsilon, and gamma values
    ALPHA = 0.001
    EPSILON = 1.0
    GAMMA = 0.99

    # manually set fraction of episodes to decay over
    EPSILON_MIN = 0.01
    DECAY_FRACTION = 0.12

    # create the Lunar Lander environment
    env = gym.make('LunarLander-v3', render_mode=RENDER_MODE)

    # build the RBF transform
    scaler, rbf = fitRbfTransform(env=env,
                                  num_samples=10_000,
                                  gamma=1.0,
                                  n_components=500)
    

    # UNCOMMENT desired feature extractor
    # feat_extractor = RBFExtractor(scaler, rbf)
    feat_extractor = LLFeatureExtractor()


    # Lunar Lander action space
        # 0: do nothing
        # 1: fire left engine
        # 2: fire main engine
        # 3: fire right engine
    actions = [0, 1, 2, 3]

    # instantiate the agent
    agent = ApproxQAgent(feat_extractor=feat_extractor,
                         actions=actions,
                         alpha=ALPHA,
                         epsilon=EPSILON,
                         gamma=GAMMA)
    
    # get the epsilon decay rate based on target % of episodes to decay over
    epsilon_decay = calculate_epsilon_decay(epsilon=EPSILON,
                                            epsilon_min=EPSILON_MIN,
                                            decay_fraction=DECAY_FRACTION,
                                            num_episodes=NUM_TRAINING_EPISODES)

    # train the agent in the environment
    training_rewards = trainAgent(agent,
                                  env,
                                  epsilon_decay=epsilon_decay,
                                  epsilon_min=EPSILON_MIN,
                                  num_episodes=NUM_TRAINING_EPISODES,
                                  verbose=False)
    
    # plot the training rewards for each episode
    plotRewards(training_rewards)

    # save the agent weights
    saveAgent(agent, filepath)




def main():
    AGENT_NAME = "lunarLander_1000_a=0.001_em=0.01_df=0.12"
    filepath = f"../trainedAgents/{AGENT_NAME}.pk1"

    # Uncomment desired run option: runLunarLander() or runCartPole()

    # runCartPole(filepath=filepath)
    runLunarLander(filepath=filepath)


if __name__ == "__main__":
    main()