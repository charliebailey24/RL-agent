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


def loadAgent(agent, filepath):
    """
    Load previously saved weights into new agent.
    """
    with open(filepath, 'rb') as f:
        agent.weights = pickle.load(f)


def testAgent(agent, env, num_episodes, verbose=False):
    """
    Test a trained agent in a new environment.
    """
    # store the test rewards for reporting
    test_rewards = []

    # iterate for required number of episodes
    for episode in range(num_episodes):
        # reset the env and get initial state
        state, _ = env.reset()

        # reset the terminal state flags
        terminated = False

        # track the total reward of this episode
        total_reward = 0

        # run the episode
        while not terminated:
            # get the action to take in this state
            action = agent.getAction(state)

            # take the action
            next_state, reward, done, truncated, _ = env.step(action)

            # check if this episode is in a terminal state or timed out
            if done or truncated:
                terminated = True
            
            # update total reward
            total_reward += reward

            # set state to next state
            state = next_state
    
        # keep a record of the total reward for this episode
        test_rewards.append(total_reward)

        if verbose:
            # report the current episode number and total reward
            print(f"Episode {episode} done. Total reward::: {total_reward}")




def testCartPole(filepath, num_episodes=5):
    """
    Test the previously trained agent in the Cart Pole environment.
    """
    # create new environment to test the agent
    test_env = gym.make('CartPole-v1', render_mode='human')

    # UNCOMMENT desired feature extractor
    # feat_extractor = RBFExtractor(scaler, rbf)
    feat_extractor = CPFeatureExtractor()

    # Cart Pole action space
        # 0: push cart to left
        # 1: push cart to right
    actions = [0, 1]

    # create new test agent
    test_agent = ApproxQAgent(feat_extractor=feat_extractor,
                              actions = actions,
                              alpha = 0.0,
                              epsilon = 0.0,
                              gamma=0.99)
    
    # load the previously saved weights into the agent
    loadAgent(test_agent, filepath)

    # test the agent in the new environment
    testAgent(test_agent, test_env, num_episodes=num_episodes)




def testLunarLander(filepath, num_episodes=5):
    """
    Test the previously trained agent in the Lunar Lander environment.
    """
    # create new environment to test the agent
    test_env = gym.make('LunarLander-v3', render_mode='human')

    # create the feature extractor
    feat_extractor = LLFeatureExtractor()

    # Lunar Lander action space
        # 0: do nothing
        # 1: fire left engine
        # 2: fire main engine
        # 3: fire right engine
    actions = [0, 1, 2, 3]

    # create new test agent
    test_agent = ApproxQAgent(feat_extractor=feat_extractor,
                              actions = actions,
                              alpha = 0.0,
                              epsilon = 0.0,
                              gamma=0.99)
    
    # load the previously saved weights into the agent
    loadAgent(test_agent, filepath)

    # test the agent in the new environment
    testAgent(test_agent, test_env, num_episodes=num_episodes)


def main():
    AGENT_NAME = "lunarLander_1000_a=0.001_em=0.01_df=0.12"
    filepath = f"../trainedAgents/{AGENT_NAME}.pk1"

    # Uncomment desired test option: testCartPole() or testLunarLander()

    # testCartPole(filepath=filepath)
    testLunarLander(filepath=filepath, num_episodes=50)

if __name__ == "__main__":
    main()