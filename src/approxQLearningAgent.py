import random
from util import *

class ApproxQAgent():
    """
    Approximate Q-Learning agent.
    Adapted from the Berkeley CS 188 Pacman intelligent agent project.
    """
    def __init__(self, feat_extractor, actions, alpha, epsilon, gamma):
        self.feat_extractor = feat_extractor
        self.alpha = float(alpha) # learning rate
        self.epsilon = float(epsilon) # exploration rate
        self.gamma = float(gamma) # discount factor
        self.actions = actions
        self.weights = Counter()
        self.isTerminal = False

    def getWeights(self):
        """
        Returns the weights vector.
        """
        return self.weights
    
    def storeWeights(self, state, action, weights):
        """
        Stores the updated weight in the weight vector.
        """
        self.weights[(state, action)] = weights
    
    def getLegalActions(self):
        """
        Returns the list of possible actions.
        If in terminal state, return empty array.
        """
        if self.isTerminal:
            actions = []
        else:
            actions = self.actions
        return actions
    
    def getApproxQValue(self, state, action):
        """
        Returns Q(s, a) = w * feat_vector (where * is the dot product).
        """
        feat_vec = self.feat_extractor.getFeatures(state, action)
        weights = self.getWeights()
        q_val = sum(weights[f] * feat_vec[f] for f in feat_vec)
        return q_val
    
    def computeMaxQValueOverActions(self, state):
        """
        Returns the max_a'(Q(s', a')).
        If in terminal state, return value 0.0.
        """
        actions = self.getLegalActions()
        if not actions:
            return 0.0
        return max(self.getApproxQValue(state, action) for action in actions)
    
    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state.
        If in terminal state, return None as action.
        """
        actions = self.getLegalActions()
        if not actions:
            return None
        return max(actions, key=lambda action: self.getApproxQValue(state, action))
    
    def determineRandomAction(self):
        """
        Returns a boolean indicating whether to take a random action
        based on the epsilon value.
        """
        return random.random() < self.epsilon
    
    def getAction(self, state):
        """
        Returns the best action to take in the current state.
        With probability self.epsilon, take a random action.
        If in terminal state, return None as action.
        """        
        actions = self.getLegalActions()
        if not actions:
            return None
        
        take_random_action = self.determineRandomAction()
        if take_random_action:
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(state)
        return action

    ############# update called by environment #############

    def update(self, state, action, next_state, reward, isTerminal):
        """
        Update the weights vector based on the
        Epsilon Greedy Approximate Q-learning algorithm.

        w_i <-- w_i + alpha ⋅ difference ⋅ f_i(s, a)

        difference = (r + gamma ⋅ max_a'(Q(s', a'))) - Q(s, a)
        """
        # set the terminal state flag
        self.isTerminal = isTerminal

        # feature vector for the current state/action pair
        feat_vec = self.feat_extractor.getFeatures(state, action)

        # current weights
        weights = self.getWeights()

        # approximate Q-value for current state/action pair
        approx_q = self.getApproxQValue(state, action)

        # get the max Q-value over possible actions in the NEXT state
        max_q_over_a = self.computeMaxQValueOverActions(next_state)

        # calculate the difference
        difference = (reward + (self.gamma * max_q_over_a)) - approx_q

        # update the weights for each feature in the feature vector
        for feat_name in feat_vec:
            feat_value = feat_vec[feat_name]
            w_i = weights[feat_name]

            # calculate the weight update
            updated_w_i = w_i + (self.alpha * difference * feat_value)
            # store the updated weight
            self.weights[feat_name] = updated_w_i