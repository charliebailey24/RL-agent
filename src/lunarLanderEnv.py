from util import *

class LLFeatureExtractor():
    """
    Returns features vector from environment.
    Maps (state, action) pairs to feature value.
    Lunar Lander state space:
        0: x coordinate of the lander (+/- 2.5)
        1: y coordinate of the lander (+/- 2.5)
        2: x velocity of the lander (+/- 10)
        3: y velocity of the lander (+/- 10)
        4: angle (+/- 6.2831855 rad)
        5: angular velocity (+/- 10)
        6: left leg touching ground (bool +/- 1)
        7: right leg touching ground (bool +/- 1)
    """
    def __init__(self):
        """
        Define the bounds of the Lunar Lander environment.
        """
        self.state_bounds = [
            [-2.5, 2.5],
            [-2.5, 2.5],
            [-10.0, 10.0],
            [-10.0, 10.0],
            [-6.28, 6.28],
            [-10.0, 10.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]


    def getFeatures(self, state, action):
        feat_vec = Counter()
        # add bias feature of 1.0
        feat_vec["bias"] = 1.0

        # iterate over the 8-dimensional state vector
        for i, value in enumerate(state):
            # get the bounds for this state
            min_bound, max_bound = self.state_bounds[i]

            # normalize the value
            norm_value = normalize(value, min_bound, max_bound)

            # create a unique feature name for each state/action pair
            feat_name = f"s_{i}_a_{action}"

            # store the normalized value for this state/action pair in the feature vector
            feat_vec[feat_name] = norm_value
        
        return feat_vec