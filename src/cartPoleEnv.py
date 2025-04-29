from util import *

class CPFeatureExtractor():
    """
    Returns features vector from the Cart Pole environment.
    Maps (state, action) pairs to feature value.
    Cart Pole state space:
        0: cart position (+/- 2.5)
        1: cart velocity (+/- 2.5)
        2: pole angle (+/- 10)
        3: pole angular velocity (+/- 10)
    """
    def __init__(self):
        """
        Define the bounds of the Cart Pole environment.
        """
        self.state_bounds = [
            [-2.5, 2.5],
            [-2.5, 2.5],
            [-10.0, 10.0],
            [-10.0, 10.0],
        ]
    

    def getFeatures(self, state, action):
        feat_vec = Counter()
        # add bias feature of 1.0
        feat_vec["bias"] = 1.0

        # iterate over the 4-dimensional state vector
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