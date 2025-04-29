import random
import numpy as np

class RBFExtractor:
    """
    Transforms a raw state/action pair into a sparse feature vector.
    """
    def __init__(self, scaler, rbf):
        self.scaler = scaler
        self.rbf = rbf

    def getFeatures(self, state, action):

        # reshape the state into a 2D array then scale using scaler
        z = self.scaler.transform(state.reshape(1, -1))
        # transform the scaled state using the RBF
        phi = self.rbf.transform(z)[0]

        feat_vec = Counter()
        feat_vec["bias"] = 1.0

        for i, value in enumerate(phi):
            if value != 0.0:
                # create unique feature name
                feat_name = f"rbf_{i}_a_{action}"
                # store the 
                feat_vec[feat_name] = value
        
        return feat_vec
    

class Counter(dict):
    """
    Extension of the Python dict type.
    Adds functionality to default all key values to 0.
    """

    # override the standard __getitem__ method
    # to return 0 if the key is not present in the dictionary
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)
    

def normalize(value, min_bound, max_bound):
    """
    Normalize value to be in range [-1, to 1].
    """

    # normalize the state value
    norm_value = 2.0 * (value - min_bound) / (max_bound - min_bound) - 1.0

    # clip the value to ensure it's between [-1, 1]
    norm_value = np.clip(norm_value, -1.0, 1.0)

    return norm_value