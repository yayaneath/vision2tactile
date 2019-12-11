import numpy as np
import torch

class ToTensor(object):
    def __call__(self, sample):
        sample_tensor = torch.from_numpy(sample)
        sample_tensor = torch.t(sample_tensor) # swap channels and points

        return sample_tensor