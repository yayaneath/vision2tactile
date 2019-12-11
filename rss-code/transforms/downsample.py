import numpy as np

class Downsample(object):
    def __init__(self, target_points):
        self.target_points = target_points

    def __call__(self, sample):
        # TODO: Two possible methodologies for samples which have less points than
        # the target downsampled cloud
        #
        # 1) Set replace=True for those smaller clouds, so some points are repeated
        # in the cloud in order to produce a 'downsampled' cloud with the target points
        #
        # 2) Find in the GraspPointsDataset which is the smallest cloud stored in a PCD
        # and set the target_points to that number.

        idx = np.random.choice(sample.shape[0], self.target_points, replace=False)
        downsampled = sample[idx, :]

        return downsampled