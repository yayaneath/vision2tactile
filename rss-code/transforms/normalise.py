import numpy as np

class NormaliseUnitSphere(object):
    def __call__(self, sample):
        # Transform to a reference frame with the centroid as the origin
        centroid = np.mean(sample, axis=0)
        normalised = sample - centroid

        # Scale to fit points in a unit sphere
        greatest_dist = np.max(np.sqrt(np.sum(abs(normalised)**2, axis=-1)))
        normalised = normalised / greatest_dist

        return normalised