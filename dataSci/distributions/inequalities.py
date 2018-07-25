import math

def get_hoeffding_n(dist, alpha = 0.05):
    '''Calculates the necessary sample size, n, to achieve a 1 - alpha confidence
     interval that a group mean of observations of random variable X_1, ..., X_n is within
     distance dist of the expected value E[\bar{X}].

    This is a very conservative bound for most uses.'''
    assert 0.0 < alpha < 1.0, "Error: alpha must be beteen zero and one"

    return math.ceil(math.log(2.0 / alpha) / (2.0 * math.pow(dist, 2)))

def get_hoeffding_interval(n, alpha = 0.05):
    '''Computes the distance, such that the level of the significance of the
    confidence interval is alpha.
    '''
    assert 0.0 < alpha < 1.0, "Error: alpha must be beteen zero and one"

    return math.sqrt(- math.log(alpha / 2.0) / (2.0*n))
