import numpy as np


# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()

    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()

    def done(self, agent, world):
        return False

    def preprocess_rewards(self, world):
        return
