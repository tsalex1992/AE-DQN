 -*- encoding: utf-8 -*-
import time
import numpy as np
import utils.logger
import tensorflow as tf
from collections import deque
from utils import checkpoint_utils
from utils.decorators import only_on_train
from .actor_learner import ActorLearner, ONE_LIFE_GAMES
from networks.policy_v_network import PolicyValueNetwork
import csv


logger = utils.logger.getLogger('AE_actor_learner')


class AE_actor_learner(ActorLearner):
    def __init__(self, args):
        super(AE_actor_learner, self).__init__(args)

        self.ae_delta = args.ae_delta
        self.ae_epsilon = args.ae_epsilon
