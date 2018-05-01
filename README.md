# AE-DQN
Usefull links:
Tensorflow code for DQN + Count based exploration: https://github.com/steveKapturowski/tensorflow-rl
You want to run it with --alg_type dqn-cts

This this the Q learner:
https://github.com/steveKapturowski/tensorflow-rl/blob/master/algorithms/intrinsic_motivation_actor_learner.py#L216 
This defines the Q network:
https://github.com/steveKapturowski/tensorflow-rl/blob/master/networks/q_network.py 
This defines the pseudo counts:
https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/cts_density_model.py Notice that they use a Cython implementation which works faster (https://github.com/steveKapturowski/tensorflow-rl/blob/master/utils/fast_cts.pyx) but in case you need to make any changes it is probably best to make the code use the Python version.

For Python 3
https://github.com/sangjin-park/tensorflow-rl

For gym environment games
http://gym.openai.com/envs/#atari
