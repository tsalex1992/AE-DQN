# -*- encoding: utf-8 -*-
import time
import pickle
import numpy as np
import utils.logger
import tensorflow as tf
import random
from skimage.transform import resize
from collections import deque
from utils import checkpoint_utils
from .actor_learner import ONE_LIFE_GAMES
from utils.decorators import Experimental
from utils.fast_cts import CTSDensityModel
from utils.replay_memory import ReplayMemory
from .policy_based_actor_learner import A3CLearner, A3CLSTMLearner
from .value_based_actor_learner import ValueBasedLearner
import math


logger = utils.logger.getLogger('intrinsic_motivation_actor_learner')


class PixelCNNDensityModel(object):
    pass


class PerPixelDensityModel(object):
    """
    Calculates image probability according to per-pixel counts: P(X) = ∏ p(x_ij)
    Mostly here for debugging purposes as CTSDensityModel is much more expressive
    """
    def __init__(self, height=42, width=42, num_bins=8, beta=0.05):
        self.counts = np.zeros((width, height, num_bins))
        self.height = height
        self.width = width
        self.beta = beta
        self.num_bins = num_bins

    def update(self, obs):
        obs = resize(obs, (self.height, self.width), preserve_range=True)
        obs = np.floor((obs*self.num_bins)).astype(np.int32)

        log_prob, log_recoding_prob = self._update(obs)
        return self.exploration_bonus(log_prob, log_recoding_prob)

    def _update(self, obs):
        log_prob = 0.0
        log_recoding_prob = 0.0

        for i in range(self.height):
            for j in range(self.height):
                self.counts[i, j, obs[i, j]] += 1

                bin_count = self.counts[i, j, obs[i, j]]
                pixel_mass = self.counts[i, j].sum()
                log_prob += np.log(bin_count / pixel_mass)
                log_recoding_prob += np.log((bin_count + 1) / (pixel_mass + 1))

        return log_prob, log_recoding_prob

    def exploration_bonus(self, log_prob, log_recoding_prob):
        recoding_prob = np.exp(log_recoding_prob)
        prob_ratio = np.exp(log_recoding_prob - log_prob)

        pseudocount = (1 - recoding_prob) / np.maximum(prob_ratio - 1, 1e-10)
        return self.beta / np.sqrt(pseudocount + .01)

    def get_state(self):
        return self.num_bins, self.height, self.width, self.beta, self.counts

    def set_state(self, state):
        self.num_bins, self.height, self.width, self.beta, self.counts = state


class DensityModelMixin(object):
    """
    Mixin to provide initialization and synchronization methods for density models
    """
    def _init_density_model(self, args):
        self.density_model_update_steps = 20*args.q_target_update_steps
        self.density_model_update_flags = args.density_model_update_flags

        model_args = {
            'height': args.cts_rescale_dim,
            'width': args.cts_rescale_dim,
            'num_bins': args.cts_bins,
            'beta': args.cts_beta
        }
        if args.density_model == 'cts':
            self.density_model = CTSDensityModel(**model_args)
        else:
            self.density_model = PerPixelDensityModel(**model_args)


    def write_density_model(self):
        logger.info('T{} Writing Pickled Density Model to File...'.format(self.actor_id))
        raw_data = pickle.dumps(self.density_model.get_state(), protocol=2)
        with self.barrier.counter.lock, open('/tmp/density_model.pkl', 'wb') as f:
            f.write(raw_data)

        for i in range(len(self.density_model_update_flags.updated)):
            self.density_model_update_flags.updated[i] = 1

    def read_density_model(self):
        logger.info('T{} Synchronizing Density Model...'.format(self.actor_id))
        with self.barrier.counter.lock, open('/tmp/density_model.pkl', 'rb') as f:
            raw_data = f.read()

        self.density_model.set_state(pickle.loads(raw_data))

class DensityModelMixinAE(object):
        """
        Mixin to provide initialization and synchronization methods for density models
        """
        def _init_density_model(self, args):
            self.density_model_update_steps = 20*args.q_target_update_steps
            self.density_model_update_flags = []
            for x in range(0, args.num_actions):
                self.density_model_update_flags.append(args.density_model_update_flags)
                print("x is: {}".format(x))

            model_args = {
                'height': args.cts_rescale_dim,
                'width': args.cts_rescale_dim,
                'num_bins': args.cts_bins,
                'beta': args.cts_beta
            }
            self.density_model = []
            for x in range (0,args.num_actions):
                if args.density_model == 'cts':
                    self.density_model.append(CTSDensityModel(**model_args))
                else:
                    self.density_model.append(PerPixelDensityModel(**model_args))


        def write_density_model(self , index):
            logger.info('T{} Writing Pickled Density  Model of action{} to File...'.format(self.actor_id,index))
            raw_data = pickle.dumps(self.density_model[index].get_state(), protocol=2)
            with self.barrier.counter.lock, open('/tmp/density_model'+str(index)+'.pkl', 'wb') as f:
                f.write(raw_data)

            for i in range(len(self.density_model_update_flags[index].updated)):
                self.density_model_update_flags[index].updated[i] = 1

        def read_density_model(self,index):
            logger.info('T{} Synchronizing Density Model...'.format(self.actor_id))
            with self.barrier.counter.lock, open('/tmp/density_model'+str(index)+'.pkl', 'rb') as f:
                raw_data = f.read()

            self.density_model[index].set_state(pickle.loads(raw_data))



class A3CDensityModelMixin(DensityModelMixin):
    """
    Mixin to share _train method between A3C and A3C-LSTM models
    """
    def _train(self):
        """ Main actor learner loop for advantage actor critic learning. """
        logger.debug("Actor {} resuming at Step {}".format(self.actor_id,
            self.global_step.value()))
        print("we are in class A3A3CDensityModelMixin")

        bonuses = deque(maxlen=100)
        while (self.global_step.value() < self.max_global_steps):
            # Sync local learning net with shared mem
            s = self.emulator.get_initial_state()
            self.reset_hidden_state()
            self.local_episode += 1
            episode_over = False
            total_episode_reward = 0.0
            episode_start_step = self.local_step

            while not episode_over:
                self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
                self.save_vars()

                rewards = list()
                states  = list()
                actions = list()
                values  = list()
                local_step_start = self.local_step
                self.set_local_lstm_state()

                while self.local_step - local_step_start < self.max_local_steps and not episode_over:
                    # Choose next action and execute it
                    a, readout_v_t, readout_pi_t = self.choose_next_action(s)
                    new_s, reward, episode_over = self.emulator.next(a)
                    total_episode_reward += reward

                    # Update density model
                    current_frame = new_s[...,-1]
                    bonus = self.density_model.update(current_frame)
                    bonuses.append(bonus)

                    if self.is_master() and (self.local_step % 400 == 0):
                        bonus_array = np.array(bonuses)
                        logger.debug('π_a={:.4f} / V={:.4f} / Mean Bonus={:.4f} / Max Bonus={:.4f}'.format(
                            readout_pi_t[a.argmax()], readout_v_t, bonus_array.mean(), bonus_array.max()))

                    # Rescale or clip immediate reward
                    reward = self.rescale_reward(self.rescale_reward(reward) + bonus)
                    rewards.append(reward)
                    states.append(s)
                    actions.append(a)
                    values.append(readout_v_t)

                    s = new_s
                    self.local_step += 1

                    global_step, _ = self.global_step.increment()
                    if global_step % self.density_model_update_steps == 0:
                        self.write_density_model()
                    if self.density_model_update_flags.updated[self.actor_id] == 1:
                        self.read_density_model()
                        self.density_model_update_flags.updated[self.actor_id] = 0

                next_val = self.bootstrap_value(new_s, episode_over)
                advantages = self.compute_gae(rewards, values, next_val)
                targets = self.compute_targets(rewards, next_val)
                # Compute gradients on the local policy/V network and apply them to shared memory
                entropy = self.apply_update(states, actions, targets, advantages)

            elapsed_time = time.time() - self.start_time
            steps_per_sec = self.global_step.value() / elapsed_time
            perf = "{:.0f}".format(steps_per_sec)
            logger.info("T{} / EPISODE {} / STEP {}k / REWARD {} / {} STEPS/s".format(
                self.actor_id,
                self.local_episode,
                self.global_step.value()/1000,
                total_episode_reward,
                perf))

            self.log_summary(total_episode_reward, np.array(values).mean(), entropy)


@Experimental
class PseudoCountA3CLearner(A3CLearner, A3CDensityModelMixin):
    """
    Attempt at replicating the A3C+ model from the paper 'Unifying Count-Based Exploration and Intrinsic Motivation' (https://arxiv.org/abs/1606.01868)
    """
    def __init__(self, args):
        super(PseudoCountA3CLearner, self).__init__(args)
        self._init_density_model(args)

    def train(self):
        self._train()


@Experimental
class PseudoCountA3CLSTMLearner(A3CLSTMLearner, A3CDensityModelMixin):
    def __init__(self, args):
        super(PseudoCountA3CLSTMLearner, self).__init__(args)
        self._init_density_model(args)

    def train(self):
        self._train()

        return new_action, q_values

class AElearner(ValueBasedLearner,DensityModelMixinAE):
    def __init__(self, args):
        self.args = args

        super(AElearner, self).__init__(args)
        self.cts_eta = args.cts_eta
        self.cts_beta = args.cts_beta
        self.ae_delta = args.ae_delta
        self.batch_size = args.batch_update_size
        self.replay_memory = ReplayMemory(
            args.replay_size,
            self.local_network_upper.get_input_shape(),
            # self.local_network.get_input_shape(),
            self.num_actions)
        #inits desity model(chooses how many steps for update )
        #20 * q targt update steps
        self._init_density_model(args)
        #computes loss
        self._double_dqn_op()
        self.which_net_to_update_counter = 0
        print("In AE class")
    ### Supposedly initializes the q uper and lower output values for each state and action
    def init_q_values(self,A,S,delta,Vmax,c):
        print ("The output layer is: {}".format(type(self.target_network_upper.output_layer)))



    ### pay attention: call it for upper q
    ### Returns minimized action pool after AE according to the paper(Q upper is larger than V lower)
    def minimize_action_pool(self, state):
        new_actions = np.zeros([self.num_actions])
        #TODO get q upperbound values
        # q_upper
        q_values_upper = self.session.run(
                self.target_network_upper.output_layer,
                feed_dict={self.target_network_upper.input_ph: [state]})[0]
        q_values_lower = self.session.run(
                self.target_network_lower.output_layer,
                feed_dict={self.target_network_lower.input_ph: [state]})[0]
        #TODO V lower upperbound
        Vlow = max(q_values_lower)
        print("The value of Vlow is {}".format(Vlow))
        for index, action in enumerate(new_actions):
            action = q_values_upper[index] >= Vlow
            print("The value of q_values_upper on index: {} is :{}".format(index,q_values_upper[index]))
        return new_actions





    def choose_next_action(self, state):
        print("we use our AE new algorithm choose next action")
        new_action = np.zeros([self.num_actions])
        #TODO check session run
        q_values = self.session.run(
            self.local_network_upper.output_layer,
            feed_dict={self.local_network_upper.input_ph: [state]})[0]
        secure_random = random.SystemRandom()
        action_pool = self.minimize_action_pool(state)
        print("The action pool {}".format(action_pool))
        random_index = secure_random.randrange(0,len(action_pool))
        indexes_valid_actions=[]
        for i, item in enumerate(action_pool):
            if item == 1 :
                indexes_valid_actions.append(i)


        print(indexes_valid_actions)
        random_index = secure_random.choice(indexes_valid_actions)

        new_action[random_index] = 1
        self.reduce_thread_epsilon()
        return new_action,q_values

    def generate_final_epsilon(self):
        if self.num_actor_learners == 1:
            return self.args.final_epsilon
        else:
            return super(AElearner, self).generate_final_epsilon()


    def _get_summary_vars(self):
        q_vars = super(AElearner, self)._get_summary_vars()

        bonus_q05 = tf.Variable(0., name='novelty_bonus_q05')
        s1 = tf.summary.scalar('Novelty_Bonus_q05_{}'.format(self.actor_id), bonus_q05)
        bonus_q50 = tf.Variable(0., name='novelty_bonus_q50')
        s2 = tf.summary.scalar('Novelty_Bonus_q50_{}'.format(self.actor_id), bonus_q50)
        bonus_q95 = tf.Variable(0., name='novelty_bonus_q95')
        s3 = tf.summary.scalar('Novelty_Bonus_q95_{}'.format(self.actor_id), bonus_q95)

        augmented_reward = tf.Variable(0., name='augmented_episode_reward')
        s4 = tf.summary.scalar('Augmented_Episode_Reward_{}'.format(self.actor_id), augmented_reward)

        return q_vars + [bonus_q05, bonus_q50, bonus_q95, augmented_reward]

    def beta_func(self, A,S,delta,k,Vmax,c):
        left = math.sqrt(k*math.log(c*k*k*S*A/delta))
        right = math.sqrt((k-1)*math.log(c*(k-1)*(k-1)*S*A/delta))
        beta = k*Vmax*(left-(1-1/k)*right)
        return beta


    #TODO: refactor to make this cleaner
    def prepare_state(self, state, total_episode_reward, steps_at_last_reward,
                      ep_t, episode_ave_max_q, episode_over, bonuses, total_augmented_reward):
        # Start a new game on reaching terminal state
        if episode_over:
            T = self.global_step.value() * self.max_local_steps
            t = self.local_step
            e_prog = float(t)/self.epsilon_annealing_steps
            episode_ave_max_q = episode_ave_max_q/float(ep_t)
            s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
            s2 = "EPS {0:.4f}".format(self.epsilon)

            self.scores.insert(0, total_episode_reward)
            if len(self.scores) > 100:
                self.scores.pop()

            logger.info('T{0} / STEP {1} / REWARD {2} / {3} / {4}'.format(
                self.actor_id, T, total_episode_reward, s1, s2))
            logger.info('ID: {0} -- RUNNING AVG: {1:.0f} +- {2:.0f} -- BEST: {3:.0f}'.format(
                self.actor_id,
                np.array(self.scores).mean(),
                2*np.array(self.scores).std(),
                max(self.scores),
            ))
            self.vis.plot_current_errors(T,total_episode_reward)
            self.wr.writerow(T)
            self.wr.writerow(total_episode_reward)
            #print ('[%s]' % ', '.join(map(str, t.vis.plot_data['X'])))


            self.log_summary(
                total_episode_reward,
                episode_ave_max_q,
                self.epsilon,
                # np.percentile(bonuses, 5),
                # np.percentile(bonuses, 50),
                # np.percentile(bonuses, 95),
                total_augmented_reward,
            )

            state = self.emulator.get_initial_state()
            ep_t = 0
            total_episode_reward = 0
            episode_ave_max_q = 0
            episode_over = False

        return (
            state,
            total_episode_reward,
            steps_at_last_reward,
            ep_t,
            episode_ave_max_q,
            episode_over
        )


    def _double_dqn_op(self):
        q_local_action_lower = tf.cast(tf.argmax(
            self.local_network_lower.output_layer, axis=1), tf.int32)
        q_target_max_lower = utils.ops.slice_2d(
            self.target_network_lower.output_layer,
            tf.range(0, self.batch_size),
            q_local_action_lower,
        )

        q_local_action_upper = tf.cast(tf.argmax(
            self.local_network_upper.output_layer, axis=1), tf.int32)
        q_target_max_upper = utils.ops.slice_2d(
            self.target_network_upper.output_layer,
            tf.range(0, self.batch_size),
            q_local_action_upper,
        )

        self.one_step_reward = tf.placeholder(tf.float32, self.batch_size, name='one_step_reward')
        self.is_terminal = tf.placeholder(tf.bool, self.batch_size, name='is_terminal')

        self.y_target_lower = self.one_step_reward + self.cts_eta*self.gamma*q_target_max_lower \
            * (1 - tf.cast(self.is_terminal, tf.float32))

        self.y_target_upper = self.one_step_reward + self.cts_eta*self.gamma*q_target_max_upper \
            * (1 - tf.cast(self.is_terminal, tf.float32))

        self.double_dqn_loss_lower = self.local_network_lower._value_function_loss(
            self.local_network_lower.q_selected_action
            - tf.stop_gradient(self.y_target_lower))

        self.double_dqn_loss_upper = self.local_network_upper._value_function_loss(
            self.local_network_upper.q_selected_action
            - tf.stop_gradient(self.y_target_upper))


        self.double_dqn_grads_lower = tf.gradients(self.double_dqn_loss_lower, self.local_network_lower.params)
        self.double_dqn_grads_upper = tf.gradients(self.double_dqn_loss_upper, self.local_network_upper.params)



    # def batch_update(self):
    #     if len(self.replay_memory) < self.replay_memory.maxlen//10:
    #         return

    #     s_i, a_i, r_i, s_f, is_terminal = self.replay_memory.sample_batch(self.batch_size)

    #     feed_dict={
    #         self.one_step_reward: r_i,
    #         self.target_network.input_ph: s_f,
    #         self.local_network.input_ph: np.vstack([s_i, s_f]),
    #         self.local_network.selected_action_ph: np.vstack([a_i, a_i]),
    #         self.is_terminal: is_terminal
    #     }
    #     grads = self.session.run(self.double_dqn_grads, feed_dict=feed_dict)
    #     self.apply_gradients_to_shared_memory_vars(grads)


    def batch_update(self):
        if len(self.replay_memory) < self.replay_memory.maxlen//10:
            return
        #TODO check if we need two replay memories
        s_i, a_i, r_i, s_f, is_terminal ,b_i = self.replay_memory.sample_batch(self.batch_size)

        # if(self.which_net_to_update_counter %2):

        feed_dict={
            self.local_network_upper.input_ph: s_f,
            self.target_network_upper.input_ph: s_f,
            self.is_terminal: is_terminal,
            self.one_step_reward: r_i+b_i,
        }
        y_target_upper = self.session.run(self.y_target_upper, feed_dict=feed_dict)

        feed_dict={
            self.local_network_upper.input_ph: s_i,
            self.local_network_upper.target_ph: y_target_upper,
            self.local_network_upper.selected_action_ph: a_i
        }
        grads = self.session.run(self.local_network_upper.get_gradients,
                                     feed_dict=feed_dict)
        self.apply_gradients_to_shared_memory_vars(grads)
        # else:
        feed_dict={
            self.local_network_lower.input_ph: s_f,
            self.target_network_lower.input_ph: s_f,
            self.is_terminal: is_terminal,
            self.one_step_reward: r_i-b_i,
        }
        y_target_lower = self.session.run(self.y_target_lower, feed_dict=feed_dict)

        feed_dict={
            self.local_network_lower.input_ph: s_i,
            self.local_network_lower.target_ph: y_target_lower,
            self.local_network_lower.selected_action_ph: a_i
        }
        grads = self.session.run(self.local_network_lower.get_gradients,
                                    feed_dict=feed_dict)
        self.apply_gradients_to_shared_memory_vars(grads)



    def train(self):
        """ Main actor learner loop for n-step Q learning. """
        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id,
            self.global_step.value(), time.ctime()))

        s = self.emulator.get_initial_state()
        print(" In train of AE")
        s_batch = list()
        a_batch = list()
        y_batch = list()
        bonuses = deque(maxlen=1000)
        episode_over = False

        t0 = time.time()
        global_steps_at_last_record = self.global_step.value()
        while (self.global_step.value() < self.max_global_steps):
            # # Sync local learning net with shared mem
            # self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            # self.save_vars()
            rewards =      list()
            states =       list()
            actions =      list()
            max_q_values = list()
            bonuses = list()
            local_step_start = self.local_step
            total_episode_reward = 0
            total_augmented_reward = 0
            episode_ave_max_q = 0
            ep_t = 0
            action_count = 0

            while not episode_over:

                A = self.num_actions
                S = 10000000
                delta = self.ae_delta
                Vmax = 10000
                c = 5
                # Sync local learning net with shared mem
                self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
                self.save_vars()

                # Choose next action and execute it
                print("intrinsic motivation print")
                #TODO we need an init for the first iteration of
                # qupper and qlower values
                if action_count == 0 :
                    self.init_q_values(A,S,delta,Vmax,c)
                else:
                    a, q_values = self.choose_next_action(s)
                action_count+= 1
                #TODO here is the update of the iteration
                new_s, reward, episode_over = self.emulator.next(a)
                total_episode_reward += reward
                max_q = np.max(q_values)
                prev_s = s
                current_frame = new_s[...,-1]
                prev_frame = prev_s[...,-1]
                print("This is a {}".format(a))
                index_of_a = np.argmax(a)
                ## TODO change back to update 2 and understand the underlying
                ## cython code
                k = (self.density_model[index_of_a]).update(prev_frame)
                #TODO add parameters




                bonus =  beta_func(A,S,delta,k,Vmax,c)


                # Rescale or clip immediate reward
                reward = self.rescale_reward(self.rescale_reward(reward))
                total_augmented_reward += reward
                ep_t += 1

                rewards.append(reward)
                states.append(s)
                actions.append(a)
                bonuses.append(bonus)
                max_q_values.append(max_q)

                s = new_s
                self.local_step += 1
                episode_ave_max_q += max_q

                global_step, _ = self.global_step.increment()
                ##We update the target network here
                if global_step % self.q_target_update_steps == 0:
                    self.update_target()
                ## We update the desity model here
                if global_step % self.density_model_update_steps == 0:
                    #returns the index of the chosen action
                    self.write_density_model(np.argmax(a))

                # Sync local tensorflow target network params with shared target network params
                if self.target_update_flags.updated[self.actor_id] == 1:
                    self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                    self.target_update_flags.updated[self.actor_id] = 0
                if self.density_model_update_flags.updated[self.actor_id] == 1:
                    #returns the index of the chosen action
                    self.read_density_model(np.argmax(a))
                    self.density_model_update_flags.updated[self.actor_id] = 0

                if self.local_step % self.q_update_interval == 0:
                    self.batch_update()
                    self.which_net_to_update_counter += 1

                if self.is_master() and (self.local_step % 500 == 0):
                    #bonus_array = np.array(bonuses)
                    steps = global_step - global_steps_at_last_record
                    global_steps_at_last_record = global_step

                    # logger.debug('Mean Bonus={:.4f} / Max Bonus={:.4f} / STEPS/s={}'.format(
                    # #    bonus_array.mean(), bonus_array.max(), steps/float(time.time()-t0)))
                    t0 = time.time()


            else:
                #compute monte carlo return
                mc_returns = np.zeros((len(rewards),), dtype=np.float32)
                running_total = 0.0
                for i, r in enumerate(reversed(rewards)):
                    running_total = r + self.gamma*running_total
                    mc_returns[len(rewards)-i-1] = running_total

                mixed_returns = self.cts_eta*np.asarray(rewards) + (1-self.cts_eta)*mc_returns

                #update replay memory
                states.append(new_s)
                episode_length = len(rewards)
                for i in range(episode_length):
                    self.replay_memory.append(
                        states[i],
                        actions[i],
                        mixed_returns[i],
                        i+1 == episode_length,
                        bonuses[i])

            s, total_episode_reward, _, ep_t, episode_ave_max_q, episode_over = \
                self.prepare_state(s, total_episode_reward, self.local_step, ep_t, episode_ave_max_q, episode_over, bonuses, total_augmented_reward)

class PseudoCountQLearner(ValueBasedLearner, DensityModelMixin):
    """
    Based on DQN+CTS model from the paper 'Unifying Count-Based Exploration and Intrinsic Motivation' (https://arxiv.org/abs/1606.01868)
    Presently the implementation differs from the paper in that the novelty bonuses are computed online rather than by computing the
    prediction gains after the model has been updated with all frames from the episode. Async training with different final epsilon values
    tends to produce better results than just using a single actor-learner.
    """
    def __init__(self, args):
        self.args = args
        super(PseudoCountQLearner, self).__init__(args)

        self.cts_eta = args.cts_eta
        self.cts_beta = args.cts_beta
        self.batch_size = args.batch_update_size
        self.replay_memory = ReplayMemory(
            args.replay_size,
            self.local_network.get_input_shape(),
            self.num_actions)
        #inits desity model(chooses how many steps for update )
        #20 * q targt update steps
        self._init_density_model(args)
        #computes loss
        self._double_dqn_op()


    def generate_final_epsilon(self):
        if self.num_actor_learners == 1:
            return self.args.final_epsilon
        else:
            return super(PseudoCountQLearner, self).generate_final_epsilon()


    def _get_summary_vars(self):
        q_vars = super(PseudoCountQLearner, self)._get_summary_vars()

        bonus_q05 = tf.Variable(0., name='novelty_bonus_q05')
        s1 = tf.summary.scalar('Novelty_Bonus_q05_{}'.format(self.actor_id), bonus_q05)
        bonus_q50 = tf.Variable(0., name='novelty_bonus_q50')
        s2 = tf.summary.scalar('Novelty_Bonus_q50_{}'.format(self.actor_id), bonus_q50)
        bonus_q95 = tf.Variable(0., name='novelty_bonus_q95')
        s3 = tf.summary.scalar('Novelty_Bonus_q95_{}'.format(self.actor_id), bonus_q95)

        augmented_reward = tf.Variable(0., name='augmented_episode_reward')
        s4 = tf.summary.scalar('Augmented_Episode_Reward_{}'.format(self.actor_id), augmented_reward)

        return q_vars + [bonus_q05, bonus_q50, bonus_q95, augmented_reward]


    #TODO: refactor to make this cleaner
    def prepare_state(self, state, total_episode_reward, steps_at_last_reward,
                      ep_t, episode_ave_max_q, episode_over, bonuses, total_augmented_reward):
        # Start a new game on reaching terminal state
        if episode_over:
            T = self.global_step.value() * self.max_local_steps
            t = self.local_step
            e_prog = float(t)/self.epsilon_annealing_steps
            episode_ave_max_q = episode_ave_max_q/float(ep_t)
            s1 = "Q_MAX {0:.4f}".format(episode_ave_max_q)
            s2 = "EPS {0:.4f}".format(self.epsilon)

            self.scores.insert(0, total_episode_reward)
            if len(self.scores) > 100:
                self.scores.pop()

            logger.info('T{0} / STEP {1} / REWARD {2} / {3} / {4}'.format(
                self.actor_id, T, total_episode_reward, s1, s2))
            logger.info('ID: {0} -- RUNNING AVG: {1:.0f} +- {2:.0f} -- BEST: {3:.0f}'.format(
                self.actor_id,
                np.array(self.scores).mean(),
                2*np.array(self.scores).std(),
                max(self.scores),
            ))
            self.vis.plot_current_errors(T,total_episode_reward)
            self.wr.writerow(T)
            self.wr.writerow(total_episode_reward)
            #print ('[%s]' % ', '.join(map(str, t.vis.plot_data['X'])))


            self.log_summary(
                total_episode_reward,
                episode_ave_max_q,
                self.epsilon,
                np.percentile(bonuses, 5),
                np.percentile(bonuses, 50),
                np.percentile(bonuses, 95),
                total_augmented_reward,
            )

            state = self.emulator.get_initial_state()
            ep_t = 0
            total_episode_reward = 0
            episode_ave_max_q = 0
            episode_over = False

        return (
            state,
            total_episode_reward,
            steps_at_last_reward,
            ep_t,
            episode_ave_max_q,
            episode_over
        )


    def _double_dqn_op(self):
        q_local_action = tf.cast(tf.argmax(
            self.local_network.output_layer, axis=1), tf.int32)
        q_target_max = utils.ops.slice_2d(
            self.target_network.output_layer,
            tf.range(0, self.batch_size),
            q_local_action,
        )
        self.one_step_reward = tf.placeholder(tf.float32, self.batch_size, name='one_step_reward')
        self.is_terminal = tf.placeholder(tf.bool, self.batch_size, name='is_terminal')

        self.y_target = self.one_step_reward + self.cts_eta*self.gamma*q_target_max \
            * (1 - tf.cast(self.is_terminal, tf.float32))

        self.double_dqn_loss = self.local_network._value_function_loss(
            self.local_network.q_selected_action
            - tf.stop_gradient(self.y_target))

        self.double_dqn_grads = tf.gradients(self.double_dqn_loss, self.local_network.params)


    # def batch_update(self):
    #     if len(self.replay_memory) < self.replay_memory.maxlen//10:
    #         return

    #     s_i, a_i, r_i, s_f, is_terminal = self.replay_memory.sample_batch(self.batch_size)

    #     feed_dict={
    #         self.one_step_reward: r_i,
    #         self.target_network.input_ph: s_f,
    #         self.local_network.input_ph: np.vstack([s_i, s_f]),
    #         self.local_network.selected_action_ph: np.vstack([a_i, a_i]),
    #         self.is_terminal: is_terminal
    #     }
    #     grads = self.session.run(self.double_dqn_grads, feed_dict=feed_dict)
    #     self.apply_gradients_to_shared_memory_vars(grads)


    def batch_update(self):
        if len(self.replay_memory) < self.replay_memory.maxlen//10:
            return

        s_i, a_i, r_i, s_f, is_terminal = self.replay_memory.sample_batch(self.batch_size)

        feed_dict={
            self.local_network.input_ph: s_f,
            self.target_network.input_ph: s_f,
            self.is_terminal: is_terminal,
            self.one_step_reward: r_i,
        }
        y_target = self.session.run(self.y_target, feed_dict=feed_dict)

        feed_dict={
            self.local_network.input_ph: s_i,
            self.local_network.target_ph: y_target,
            self.local_network.selected_action_ph: a_i
        }
        grads = self.session.run(self.local_network.get_gradients,
                                 feed_dict=feed_dict)
        self.apply_gradients_to_shared_memory_vars(grads)


    def train(self):
        """ Main actor learner loop for n-step Q learning. """
        logger.debug("Actor {} resuming at Step {}, {}".format(self.actor_id,
            self.global_step.value(), time.ctime()))

        s = self.emulator.get_initial_state()
        s_batch = list()
        a_batch = list()
        y_batch = list()
        bonuses = deque(maxlen=1000)
        episode_over = False

        t0 = time.time()
        global_steps_at_last_record = self.global_step.value()
        while (self.global_step.value() < self.max_global_steps):
            # # Sync local learning net with shared mem
            # self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
            # self.save_vars()
            rewards =      list()
            states =       list()
            actions =      list()
            max_q_values = list()
            local_step_start = self.local_step
            total_episode_reward = 0
            total_augmented_reward = 0
            episode_ave_max_q = 0
            ep_t = 0

            while not episode_over:
                # Sync local learning net with shared mem
                self.sync_net_with_shared_memory(self.local_network, self.learning_vars)
                self.save_vars()

                # Choose next action and execute it
                a, q_values = self.choose_next_action(s)
                print ("Inside PseudoCountQLearner")

                #TODO here is the update of the iteration
                new_s, reward, episode_over = self.emulator.next(a)
                total_episode_reward += reward
                max_q = np.max(q_values)

                current_frame = new_s[...,-1]
                bonus = self.density_model.update(current_frame)
                bonuses.append(bonus)

                # Rescale or clip immediate reward
                reward = self.rescale_reward(self.rescale_reward(reward) + bonus)
                total_augmented_reward += reward
                ep_t += 1

                rewards.append(reward)
                states.append(s)
                actions.append(a)
                max_q_values.append(max_q)

                s = new_s
                self.local_step += 1
                episode_ave_max_q += max_q

                global_step, _ = self.global_step.increment()
                ##We update the target network here
                if global_step % self.q_target_update_steps == 0:
                    self.update_target()
                ## We update the desity model here
                if global_step % self.density_model_update_steps == 0:
                    self.write_density_model()

                # Sync local tensorflow target network params with shared target network params
                if self.target_update_flags.updated[self.actor_id] == 1:
                    self.sync_net_with_shared_memory(self.target_network, self.target_vars)
                    self.target_update_flags.updated[self.actor_id] = 0
                if self.density_model_update_flags.updated[self.actor_id] == 1:
                    self.read_density_model()
                    self.density_model_update_flags.updated[self.actor_id] = 0

                if self.local_step % self.q_update_interval == 0:
                    self.batch_update()

                if self.is_master() and (self.local_step % 500 == 0):
                    bonus_array = np.array(bonuses)
                    steps = global_step - global_steps_at_last_record
                    global_steps_at_last_record = global_step

                    logger.debug('Mean Bonus={:.4f} / Max Bonus={:.4f} / STEPS/s={}'.format(
                        bonus_array.mean(), bonus_array.max(), steps/float(time.time()-t0)))
                    t0 = time.time()


            else:
                #compute monte carlo return
                mc_returns = np.zeros((len(rewards),), dtype=np.float32)
                running_total = 0.0
                for i, r in enumerate(reversed(rewards)):
                    running_total = r + self.gamma*running_total
                    mc_returns[len(rewards)-i-1] = running_total

                mixed_returns = self.cts_eta*np.asarray(rewards) + (1-self.cts_eta)*mc_returns

                #update replay memory
                states.append(new_s)
                episode_length = len(rewards)
                for i in range(episode_length):
                    self.replay_memory.append(
                        states[i],
                        actions[i],
                        mixed_returns[i],
                        i+1 == episode_length)

            s, total_episode_reward, _, ep_t, episode_ave_max_q, episode_over = \
                self.prepare_state(s, total_episode_reward, self.local_step, ep_t, episode_ave_max_q, episode_over, bonuses, total_augmented_reward)
