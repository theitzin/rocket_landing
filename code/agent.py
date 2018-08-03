import multiprocessing as mp
import tensorflow as tf
import numpy as np
import os
import shutil
import time
import math
from collections import deque

class EnvironmentStack(object):
    def __init__(self, Environment, env_args={}, n_env=1):
        self.n_env = n_env
        self.environments = []
        for _ in range(self.n_env):
            self.environments.append(Environment(**env_args))

        self.state_shape = self.environments[0].observation_space.shape
        self.n_states = np.prod(self.state_shape)
        self.n_actions = self.environments[0].action_space.n

    def reset(self, **args):
        return [a.reset(**args) for a in self.environments]

    def step(self, actions):
        assert len(self.environments) == len(actions)
        
        data = [e.step(a) for e, a in zip(self.environments, actions)]
        return zip(*data)

    def update_difficulty(self, difficulty):
    	for e in self.environments:
    		e.difficulty = difficulty

    def render(self):
        self.environments[0].render()


class ActorCriticNetwork(object):
	def __init__(self, n_states, n_actions, config):
		self.n_states = n_states
		self.n_actions = n_actions
		self.config = config

		self.s = tf.placeholder(tf.float32, [None, *self.n_states], 'state')

		self.a = tf.placeholder(tf.int32, [None], 'action')
		self.v_target = tf.placeholder(tf.float32, [None], 'value_target')

		self.a_prob, self.v, self.ac_params = self._build_net()
		td = tf.subtract(self.v_target, self.v, name='td_error')

		with tf.name_scope('ac_loss'):
			clip = lambda x: tf.clip_by_value(x, 1e-6, 1 - 1e-6)

			probs = tf.reduce_sum(clip(self.a_prob)*tf.one_hot(self.a, self.n_actions), axis=-1)

			a_loss = tf.reduce_mean(tf.stop_gradient(td) * -tf.log(probs))
			entropy = -tf.reduce_mean(tf.reduce_sum(self.a_prob * tf.log(clip(self.a_prob)), axis=-1))

			c_loss = tf.reduce_mean(tf.square(td))
			self.ac_loss = a_loss - self.config['entropy_beta']*entropy + 0.5*c_loss

		with tf.name_scope('local_grad'):
			self.ac_grads = tf.gradients(self.ac_loss, self.ac_params)
			if self.config['max_grad_norm'] is not None:
				self.ac_grads, grad_norm = tf.clip_by_global_norm(self.ac_grads, self.config['max_grad_norm'])

		self.opt_ac = config['optimizer']

		with tf.name_scope('update'):
			self.update_ac_op = self.opt_ac.apply_gradients(zip(self.ac_grads, self.ac_params))

		# tensorboard variables
		tf.summary.scalar("max-p", tf.reduce_max(self.a_prob))
		tf.summary.scalar("ac-loss", self.ac_loss)
		self.mean_episode_reward = tf.Variable(0.)
		tf.summary.scalar("mean episode reward", self.mean_episode_reward)
		self.landing_success_rate = tf.Variable(0.)
		tf.summary.scalar("landing success rate", self.landing_success_rate)
		self.landing_duration = tf.Variable(0.)
		tf.summary.scalar("landing duration", self.landing_duration)
		self.difficulty = tf.Variable(0.)
		tf.summary.scalar("difficulty", self.difficulty)
		self.summary = tf.summary.merge_all()

	def _build_net(self):
		a, v = self.config['network'](self.s, self.n_actions)

		ac_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)        
		n_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in ac_params])
		print('number of network parameters: ', n_parameters)
		return a, v, ac_params

	def update(self, feed_dict, sess):
		sess.run([self.update_ac_op], feed_dict)

	def choose_action(self, s, sess):
		weights = sess.run(self.a_prob, feed_dict={self.s: s})
		actions = [np.random.choice(range(len(w)), p=w) for w in weights]
		return actions


class Agent(object):
	def __init__(self, Environment, config, n_env=1, n_threads=1):
		self.config = config
		self.workers = []
		self.summary = {
			'landing_success_rate' : deque(maxlen=100),
			'landing_duration' : deque(maxlen=100)
		}

		dummy_env = EnvironmentStack(
			Environment, 
			env_args={'initial_generator' : config['initial_generator']},
			n_env=n_env)
		self.state_shape = dummy_env.state_shape
		self.n_states = dummy_env.n_states
		self.n_actions = dummy_env.n_actions

		self.network = ActorCriticNetwork(self.state_shape, self.n_actions, config)

		n_env_remaining = n_env
		for i in range(n_threads):
			worker_n_env = math.ceil(n_env_remaining / (n_threads - i))
			self.workers.append(Worker(
				'worker%d' % i,
				Environment,
				self.config, 
				n_env=worker_n_env))
			
			n_env_remaining -= worker_n_env
			if n_env_remaining == 0:
				break

		mp.set_start_method('spawn')

	def init_log(self, directory, sess):
		self.file_writer = tf.summary.FileWriter(directory + 'log', sess.graph)

	def work(self, sess, episode, difficulty, parallel=False, test=False):
		for w in self.workers:
			w.env.update_difficulty(difficulty)

		if test:
			self.workers[0].work(sess, self.network, render=True)
			return

		if parallel:
			processes = []
			for worker in self.workers:
				print('process %s starting' % worker.name)
				p = mp.Process(target=worker.work, args=(sess, self.network,))
				p.start()
				processes.append(p)

			for p in processes:
				p.join()
		else:
			for worker in self.workers:
				worker.work(sess, self.network)

		state, action, reward, summary = [], [], [], []
		for worker in self.workers:
			thread_state, thread_action, thread_reward, thread_summary = worker.data
			state.append(thread_state)
			action.append(thread_action)
			reward.append(thread_reward)
			summary.append(thread_summary)

		self.summary['landing_success_rate'].append(np.mean([s['landing_success_rate'] for s in summary]))
		self.summary['landing_duration'].append(np.mean([s['landing_duration'] for s in summary]))

		feed_dict = {
			self.network.s: np.concatenate(state, axis=0),
			self.network.a: np.concatenate(action, axis=0),
			self.network.v_target: np.concatenate(reward, axis=0),
			self.network.mean_episode_reward: np.mean(np.concatenate(reward, axis=0)),
			self.network.landing_success_rate: self.summary['landing_success_rate'][-1],
			self.network.landing_duration: self.summary['landing_duration'][-1],
			self.network.difficulty : difficulty
		}
		self.network.update(feed_dict, sess)

		if self.file_writer:
			self.file_writer.add_summary(sess.run(self.network.summary, feed_dict=feed_dict), episode)
			self.file_writer.flush()

class Worker:
	def __init__(self, name, Environment, config, n_env=1):
		self.name = name
		# self.process = None
		self.config = config
		self.env = EnvironmentStack(
			Environment, 
			env_args={'initial_generator' : config['initial_generator']},
			n_env=n_env)
		self.state = None
		self.data = None
		self.summary = {
			'landing_success_rate' : 0.0,
			'landing_duration' : 0
		}

	def work(self, sess, network, render=False):
		if not self.state:
			self.state = self.env.reset()

		states, actions, rewards, dones, infos = [], [], [], [], []

		for _ in range(self.config['frames_per_episode']):
			a = network.choose_action(self.state, sess)
			a = self.manipulate_actions(self.state, a)
			s, r, d, i = self.env.step(a)

			states.append(np.copy(self.state))
			actions.append(a)
			rewards.append(r)
			dones.append(list(d))
			infos.append(list(i))
			self.state = list(s)

			if render:
				self.env.render()

		# get values for last state
		R = sess.run(network.v, {network.s: self.state})
		targets = []
		for r, d in zip(reversed(rewards), reversed(dones)):
			R = r + self.config['gamma'] * R * np.logical_not(d)
			targets.append(R)

		targets = np.asarray(targets[::-1]).swapaxes(1, 0).flatten()
		states = np.vstack(np.asarray(states, dtype=np.float32).swapaxes(1, 0))
		actions = np.asarray(actions, dtype=np.float32).swapaxes(1, 0).flatten()

		attempts = [i for i in np.concatenate(infos) if i['attempt_over']]
		if not len(attempts) == 0:
			self.summary['landing_success_rate'] = np.mean([1 if i['attempt_succesful'] else 0 for i in attempts])
			self.summary['landing_duration'] = np.mean([i['attempt_duration'] for i in attempts])

		self.data = (states, actions, targets, self.summary)

	def manipulate_actions(self, states, actions):
		for i, s in enumerate(states):
			if s[6] and s[7]: # no action if touching ground
				actions[i] = 0
		return actions
