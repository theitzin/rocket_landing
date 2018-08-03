import sys
import numpy as np
import tensorflow as tf
import random

def network_init(input_data, n_actions):
	w_init = tf.orthogonal_initializer()

	hidden = tf.layers.dense(input_data, 512, 
		activation=tf.nn.relu, 
		kernel_initializer=w_init, 
		name='ac_h1')
	hidden = tf.layers.dense(hidden, 256, 
		activation=tf.nn.relu, 
		kernel_initializer=w_init, 
		name='ac_h2')

	hidden_a = tf.layers.dense(hidden, 128, 
		activation=tf.nn.relu, 
		kernel_initializer=w_init,
		name='ac_ha')
	a = tf.layers.dense(hidden_a, n_actions, activation=tf.nn.softmax, name='ac_a_out')

	hidden_v = tf.layers.dense(hidden, 128, 
		activation=tf.nn.relu, 
		kernel_initializer=w_init,
		name='ac_hv')
	v = tf.layers.dense(hidden_v, 1, name='ac_v_out')[:,0]

	return (a, v)

def get_optimizer():
	return tf.train.RMSPropOptimizer(
		name='RMSPropAC', 
		learning_rate=3e-5, 
		decay=0.9, 
		epsilon=1e-10)

def initial_generator(gd): # gd: global difficulty, d is individual
	generator_lambdas = {
		'pos_x' : lambda d: gd * (random.choice((-1, 1)) * d * 0.8),
		'pos_y' : lambda d: min(gd, 1) * (0.9 + d * 0.1),
		'vel' : lambda d: 2000 + gd * (3000 + d * 2000),
		'dir' : lambda d: gd * (random.choice((-1, 1)) * d * 0.5),
		'vdir' : lambda d: gd * (random.choice((-1, 1)) * d * 5.0),
		'ship_width' : lambda d: 100 + (1 - min(gd, 1)) * 200
	}

	n = len(generator_lambdas)
	rnd = np.random.uniform(size=n)
	rnd = rnd / sum(rnd) * n / 2 # random variables uniformly distributed in [0, 1) but summing up to n
	rnd = np.clip(rnd, 0, 1)
	return {k : generator_lambdas[k](rnd[i]) for i, k in enumerate(generator_lambdas)}

def adjust_difficulty(program_state, summary):
	if len(summary['landing_success_rate']) < summary['landing_success_rate'].maxlen:
		return

	if program_state['episode'] < program_state['last_difficulty_change'] + summary['landing_success_rate'].maxlen:
		return

	if np.mean(summary['landing_success_rate']) > 0.8:
		program_state['difficulty'] = min(2, program_state['difficulty'] + 0.01)
		program_state['last_difficulty_change'] = program_state['episode']

	if np.mean(summary['landing_success_rate']) < 0.7:
		program_state['difficulty'] = max(0.1, program_state['difficulty'] - 0.001)
		program_state['last_difficulty_change'] = program_state['episode']

config = {
	'network' : network_init,
	'optimizer' : get_optimizer(),
	'n_env' : 100,
	'frames_per_episode' : 20,
	#'render' : False,

	'gamma' : 0.995,
	'entropy_beta' : 0.05,
	'max_grad_norm' : 0.5,

	'initial_generator' : initial_generator,
	'adjust_difficulty' : adjust_difficulty
}
