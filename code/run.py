#!/usr/bin/env python3
import sys
import os
import imp
import json
import datetime
import argparse
import numpy as np
import tensorflow as tf

from lander import Lander
from agent import Agent

def read_program_state(directory):
	if os.path.exists(directory + 'state.json'):
		program_state = json.load(open(directory + 'state.json'))
	else: # first run
		program_state = {
			'episode' : 0,
			'timestamp' : str(datetime.datetime.now()),
			'last_difficulty_change' : 0,
			'difficulty' : 0.1
		}
	return program_state

def save_program_state(directory, program_state):
	json.dump(program_state, open(directory + 'state.json', 'w'), indent=4)

def get_config(directory):
	config_path = directory + 'config.py'
	if not os.path.exists(config_path):
		raise ValueError('error: no configuration file found at %s' % config_path)

	module = imp.load_source('config', config_path)
	return getattr(module, 'config')

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('directory', help='directory containing config. (saves/<directory>/config.py)')
	parser.add_argument('-n', '--n_episodes', help='number of episodes to run', type=int)
	parser.add_argument('-t', '--test', help='test network without training', action='store_true')
	return parser.parse_args()

def init(directory, n_threads=1):
	# ----------------
	tf_config = tf.ConfigProto(
		inter_op_parallelism_threads=n_threads,
		intra_op_parallelism_threads=n_threads)
	tf_config.gpu_options.allocator_type = 'BFC'
	# ----------------

	sess = tf.Session(config=tf_config)

	model_directory = directory + 'model/'
	if not os.path.exists(model_directory):
		os.makedirs(model_directory)

	checkpoint_path = tf.train.latest_checkpoint(model_directory)
	if checkpoint_path is None:
		sess.run(tf.global_variables_initializer())
		print('model initialized.')
	else:
		saver = tf.train.Saver()
		saver.restore(sess, checkpoint_path)
		print('model loaded.')
	return sess

def save(directory):
	model_directory = directory + 'model/'
	saver = tf.train.Saver()
	saver.save(sess, model_directory + 'model')
	print('model saved.')

if __name__ == '__main__':
	args = get_args()
	directory = './saves/%s/' % args.directory
	if not os.path.exists(directory):
		raise ValueError('error: directory %s does not exist' % directory)

	n_threads = 1
	config = get_config(directory)
	program_state = read_program_state(directory)
	agent = Agent(Lander, config, n_env=config['n_env'] if not args.test else 1, n_threads=n_threads)
	sess = init(directory)
	agent.init_log(directory, sess)

	target_episode = args.n_episodes and program_state['episode'] + args.n_episodes 
	while program_state['episode'] < target_episode if args.n_episodes is not None else True:
		try:
			config['adjust_difficulty'](program_state, agent.summary)
			agent.work(sess, program_state['episode'], program_state['difficulty'], test=True if args.test else False)
			program_state['episode'] += 1
			if args.n_episodes:
				print('end of episode %d (%d left)' % (program_state['episode'], target_episode - program_state['episode']))
			elif not args.test:
				print('end of episode %d' % program_state['episode'])
		except KeyboardInterrupt:
			break

	if not args.test:
		save(directory)
		program_state['timestamp'] = str(datetime.datetime.now())
		save_program_state(directory, program_state)

