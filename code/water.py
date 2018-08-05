#!/usr/bin/env python3
import time
import numpy as np
from scipy import sparse
from collections import deque

class Water:
	def __init__(self, domain, **args):
		self.config = {
			'h_t' : 1e-2,
			'h_x' : 5e-3,
			'c' : 0.05,
			'inertia' : 1e-2,
			'force_const_volume' : False
		}
		self.config.update(args)
		self.scal = self.config['c']**2 * self.config['h_t']**2 / self.config['h_x']**2

		self.domain = domain
		self.x_data = np.arange(domain[0], domain[1] + 1e-10, self.config['h_x'])
		self.y_data = []
		self.n = len(self.x_data)

		self.wave = deque(maxlen=2)
		wave0 = np.zeros(self.n)

		# initial condition
		self.wave.append(wave0)
		self.wave.append(wave0)

		self.A = -2 * sparse.eye(self.n) + sparse.eye(self.n, k=1) + sparse.eye(self.n, k=-1)
		# reflecting
		self.A[0, 0] = -1
		self.A[-1, -1] = -1

	def step(self):
		# perfect transparent bc: u_x = u_t at x=0, u_x = -u_t at x=1
		# factorize wave into incoming and outgoing part 
		# u_tt - u_xx = (∂_t - ∂_x)(∂_t + ∂_x)u => first part is incoming, second outgoing
		BC = np.zeros(self.n)
		BC[0] = self.wave[1][0] - self.wave[0][0] # -u_t(0)
		BC[-1] = self.wave[1][-1] - self.wave[0][-1] # -u_t(1)

		step = self.scal * (self.A.dot(self.wave[1]) - BC) + 2*self.wave[1] - self.wave[0]
		self.y_data = (1 - self.config['inertia'])*step + self.config['inertia']*self.wave[1] # artificial inertia

		# add drops falling into the water
		if np.random.uniform(0, 1) < 1e-1:
			self.y_data += self.drop(np.random.uniform(*self.domain))

		if self.config['force_const_volume']:
			self.y_data -= sum(self.y_data) / len(self.y_data)

		self.wave.append(self.y_data)
		return (self.x_data, self.y_data)

	# function with mass 0
	# scaled gauss bubble second derivative
	def drop(self, x_drop, size=(1, 1)):
		x_scale = 2e-2 * size[0]
		y_scale = 2e-3 * size[1]
		y_drop = np.array([-(4*x**2 - 2)*np.exp(-x**2) for x in (self.x_data - x_drop)/x_scale]) * y_scale
		return y_drop

