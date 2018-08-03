import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

from water import Water

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

ROCKET = {
	'main_engine_power' : 20.0,
	'side_engine_power' :  1.5,

	'vertices' : [(-6,+130), (-6,-10), (+6,-10), (+6,+130)],
	'leg_away' : 8,
	'leg_down' : 18,
	'leg_w' : 2,
	'leg_h' : 10,
	'leg_spring_torque' : 55,

	'side_engine_height' : 90.0,
	'side_engine_away' : 12.0,

	'body_color1' : (1.0, 1.0, 1.0),
	'body_color2' : (0.0, 0.0, 0.0),
	'leg_color1' : (0.2, 0.2, 0.2),
	'leg_color2' : (0.0, 0.0, 0.0)
}

SHIP = {
	'vertices' : lambda x: [(-x, 0), (x, 0), (x, -10), ((x-20), -20), (-(x-20), -20), (-x, -10)],

	'color1' : (0.5,0.5,0.5),
	'color2' : (0.3,0.3,0.3)
}

# environment data, vertices in relative coordinates
ENV = {
	'sky_vertices' : [(0, 0), (0, 1), (1, 1), (1, 0)],
	'water_level' : 1/21,
	'water_nodes_n' : 100,

	'sky_color' : (165/255, 201/255, 255/255),
	'water_color' : (59/255, 127/255, 229/255)
}

VIEWPORT_W = 500
VIEWPORT_H = 1000

BOUNDS_SCALE = 10.0
W = VIEWPORT_W/BOUNDS_SCALE
H = VIEWPORT_H/BOUNDS_SCALE

class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env
	def BeginContact(self, contact):
		if self.env.lander==contact.fixtureA.body or self.env.lander==contact.fixtureB.body:
			self.env.game_over = True
		for i in range(2):
			if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
				self.env.legs[i].ground_contact = True
	def EndContact(self, contact):
		for i in range(2):
			if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
				self.env.legs[i].ground_contact = False

class Lander(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : FPS
	}

	continuous = False

	def __init__(self, initial_generator):
		self.seed()
		self.viewer = None

		self.world = Box2D.b2World()
		self.ship = None
		self.lander = None
		self.particles = []
		self.water = Water([0, 1], force_const_volume=True)

		self.prev_reward = None

		self.origin = (W / 2, H / 20)

		# dummy initial variables, will be set in update_initials
		self.initial_pos_x = self.origin[0]
		self.initial_pos_y = self.origin[1] + H / 2
		self.initial_vel = 0
		self.initial_dir = 0
		self.initial_vdir = 0
		self.ship_width = 100
		# non constant hyperparameters
		self.initial_generator = initial_generator
		self.difficulty = 0.1

		self.steps = 0

		high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
		self.observation_space = spaces.Box(-high, high)

		if self.continuous:
			# Action is two floats [main engine, left-right engines].
			# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
			# Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
			self.action_space = spaces.Box(-1, +1, (2,))
		else:
			# Nop, fire left engine, main engine, right engine
			self.action_space = spaces.Discrete(4)

		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _destroy(self):
		if not self.ship: return
		self.world.contactListener = None
		self._clean_particles(True)
		self.world.DestroyBody(self.ship)
		self.ship = None
		self.world.DestroyBody(self.lander)
		self.lander = None
		self.world.DestroyBody(self.legs[0])
		self.world.DestroyBody(self.legs[1])

	def reset(self):
		self.update_initials()

		self._destroy()
		self.world.contactListener_keepref = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_keepref
		self.game_over = False
		self.prev_shaping = None

		# ship
		self.ship = self.world.CreateStaticBody(
			position=self.origin,
			shapes=polygonShape(
				vertices=[(x/SCALE,y/SCALE) for x,y in SHIP['vertices'](self.ship_width)]))

		self.ship.color1 = SHIP['color1']
		self.ship.color2 = SHIP['color2']

		# lander
		self.lander = self.world.CreateDynamicBody(
			position = (self.initial_pos_x, self.initial_pos_y),
			angle=self.initial_dir,
			fixtures = fixtureDef(
				shape=polygonShape(
					vertices=[(x/SCALE,y/SCALE) for x,y in ROCKET['vertices']]),
				density=5.0,
				friction=0.1,
				categoryBits=0x0010,
				maskBits=0x001,  # collide only with ground
				restitution=0.0) # 0.99 bouncy
				)
		self.lander.color1 = ROCKET['body_color1']
		self.lander.color2 = ROCKET['body_color2']
		self.lander.ApplyForceToCenter(
			(self.initial_vel * math.sin(self.initial_dir),
			-self.initial_vel * math.cos(self.initial_dir)), True)
		self.lander.ApplyAngularImpulse(self.initial_vdir, True)

		self.legs = []
		for i in [-1,+1]:
			leg = self.world.CreateDynamicBody(
				position = (self.initial_pos_x - i*ROCKET['leg_away']/SCALE, self.initial_pos_y),
				angle = self.initial_dir + i*0.05,
				fixtures = fixtureDef(
					shape=polygonShape(box=(ROCKET['leg_w']/SCALE, ROCKET['leg_h']/SCALE)),
					density=1.0,
					restitution=0.0,
					categoryBits=0x0020,
					maskBits=0x001)
				)
			leg.ground_contact = False
			leg.color1 = ROCKET['leg_color1']
			leg.color2 = ROCKET['leg_color2']
			rjd = revoluteJointDef(
				bodyA=self.lander,
				bodyB=leg,
				localAnchorA=(0, 0),
				localAnchorB=(i*ROCKET['leg_away']/SCALE, ROCKET['leg_down']/SCALE),
				enableMotor=True,
				enableLimit=True,
				maxMotorTorque=ROCKET['leg_spring_torque'],
				motorSpeed=+0.3*i  # low enough not to jump back into the sky
				)
			if i==-1:
				rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
				rjd.upperAngle = +0.9
			else:
				rjd.lowerAngle = -0.9
				rjd.upperAngle = -0.9 + 0.5
			leg.joint = self.world.CreateJoint(rjd)
			self.legs.append(leg)

		self.drawlist = [self.lander, self.ship] + self.legs

		return self.step(np.array([0,0]) if self.continuous else 0)[0]

	def _create_particle(self, mass, x, y, ttl):
		p = self.world.CreateDynamicBody(
			position = (x,y),
			angle=0.0,
			fixtures = fixtureDef(
				shape=circleShape(radius=5/SCALE, pos=(0,0)),
				density=mass,
				friction=0.1,
				categoryBits=0x0100,
				maskBits=0x001,  # collide only with ground
				restitution=0.3)
				)
		p.ttl = ttl
		self.particles.append(p)
		self._clean_particles(False)
		return p

	def _clean_particles(self, all):
		while self.particles and (all or self.particles[0].ttl<0):
			self.world.DestroyBody(self.particles.pop(0))

	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

		# Engines
		tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
		side = (-tip[1], tip[0]);
		dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

		m_power = 0.0
		if (self.continuous and action[0] > 0.0) or (not self.continuous and action==2):
			# Main engine
			if self.continuous:
				m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
				assert m_power>=0.5 and m_power <= 1.0
			else:
				m_power = 1.0
			ox =  tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
			oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
			impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
			p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], m_power)    # particles are just a decoration, 3.5 is here to make particle speed adequate
			p.ApplyLinearImpulse(           ( ox*ROCKET['main_engine_power']*m_power,  oy*ROCKET['main_engine_power']*m_power), impulse_pos, True)
			self.lander.ApplyLinearImpulse( (-ox*ROCKET['main_engine_power']*m_power, -oy*ROCKET['main_engine_power']*m_power), impulse_pos, True)

		s_power = 0.0
		if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1,3]):
			# Orientation engines
			if self.continuous:
				direction = np.sign(action[1])
				s_power = np.clip(np.abs(action[1]), 0.5,1.0)
				assert s_power>=0.5 and s_power <= 1.0
			else:
				direction = action-2
				s_power = 1.0
			ox =  tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*ROCKET['side_engine_away']/SCALE)
			oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*ROCKET['side_engine_away']/SCALE)
			impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*ROCKET['side_engine_height']/SCALE)
			p = self._create_particle(0.14, impulse_pos[0], impulse_pos[1], s_power)
			p.ApplyLinearImpulse(           ( ox*ROCKET['side_engine_power']*s_power,  oy*ROCKET['side_engine_power']*s_power), impulse_pos, True)
			self.lander.ApplyLinearImpulse( (-ox*ROCKET['side_engine_power']*s_power, -oy*ROCKET['side_engine_power']*s_power), impulse_pos, True)

		self.world.Step(1.0/FPS, 6*30, 2*30)

		pos = self.lander.position
		vel = self.lander.linearVelocity
		state = [
			(pos.x - W/2) / (W/2),
			(pos.y - (self.origin[1]+ROCKET['leg_down']/SCALE)) / (W/2),
			vel.x*(W/2)/FPS,
			vel.y*(H/2)/FPS,
			3*self.lander.angle,
			100.0*self.lander.angularVelocity/FPS,
			1.0 if self.legs[0].ground_contact else 0.0,
			1.0 if self.legs[1].ground_contact else 0.0
			]
		assert len(state)==8

		reward = 0
		shaping = \
			- 10*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
			- 10*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
			- 30*abs(state[4]) + 1*state[6] + 1*state[7]   # And one point for legs contact, the idea is if you
															  # lose contact again after landing, you get negative reward
		if self.prev_shaping is not None:
			reward = shaping - self.prev_shaping
		self.prev_shaping = shaping

		reward -= m_power*0.3  # less fuel spent is better, about -30 for heuristic landing
		reward -= s_power*0.03

		done = False
		success = False
		if self.game_over or abs(state[0]) >= 1.5 or state[1] <= -0.05:
			done   = True
			reward = -100
		if not self.lander.awake:
			done   = True
			reward = +500
			success = True

		self.steps += 1

		info = {
			'attempt_over' : done,
			'attempt_succesful' : success,
			'attempt_duration' : self.steps
		}

		if done:
			state = self.reset()
			self.steps = 0

		return np.array(state), reward, done, info

	def render(self, mode='human'):
		from gym.envs.classic_control import rendering

		if self.viewer is None:
			self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
			self.viewer.set_bounds(0, W, 0, H)

			def key_press(key, mod):
				if key==0xff1b: # Escape
					self.close() # close window
					sys.exit()
			self.unwrapped.viewer.window.on_key_press = key_press

		# environment
		self.env_polys = []
		self.env_colors = []
		self.env_polys.append([(x*W, y*H) for x,y in ENV['sky_vertices']])
		self.env_colors.append(ENV['sky_color'])

		water_x, water_y = self.water.step()

		scale = 0.02
		water_poly = [(W, 0), (0, 0)] + list(zip(W*water_x, H*(ENV['water_level'] + scale*water_y))) + [(W, 0)]
		self.env_polys.append(water_poly)
		self.env_colors.append(ENV['water_color'])

		for i, p in enumerate(self.env_polys):
			self.viewer.draw_polygon(p, color=self.env_colors[i])


		# particles and lander
		for obj in self.particles:
			obj.ttl -= 0.15
			obj.color1 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))
			obj.color2 = (max(0.2,0.2+obj.ttl), max(0.2,0.5*obj.ttl), max(0.2,0.5*obj.ttl))

		self._clean_particles(False)

		for obj in self.particles + self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				if type(f.shape) is circleShape:
					t = rendering.Transform(translation=trans*f.shape.pos)
					self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
					self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=1).add_attr(t)
				else:
					path = [trans*v for v in f.shape.vertices]
					self.viewer.draw_polygon(path, color=obj.color1)
					path.append(path[0])
					self.viewer.draw_polyline(path, color=obj.color2, linewidth=1)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def close(self):
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None

	def update_initials(self):
		if not self.initial_generator:
			return

		difficulties = self.initial_generator(self.difficulty)

		self.initial_pos_x = (difficulties['pos_x'] + 1) * W / 2
		self.initial_pos_y = self.origin[1] + difficulties['pos_y'] * (H - self.origin[1])
		self.initial_vel = difficulties['vel']
		self.initial_dir = difficulties['dir']
		self.initial_vdir = difficulties['vdir']

		self.ship_width = difficulties['ship_width']

class LanderContinuous(Lander):
	continuous = True
