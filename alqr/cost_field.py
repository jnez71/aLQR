"""
A class for expressing and modifying the cost field
associated with your alqr planner.

It prescribes how much you care about getting to the
goal state, avoiding obstacles, keeping within actuator
constraints, avoiding excessive effort etc...

The state space must be ordered as:
[position_1, position_2, ..., position_n, velocity_1, velocity_2, ..., velocity_n]

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
from scipy.optimize import approx_fprime

################################################# PRIMARY CLASS

class Cost_Field:
	"""
	Instances must be initialized with:
	---
	nstates: the dimensionality of the state space
	ncontrols: the dimensionality of the effort space
	goal0: the initial goal state
	position_weight: scalar in cost function (position_weight * position_error^2)
	velocity_weight: scalar in cost function (velocity_weight * velocity_error^2)
	obstacle_weight: scalar in cost function (obstacle_weight * obstacle_nearness^2)
	effort_weight: scalar in cost function (effort_weight * effort^2)
	umin and umax: arrays of minimum and maximum allowable actuator efforts (default: no limits)
	strictness: scalar for how strict the effort limits are... don't be too strict or poor convergence
	arb costs: functions that add onto the state and effort cost calculations (default: no added cost)
			   (must take state or effort as argument respectively and return a scalar cost)

	"""
	def __init__(self, nstates, ncontrols, goal0,
				 position_weight, velocity_weight,
				 obstacle_weight, effort_weight,
				 umin=None, umax=None, strictness=100,
				 arb_state_cost=None, arb_effort_cost=None):
		# Dimensionality
		self.nstates = int(nstates)
		self.ncontrols = int(ncontrols)

		# Get your goals in order and your priorities straight
		self.set_goal(goal0)
		self.set_weights(position_weight, velocity_weight, obstacle_weight, effort_weight)
		self.reset_obstacles()

		# Initialize and then set limits
		self.umin = -np.inf * np.ones(ncontrols)
		self.umax = np.inf * np.ones(ncontrols)
		self.set_constraints(umin, umax, strictness)

		# Initialize and then store arbitrary cost functions
		self.arb_state_cost = lambda x: 0
		self.arb_effort_cost = lambda u: 0
		self.set_arb_costs(arb_state_cost, arb_effort_cost)

		# Finite difference delta size
		self.eps = (np.finfo(float).eps)**0.5


	def state_cost(self, x):
		"""
		Computes the instantaneous cost for being at state x.

		"""
		# Distance from goal
		goal_error = self.goal - x
		# Upwards quadratic, centered at goal
		c_goal = (self.goal_weight * goal_error).dot(goal_error)
		# Find which obstacles we are in the region of influence of
		if len(self.obstacle_ids):
			distances = np.array([npl.norm(self.obstacle_positions - x[:int(self.nstates/2)], axis=1)]).T
			contributors = (distances <= self.obstacle_rois)
			# Apply cosine hump and arbitrary user-defined cost
			c_obs = np.sum((self.obstacle_weight*(np.cos((distances/self.obstacle_rois)*np.pi)+1)/2)[contributors])
		else:
			c_obs = 0
		return c_goal + c_obs + self.arb_state_cost(x)


	def effort_cost(self, u):
		"""
		Computes the instantaneous cost for applying effort u.

		"""
		# Upwards quadratic, centered at zero effort
		c = self.effort_weight * u.dot(u)
		# Quadratically increase cost for leaving effort bounds
		for i, eff in enumerate(u):
			if eff >= self.umax[i]:
				c = c + self.strictness*(eff - self.umax[i])**2
			elif eff <= self.umin[i]:
				c = c + self.strictness*(eff - self.umin[i])**2
		return c + self.arb_effort_cost(u)


	def state_cost_gradient(self, x):
		"""
		Computes the gradient of the state-space cost field at state x.

		"""
		return approx_fprime(x, self.state_cost, self.eps)


	def effort_cost_gradient(self, u):
		"""
		Computes the gradient of the effort-space cost field at actuation u.

		"""
		return approx_fprime(u, self.effort_cost, self.eps)


	def state_cost_hessian(self, x):
		"""
		Computes the jacobian of the gradient of the cost field
		with respect to the state. One might approximate this as
		np.outer(gradient, gradient), but not me, because reasons.

		"""
		Q = np.eye(self.nstates)
		grad = self.state_cost_gradient(x)
		for i in xrange(self.nstates):
			x_perturbed = x.astype(np.float64)
			x_perturbed[i] = x_perturbed[i] + self.eps
			Q[:self.nstates, i] = (self.state_cost_gradient(x_perturbed) - grad)
		return Q / self.eps


	def effort_cost_hessian(self, u):
		"""
		Computes the jacobian of the gradient of the cost field
		with respect to the effort. One might approximate this as
		np.outer(gradient, gradient), but not me, because reasons.

		"""
		R = np.eye(self.ncontrols)
		grad = self.effort_cost_gradient(u)
		for i in xrange(self.ncontrols):
			u_perturbed = u.astype(np.float64)
			u_perturbed[i] = u_perturbed[i] + self.eps
			R[:self.ncontrols, i] = (self.effort_cost_gradient(u_perturbed) - grad)
		return R / self.eps


	def add_obstacle(self, name, position, influence_radius):
		"""
		Obstacles must be given a name (string), a central
		position (array), and an influence radius (float).

		"""
		if len(self.obstacle_ids) == 0:
			self.obstacle_ids = np.array([name])
			self.obstacle_positions = np.array(position, dtype=np.float64)
			self.obstacle_rois = np.array([influence_radius], dtype=np.float64)
		else:
			self.obstacle_ids = np.vstack((self.obstacle_ids, name))
			self.obstacle_positions = np.vstack((self.obstacle_positions, position))
			self.obstacle_rois = np.vstack((self.obstacle_rois, influence_radius))


	def remove_obstacle(self, name):
		"""
		Remove an obstacle by providing its name (string).

		"""
		keepers = (self.obstacle_ids != name).flatten()
		self.obstacle_ids = self.obstacle_ids[keepers]
		self.obstacle_positions = self.obstacle_positions[keepers]
		self.obstacle_rois = self.obstacle_rois[keepers]


	def reset_obstacles(self):
		"""
		Clears all obstacles.

		"""
		self.obstacle_ids = np.array([])
		self.obstacle_positions = np.array([])
		self.obstacle_rois = np.array([])


	def set_goal(self, goal):
		"""
		Use to modify the overall goal state ("waypoint").

		"""
		if len(goal) == self.nstates:
			self.goal = np.array(goal, dtype=np.float64)
		else:
			raise ValueError("The goal must be a state vector (with nstates elements).")


	def set_weights(self, position_weight=None, velocity_weight=None, obstacle_weight=None, effort_weight=None):
		"""
		Use to modify the scalar weights for the various influences on behavior.
		Weights that aren't given aren't changed.

		"""
		if position_weight is not None:
			if type(position_weight) in [int, float]:
				self.position_weight = float(position_weight)
			else:
				raise ValueError("The position_weight must be a scalar.")

		if velocity_weight is not None:
			if type(velocity_weight) in [int, float]:
				self.velocity_weight = float(velocity_weight)
			else:
				raise ValueError("The velocity_weight must be a scalar.")

		self.goal_weight = np.concatenate((
										   self.position_weight * np.ones(int(self.nstates/2)),
										   self.velocity_weight * np.ones(int(self.nstates/2))
										 ))

		if obstacle_weight is not None:
			if type(obstacle_weight) in [int, float]:
				self.obstacle_weight = float(obstacle_weight)
			else:
				raise ValueError("The obstacle_weight must be a scalar.")

		if effort_weight is not None:
			if type(effort_weight) in [int, float]:
				self.effort_weight = float(effort_weight)
			else:
				raise ValueError("The effort_weight must be a scalar")


	def set_constraints(self, umin=None, umax=None, strictness=None):
		"""
		Use to modify effort limits.
		Limits that aren't given aren't changed.

		"""
		if umin is not None:
			if len(umin) == self.ncontrols:
				self.umin = np.array(umin, dtype=np.float64)
			else:
				raise ValueError("Actuator constraint umin must have same length as the number of controls.")

		if umax is not None:
			if len(umax) == self.ncontrols:
				self.umax = np.array(umax, dtype=np.float64)
			else:
				raise ValueError("Actuator constraint umax must have same length as the number of controls.")

		if strictness is not None:
			self.strictness = strictness


	def set_arb_costs(self, arb_state_cost=None, arb_effort_cost=None):
		"""
		Use to modify arbitrary additions to the cost field.
		Arguments not given are not changed.

		"""
		if arb_state_cost is not None:
			if hasattr(arb_state_cost, '__call__'):
				self.arb_state_cost = arb_state_cost
			else:
				raise ValueError("Expected arb_state_cost to be a function.")

		if arb_effort_cost is not None:
			if hasattr(arb_effort_cost, '__call__'):
				self.arb_effort_cost = arb_effort_cost
			else:
				raise ValueError("Expected arb_effort_cost to be a function.")
