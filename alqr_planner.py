"""
Class for planning with an augmented linear-quadratic regulator (aLQR)
on a given system and associated with a given cost field.

The system must be linearizable, time-invarient, and control affine:
xdot = f(x) + B(x)*u  with  jac(f)|x_i = A_i  such that
xdot = A_i*x + B(x_i)*u  is accurate near x_i.

"""

################################################# DEPENDENCIES

from __future__ import division
import numpy as np
import numpy.linalg as npl
from cost_field import Cost_Field

################################################# PRIMARY CLASS

class ALQR_Planner:
	"""
	Instances must be initialized with a dynamics function, 
	a linearize function, a cost_field, a planning horizon and resolution.

	The dynamics function must take the current state x and control effort
	u, and return the state derivative xdot using the nonlinear model.
	
	The linearize function must take the current state x and return the
	instantaneous linearized dynamics as a tuple of numpy.ndarrays (A, B)
	where xdot = Ax + Bu is accurate near the current x.

	The cost_field is an instance of the class Cost_Field. It prescribes
	how much you care about getting to the goal state, avoiding obstacles,
	keeping within kinematic constraints, applying excessive effort etc...

	The planning_horizon is how much time into the future each plan accounts
	for. The planning_resolution is the timestep used in planning simulations.

	The maximum iteration number for planning and convergence criteria can also
	be set. The convergence criteria is for the cost drop between iterations.

	Setting demo_plots to True will import matplotlib and plot all the
	trajectory iterations on the first call of update_plan. FOR DEMOS ONLY.

	"""
	def __init__(self, dynamics, linearize, cost_field,
				 planning_horizon, planning_resolution,
				 max_iter=100, eps_converge=0.005,
				 demo_plots = False):
		self.set_dynamics(dynamics, linearize)
		self.set_cost_field(cost_field)
		self.set_planning_params(planning_horizon, planning_resolution, max_iter, eps_converge)
		self.demo_plots = demo_plots
		if demo_plots:
			self.plot_setup()


	def update_plan(self, x0, warm_start_iter=None):
		"""
		Takes x0, the state to begin planning from.

		Updates the internally stored optimal effort sequence
		(u_seq) and expected state sequence (x_seq). The plan
		is also returned (x_seq, u_seq) but you should use the
		get_effort and get_state functions for interfacing.

		If you want to initialize the first u_seq guess based
		on the last u_seq, you can set warm_start_iter to the
		number of iterations into the last plan to start from.

		"""
		# Safety first
		x0 = np.array(x0, dtype=np.float64)

		# Initialize first u_seq guess
		if warm_start_iter is not None:
			if type(warm_start_iter) is int:
				self.u_seq = np.concatenate((self.u_seq[warm_start_iter:], np.zeros((warm_start_iter, self.ncontrols))))
			else:
				raise ValueError("Set warm_start_iter to the INTEGER number of iterations into the last plan to start from.")
		else:
			self.u_seq = np.zeros((self.N, self.ncontrols))

		# Begin Travis DeWolf's iLQR Implementation
		# https://studywolf.wordpress.com/2016/02/03/the-iterative-linear-quadratic-regulator-method/
		# ---

		lamb = 1.0
		sim_new_trajectory = True

		for ii in xrange(self.max_iter):

			if sim_new_trajectory == True:

				# For storing linearized dynamics (pretend f_ux is zero)
				f_x = np.zeros((self.N, self.nstates, self.nstates)) # df / dx
				f_u = np.zeros((self.N, self.nstates, self.ncontrols)) # df / du

				# For storing quadraticized cost function
				l = np.zeros((self.N, 1)) # immediate state cost
				l_x = np.zeros((self.N, self.nstates)) # dl / dx
				l_xx = np.zeros((self.N, self.nstates, self.nstates)) # d^2 l / dx^2
				l_u = np.zeros((self.N, self.ncontrols)) # dl / du
				l_uu = np.zeros((self.N, self.ncontrols, self.ncontrols)) # d^2 l / du^2
				l_ux = np.zeros((self.N, self.ncontrols, self.nstates)) # d^2 l / du / dx

				# For storing forward simulation
				self.x_seq = np.zeros((self.N, self.nstates))
				x = x0.copy()
				cost = 0

				# Forward integrate while also storing LQR approximation over sequence
				for i in xrange(self.N):
					self.x_seq[i] = x
					A, B = self.linearize(self.x_seq[i])
					f_x[i] = np.eye(self.nstates) + A * self.dt
					f_u[i] = B * self.dt
					l[i] = (self.cost_field.state_cost(self.x_seq[i]) + self.cost_field.effort_cost(self.u_seq[i])) * self.dt
					l_x[i] = self.cost_field.state_cost_gradient(self.x_seq[i]) * self.dt
					l_xx[i] = self.cost_field.state_cost_hessian(self.x_seq[i]) * self.dt
					l_u[i] = self.cost_field.effort_cost_gradient(self.u_seq[i]) * self.dt
					l_uu[i] = self.cost_field.effort_cost_hessian(self.u_seq[i]) * self.dt
					cost += l[i]
					x += self.dynamics(x, self.u_seq[i]) * self.dt

				# Copy for exit condition check
				oldcost = np.copy(cost)
 
			# Initialize Vs with final state cost and set up k, K
			V = l[-1].copy() # value function
			V_x = l_x[-1].copy() # dV / dx
			V_xx = l_xx[-1].copy() # d^2 V / dx^2
			k = np.zeros((self.N, self.ncontrols)) # feedforward modification
			K = np.zeros((self.N, self.ncontrols, self.nstates)) # feedback gain

			# Work backwards to solve for V, Q, k, and K
			for i in xrange(self.N-1, -1, -1):

				# 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
				Q_x = l_x[i] + np.dot(f_x[i].T, V_x)
				# 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
				Q_u = l_u[i] + np.dot(f_u[i].T, V_x)

				# NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
				# but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
				
				# 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
				Q_xx = l_xx[i] + np.dot(f_x[i].T, np.dot(V_xx, f_x[i])) 
				# 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
				Q_ux = l_ux[i] + np.dot(f_u[i].T, np.dot(V_xx, f_x[i]))
				# 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)
				Q_uu = l_uu[i] + np.dot(f_u[i].T, np.dot(V_xx, f_u[i]))

				# Calculate Q_uu^-1 with regularization term set by 
				# Levenberg-Marquardt heuristic (at end of this loop)
				Q_uu_evals, Q_uu_evecs = npl.eig(Q_uu)
				Q_uu_evals[Q_uu_evals < 0] = 0.0
				Q_uu_evals += lamb
				Q_uu_inv = np.dot(Q_uu_evecs, 
						np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

				# 5b) k = -np.dot(Q_uu^-1, Q_u)
				k[i] = -np.dot(Q_uu_inv, Q_u)
				# 5b) K = -np.dot(Q_uu^-1, Q_ux)
				K[i] = -np.dot(Q_uu_inv, Q_ux)

				# 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
				# 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
				V_x = Q_x - np.dot(K[i].T, np.dot(Q_uu, k[i]))
				# 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
				V_xx = Q_xx - np.dot(K[i].T, np.dot(Q_uu, K[i]))

			# Initialize new optimal trajectory estimate
			Unew = np.zeros((self.N, self.ncontrols))
			Xnew = np.zeros((self.N, self.nstates))
			xnew = x0.copy()
			costnew = 0

			# calculate the optimal change to the control trajectory 7a)
			for i in xrange(self.N):
				# use feedforward (k) and feedback (K) gain matrices 
				# calculated from our value function approximation
				# to take a stab at the optimal control sequence
				# (evaluate the new sequence along the way)
				Unew[i] = self.u_seq[i] + k[i] + np.dot(K[i], xnew - self.x_seq[i]) # 7b)
				xnew += self.dynamics(xnew, Unew[i]) * self.dt
				Xnew[i] = xnew
				costnew += (self.cost_field.state_cost(xnew) + self.cost_field.effort_cost(Unew[i])) * self.dt

			# Levenberg-Marquardt heuristic
			if costnew < cost: 
				# Decrease lambda (get closer to Newton's method)
				lamb /= self.lamb_factor
				# Update
				self.u_seq = np.copy(Unew)
				oldcost = np.copy(cost)
				cost = np.copy(costnew)
				# Do another rollout
				sim_new_trajectory = True 
				print("iteration = %d; Cost = %.4f;"%(ii, costnew) + 
				        " logLambda = %.1f"%np.log(lamb))
				# Check to see if update is small enough to exit
				if ii > 0 and ((abs(oldcost-cost)/cost) < self.eps_converge):
					print("Converged at iteration = %d; Cost = %.4f;"%(ii,costnew) + 
							" logLambda = %.1f"%np.log(lamb))
					break

			else: 
				# Skip next rollout
				sim_new_trajectory = False
				# Increase lambda (get closer to gradient descent)
				lamb *= self.lamb_factor
				if lamb > self.lamb_max: 
					print("lambda > max_lambda at iteration = %d;"%ii + 
						" Cost = %.4f; logLambda = %.1f"%(cost, 
														  np.log(lamb)))
					break

			if self.demo_plots:
				self.ax.plot(self.x_seq[:, 0] , self.x_seq[:, 1], color=np.random.rand(3,1))
		if self.demo_plots:
			graphic_robot = self.ax.add_patch(plt.Circle((x0[0], x0[1]), radius=0.02, fc='k'))
			plt.show()

		# ...and return if ya want it
		return (self.x_seq, self.u_seq)


	def get_effort(self, t):
		"""
		Returns the optimal effort at time t
		within the last generated plan.

		"""
		#<<< linearly interp to find u at t in t_seq
		pass


	def get_state(self, t):
		"""
		Returns the expected state at time t
		within the last generated plan.

		"""
		#<<< linearly interp to find x at t in t_seq
		pass


	def set_dynamics(self, dynamics, linearize):
		"""
		Use for scheduling different dynamics if needed.

		"""
		if hasattr(dynamics, '__call__'):
			self.dynamics = dynamics
		else:
			raise ValueError("Expected dynamics to be a function.")

		if hasattr(linearize, '__call__'):
			self.linearize = linearize
		else:
			raise ValueError("Expected linearize to be a function.")


	def set_cost_field(self, cost_field):
		"""
		Use for scheduling different behaviors if needed.

		"""
		if isinstance(cost_field, Cost_Field):
			self.cost_field = cost_field
			self.nstates = self.cost_field.nstates
			self.ncontrols = self.cost_field.ncontrols
		else:
			raise ValueError("Expected cost_field to be an instance of Cost_Field.")


	def set_planning_params(self, planning_horizon=None, planning_resolution=None, max_iter=None, eps_converge=None):
		"""
		Use for modifying the simulation duration and timestep used during planning,
		and maximum number of allowable iterations for converging. Parameters not
		given are not changed. When called, the current plan is reset.

		"""
		if max_iter is not None:
			self.max_iter = max_iter

		if planning_horizon is not None:
			self.T = planning_horizon

		if planning_resolution is not None:
			self.dt = planning_resolution

		if eps_converge is not None:
			self.eps_converge = eps_converge

		self.t_seq = np.arange(0, self.T, self.dt)
		self.N = len(self.t_seq)

		self.x_seq = np.zeros((self.N, self.nstates))
		self.u_seq = np.zeros((self.N, self.ncontrols))

		self.lamb_factor = 10
		self.lamb_max = 1000


	def plot_setup(self):
		"""
		Prepares iteration plot.

		"""
		global plt
		from matplotlib import pyplot as plt

		self.fig = plt.figure()
		self.fig.suptitle('Planning', fontsize=24)
		plt.axis('equal')

		self.ax = self.fig.add_subplot(1, 1, 1)
		self.ax.set_xlabel('- Position 1 +')
		self.ax.set_ylabel('- Position 2 +')
		self.ax.grid(True)

		radius = 0.02
		xlim = (0*1.1 - radius, 1*1.1 + radius)
		ylim = (0*1.1 - radius, 1*1.1 + radius)
		self.ax.set_xlim(xlim)
		self.ax.set_ylim(ylim)

		# Color map of cost function over position space, zero velocity
		if True:
			# Resolution
			dX, dY = 0.01, 0.01
			# Grid
			X, Y = np.mgrid[slice(xlim[0], xlim[1] + dX, dX),
			                slice(ylim[0], ylim[1] + dY, dY)]
			Jmap = np.zeros_like(X)
			# Evaluate cost field
			for i, xval in enumerate(X[:, 0]):
				for j, yval in enumerate(Y[0, :]):
					Jmap[i, j] = self.cost_field.state_cost([xval, yval, 0, 0])
					if Jmap[i, j] < 0:
						print "Negative cost! At ({0}, {1})".format(xval, yval)
			Jmap = Jmap[:-1, :-1]
			plt.pcolor(X, Y, Jmap, cmap='YlOrRd', vmin=np.min(Jmap), vmax=np.max(Jmap))
			plt.colorbar()

		graphic_goal = self.ax.add_patch(plt.Circle((self.cost_field.goal[0], self.cost_field.goal[1]), radius=radius, fc='g'))
		for p in self.cost_field.obstacle_positions:
			self.ax.add_patch(plt.Circle((p[0], p[1]), radius=radius, fc='r'))
