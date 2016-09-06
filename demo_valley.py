from __future__ import division
import numpy as np
import numpy.linalg as npl

from matplotlib import pyplot as plt
import matplotlib.animation as ani

import alqr


# Basic linear particle
nstates = 4
ncontrols = 2
drag = 3

def linearize(x):
	A = np.array([
				  [ 0, 0,     1,     0],
				  [ 0, 0,     0,     1],
				  [ 0, 0, -drag,     0],
				  [ 0, 0,     0, -drag]
				])
	B = np.array([
				  [0, 0],
				  [0, 0],
				  [1, 0],
				  [0, 1]
				])
	return (A, B)

def dynamics(x, u):
	# Truly linear system!
	A, B = linearize(x)
	return A.dot(x) + B.dot(u)


# Create arbitrary state cost function that produces desired valley path
goal = [1, 0.5, 0, 0]
valley_intensity = 100
def valley(x):
	x = np.array(x)
	goal_error = goal - x
	c = 3 * np.sum(goal_error**4)
	upper_limit = 0.4*np.sin(4*np.pi * x[0]) + 0.01 + 0.5
	lower_limit = 0.4*np.sin(4*np.pi * x[0]) - 0.01 + 0.5
	if x[1] >= upper_limit:
		c = c + valley_intensity*(x[1] - upper_limit)**2
	elif x[1] <= lower_limit:
		c = c + valley_intensity*(x[1]- lower_limit)**2
	else:
		c = c + 0
	return c


# Set up a cost field
cost_field = alqr.Cost_Field(nstates, ncontrols, goal,
							 position_weight=0, velocity_weight=0,
							 obstacle_weight=0, effort_weight=0.05,
							 arb_state_cost=valley)

# Associate an alqr planner
planning_horizon = 20  # s
planning_resolution = 0.01  # s
planner = alqr.Planner(dynamics, linearize, cost_field,
					   planning_horizon, planning_resolution,
					   demo_plots=True)


# Initial condition and time
x = [0, 0.6, 0, 0]
dt = planning_resolution  # convenient to use in sim testing too
t_arr = np.arange(0, planning_horizon, dt)
framerate = 30
show_cost_field = True

# Plan a path from these initial conditions
planner.update_plan(x)


# Preallocate results memory
x_history = np.zeros((len(t_arr), nstates))
goal_history = np.zeros((len(t_arr), nstates))
u_history = np.zeros((len(t_arr), ncontrols))
c_history = np.zeros(len(t_arr))


# Integrate dynamics
for i, t in enumerate(t_arr):

	# Planner's decision
	u = planner.get_effort(t)

	# Record this instant
	x_history[i, :] = x
	goal_history[i, :] = goal
	u_history[i, :] = u
	c_history[i] = cost_field.state_cost(x)

	# First-order integrate
	xdot = dynamics(x, u)
	x = x + xdot*dt


# Plot results
fig1 = plt.figure()
fig1.suptitle('Results', fontsize=20)

ax1 = fig1.add_subplot(2, 3, 1)
ax1.set_ylabel('Position 1', fontsize=16)
ax1.plot(t_arr, x_history[:, 0], 'k',
		 t_arr, goal_history[:, 0], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 2)
ax1.set_ylabel('Position 2', fontsize=16)
ax1.plot(t_arr, x_history[:, 1], 'k',
		 t_arr, goal_history[:, 1], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 3)
ax1.set_ylabel('Efforts', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
		 t_arr, u_history[:, 1], 'g')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 4)
ax1.set_ylabel('Velocity 1', fontsize=16)
ax1.plot(t_arr, x_history[:, 2], 'k',
		 t_arr, goal_history[:, 2], 'g--')
ax1.set_xlabel('Time')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 5)
ax1.set_ylabel('Velocity 2', fontsize=16)
ax1.plot(t_arr, x_history[:, 3], 'k',
		 t_arr, goal_history[:, 3], 'g--')
ax1.set_xlabel('Time')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 3, 6)
ax1.set_ylabel('State Cost', fontsize=16)
ax1.plot(t_arr, c_history, 'k')
ax1.grid(True)
ax1.set_xlabel('Time')

print("\nClose the plot window to continue to animation.")
plt.show()


# Animation
fig2 = plt.figure()
fig2.suptitle('Evolution', fontsize=24)
plt.axis('equal')

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('- Position 1 +')
ax2.set_ylabel('- Position 2 +')
ax2.grid(True)

radius = 0.02
xlim = (min(np.concatenate((x_history[:, 0], [0])))*1.1 - radius, max(np.concatenate((x_history[:, 0], [1])))*1.1 + radius)
ylim = (min(np.concatenate((x_history[:, 1], [0])))*1.1 - radius, max(np.concatenate((x_history[:, 1], [1])))*1.1 + radius)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

# (color map of cost function over position space, zero velocity)
if show_cost_field:
	# resolution
	dX, dY = 0.01, 0.01
	# grid
	X, Y = np.mgrid[slice(xlim[0], xlim[1] + dX, dX),
	                slice(ylim[0], ylim[1] + dY, dY)]
	Jmap = np.zeros_like(X)
	# evaluate cost field
	for i, xval in enumerate(X[:, 0]):
		for j, yval in enumerate(Y[0, :]):
			Jmap[i, j] = cost_field.state_cost([xval, yval, 0, 0])
			if Jmap[i, j] < 0:
				print "Negative cost! At ({0}, {1})".format(xval, yval)
	Jmap = Jmap[:-1, :-1]
	plt.pcolor(X, Y, Jmap, cmap='YlOrRd', vmin=np.min(Jmap), vmax=np.max(Jmap))
	plt.colorbar()

graphic_robot = ax2.add_patch(plt.Circle((x_history[0, 0], x_history[0, 1]), radius=radius, fc='k'))
graphic_goal = ax2.add_patch(plt.Circle((goal_history[0, 0], goal_history[0, 1]), radius=radius, fc='g'))
for p in cost_field.obstacle_positions:
	ax2.add_patch(plt.Circle((p[0], p[1]), radius=radius, fc='r'))

def ani_update(arg, ii=[0]):

	i = ii[0]  # don't ask...

	if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
		fig2.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

	graphic_robot.center = ((x_history[i, 0], x_history[i, 1]))
	
	ii[0] += int(1 / (dt * framerate))
	if ii[0] >= len(t_arr):
		print("Resetting animation!")
		ii[0] = 0

	return [graphic_robot]

# Run animation
print("\nStarting animation. \nBlack: robot \nRed: obstacles \nGreen: goal \nHeat Map: state cost\n")
animation = ani.FuncAnimation(fig2, func=ani_update, interval=dt*1000)
plt.show()
