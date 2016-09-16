from __future__ import division
import numpy as np
import numpy.linalg as npl

from matplotlib import pyplot as plt
import matplotlib.animation as ani

import alqr

################################################# PHYSICAL PARAMETERS

# Boat inertia and center of gravity
m = 50  # kg
Iz = 100  # kg*m**2
xg = 0.01  # m

# Fluid inertial effects
wm_xu = -0.025*m  # kg
wm_yv = -0.25*m  # kg
wm_yr = -0.25*m*xg  # kg*m
wm_nr = -0.25*Iz  # kg*m**2

# Drag
d_xuu = 0.25 * wm_xu  # N/(m/s)**2
d_yvv = 0.25 * wm_yv # N/(m/s)**2
d_nrr = 0.25 * (wm_nr + wm_yr) # (N*m)/(rad/s)**2

# Cross-flow
d_yrr = 0.25 * wm_yr # N/(rad/s)**2
d_yrv = 0.25 * wm_yr  # N/(m*rad/s**2)
d_yvr = 0.25 * wm_yv  # N/(m*rad/s**2)
d_nvv = 0.25 * d_yvv # (N*m)/(m/s)**2
d_nrv = 0.25 * d_yrv # (N*m)/(m*rad/s**2)
d_nvr = 0.25 * (wm_nr + wm_yv) # (N*m)/(m*rad/s**2)

################################################# EQUATIONS OF MOTION

nstates = 6
ncontrols = 2  # assume no body-y control


def dynamics(x, u):
	"""
	Returns xdot = f(x, u)

	"""
	# Externally set parameters
	global m, Iz, xg, wm_xu, wm_yv, wm_yr, wm_nr,\
	       d_xuu, d_yvv, d_nrr, d_yrr, d_yrv, d_yvr,\
	       d_nvv, d_nrv, d_nvr

	# True wrench with zero body-y effort
	u = np.array([u[0], 0, u[1]], dtype=np.float64)

	# Mass matrix
	M = np.array([
	              [m - wm_xu,            0,            0],
	              [        0,    m - wm_yv, m*xg - wm_yr],
	              [        0, m*xg - wm_yr,   Iz - wm_nr]
	            ])

	# Centripetal coriolis matrix
	C = np.array([
	              [                                     0,                0, (wm_yr - m*xg)*x[5] + (wm_yv - m)*x[4]],
	              [                                     0,                0,                       (m - wm_xu)*x[3]],
	              [(m*xg - wm_yr)*x[5] + (m - wm_yv)*x[4], (wm_xu - m)*x[3],                                      0]
	            ])

	# Drag matrix
	D = np.array([
	              [-d_xuu*abs(x[3]),                                    0,                                    0],
	              [               0, -(d_yvv*abs(x[4]) + d_yrv*abs(x[5])), -(d_yvr*abs(x[4]) + d_yrr*abs(x[5]))],
	              [               0, -(d_nvv*abs(x[4]) + d_nrv*abs(x[5])), -(d_nvr*abs(x[4]) + d_nrr*abs(x[5]))]
	            ])

	# Rotation matrix (orientation, converts body to world)
	R = np.array([
	              [np.cos(x[2]), -np.sin(x[2]), 0],
	              [np.sin(x[2]),  np.cos(x[2]), 0],
	              [           0,             0, 1]
	            ])

	# M*vdot + C*v + D*v = u  and  pdot = R*v
	return np.concatenate((R.dot(x[3:]), np.linalg.inv(M).dot(u - (C + D).dot(x[3:]))))


def linearize(x):
	"""
	Returns A = jac(f, x) and B = jac(f, u)

	"""
	# Externally set parameters
	global m, Iz, xg, wm_xu, wm_yv, wm_yr, wm_nr,\
	       d_xuu, d_yvv, d_nrr, d_yrr, d_yrv, d_yvr,\
	       d_nvv, d_nrv, d_nvr

	A = np.array([
				  [ 0, 0, - x[4]*np.cos(x[2]) - x[3]*np.sin(x[2]),                                                                                                                                                                                                                                             np.cos(x[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         -np.sin(x[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
				  [ 0, 0,   x[3]*np.cos(x[2]) - x[4]*np.sin(x[2]),                                                                                                                                                                                                                                             np.sin(x[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          np.cos(x[2]),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 0],
				  [ 0, 0,                         0,                                                                                                                                                                                                                                                   0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1],
				  [ 0, 0,                         0,                                                                                                                                                                                                     (d_xuu*abs(x[3]) + d_xuu*x[3]*np.sign(x[3]))/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (m*x[5] - x[5]*wm_yv)/(m - wm_xu),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            (m*x[4] - 2*x[5]*wm_yr - x[4]*wm_yv + 2*m*x[5]*xg)/(m - wm_xu)],
				  [ 0, 0,                         0, -(x[5]*wm_yr**2 + m**2*x[5]*xg**2 - Iz*m*x[5] + Iz*x[5]*wm_xu + m*x[5]*wm_nr - x[5]*wm_nr*wm_xu - x[4]*wm_yr*wm_xu + x[4]*wm_yr*wm_yv - 2*m*x[5]*wm_yr*xg + m*x[4]*wm_xu*xg - m*x[4]*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(d_nrv*wm_yr*abs(x[5]) - d_yrv*wm_nr*abs(x[5]) + d_nvv*wm_yr*abs(x[4]) - d_yvv*wm_nr*abs(x[4]) - x[3]*wm_yr*wm_xu + x[3]*wm_yr*wm_yv + Iz*d_yrv*abs(x[5]) + Iz*d_yvv*abs(x[4]) + m*x[3]*wm_xu*xg - m*x[3]*wm_yv*xg + Iz*d_yvr*x[5]*np.sign(x[4]) + Iz*d_yvv*x[4]*np.sign(x[4]) - d_nrv*m*xg*abs(x[5]) - d_nvv*m*xg*abs(x[4]) + d_nvr*x[5]*wm_yr*np.sign(x[4]) - d_yvr*x[5]*wm_nr*np.sign(x[4]) + d_nvv*x[4]*wm_yr*np.sign(x[4]) - d_yvv*x[4]*wm_nr*np.sign(x[4]) - d_nvr*m*x[5]*xg*np.sign(x[4]) - d_nvv*m*x[4]*xg*np.sign(x[4]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg), -(x[3]*wm_yr**2 + d_nrr*wm_yr*abs(x[5]) - d_yrr*wm_nr*abs(x[5]) + d_nvr*wm_yr*abs(x[4]) - d_yvr*wm_nr*abs(x[4]) + m**2*x[3]*xg**2 - Iz*m*x[3] + Iz*x[3]*wm_xu + m*x[3]*wm_nr - x[3]*wm_nr*wm_xu + Iz*d_yrr*abs(x[5]) + Iz*d_yvr*abs(x[4]) - 2*m*x[3]*wm_yr*xg + Iz*d_yrr*x[5]*np.sign(x[5]) + Iz*d_yrv*x[4]*np.sign(x[5]) - d_nrr*m*xg*abs(x[5]) - d_nvr*m*xg*abs(x[4]) + d_nrr*x[5]*wm_yr*np.sign(x[5]) - d_yrr*x[5]*wm_nr*np.sign(x[5]) + d_nrv*x[4]*wm_yr*np.sign(x[5]) - d_yrv*x[4]*wm_nr*np.sign(x[5]) - d_nrr*m*x[5]*xg*np.sign(x[5]) - d_nrv*m*x[4]*xg*np.sign(x[5]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
				  [ 0, 0,                         0,                                             (x[4]*wm_yv**2 + m*x[4]*wm_xu - m*x[4]*wm_yv - x[5]*wm_yr*wm_xu + x[5]*wm_yr*wm_yv - x[4]*wm_xu*wm_yv + m*x[5]*wm_xu*xg - m*x[5]*wm_yv*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                (x[3]*wm_yv**2 - d_nrv*m*abs(x[5]) - d_nvv*m*abs(x[4]) + d_nrv*wm_yv*abs(x[5]) - d_yrv*wm_yr*abs(x[5]) + d_nvv*wm_yv*abs(x[4]) - d_yvv*wm_yr*abs(x[4]) + m*x[3]*wm_xu - m*x[3]*wm_yv - x[3]*wm_xu*wm_yv + d_yrv*m*xg*abs(x[5]) + d_yvv*m*xg*abs(x[4]) - d_nvr*m*x[5]*np.sign(x[4]) - d_nvv*m*x[4]*np.sign(x[4]) + d_nvr*x[5]*wm_yv*np.sign(x[4]) - d_yvr*x[5]*wm_yr*np.sign(x[4]) + d_nvv*x[4]*wm_yv*np.sign(x[4]) - d_yvv*x[4]*wm_yr*np.sign(x[4]) + d_yvr*m*x[5]*xg*np.sign(x[4]) + d_yvv*m*x[4]*xg*np.sign(x[4]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg),                                      -(d_nrr*m*abs(x[5]) + d_nvr*m*abs(x[4]) - d_nrr*wm_yv*abs(x[5]) + d_yrr*wm_yr*abs(x[5]) - d_nvr*wm_yv*abs(x[4]) + d_yvr*wm_yr*abs(x[4]) + x[3]*wm_yr*wm_xu - x[3]*wm_yr*wm_yv - m*x[3]*wm_xu*xg + m*x[3]*wm_yv*xg - d_yrr*m*xg*abs(x[5]) - d_yvr*m*xg*abs(x[4]) + d_nrr*m*x[5]*np.sign(x[5]) + d_nrv*m*x[4]*np.sign(x[5]) - d_nrr*x[5]*wm_yv*np.sign(x[5]) + d_yrr*x[5]*wm_yr*np.sign(x[5]) - d_nrv*x[4]*wm_yv*np.sign(x[5]) + d_yrv*x[4]*wm_yr*np.sign(x[5]) - d_yrr*m*x[5]*xg*np.sign(x[5]) - d_yrv*m*x[4]*xg*np.sign(x[5]))/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
				])

	# Underactuated B-matrix
	B = np.array([ 
				  [             0,                                                                                                0],
				  [             0,                                                                                                0],
				  [             0,                                                                                                0],
				  [ 1/(m - wm_xu),                                                                                                0],
				  [             0, -(wm_yr - m*xg)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)],
				  [             0,    -(m - wm_yv)/(m**2*xg**2 - Iz*m + Iz*wm_yv + m*wm_nr - wm_nr*wm_yv + wm_yr**2 - 2*m*wm_yr*xg)]
				])

	return (A, B)

################################################# SIMULATION

# Initial condition and time
x = [0, 0.5, np.deg2rad(0), 0, 0, 0]
dt = 0.02  # s
T = 10  # s
t_arr = np.arange(0, T, dt)
framerate = 30
show_cost_field = True


# Set up a cost field
goal = [1, 1, np.deg2rad(0), 0, 0, 0]
cost_field = alqr.Cost_Field(nstates, ncontrols, 2, goal,
							 goal_weight=[400, 400, 100, 400, 400, 1],
							 effort_weight=[0.2, 0.00001], obstacle_weight=600)

# Noised grid of obstacles
obs_grid_x, obs_grid_y = np.mgrid[slice(0.3, 0.8+0.2, 0.2), slice(0, 1+0.2, 0.2)]
obs_grid_x = obs_grid_x.reshape(obs_grid_x.size)
obs_grid_y = obs_grid_y.reshape(obs_grid_y.size)
obs = [np.zeros(2)] * obs_grid_x.size
for i in range(len(obs)):
	obs[i] = np.round([obs_grid_x[i], obs_grid_y[i]] + 0.07*(np.random.rand(2)-0.5), 2)
	name = 'buoy' + str(i)
	if npl.norm(obs[i] - goal[:2]) > 0.2:
		cost_field.add_obstacle(name, obs[i], 0.1)


# Associate an alqr planner
planning_horizon = T  # s
planning_resolution = dt  # s
planner = alqr.Planner(dynamics, linearize, cost_field,
					   planning_horizon, planning_resolution,
					   eps_converge=0.001, demo_plots=True)

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

ax1 = fig1.add_subplot(2, 4, 1)
ax1.set_ylabel('X Position (m)', fontsize=16)
ax1.plot(t_arr, x_history[:, 0], 'k',
		 t_arr, goal_history[:, 0], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 2)
ax1.set_ylabel('Y Position (m)', fontsize=16)
ax1.plot(t_arr, x_history[:, 1], 'k',
		 t_arr, goal_history[:, 1], 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 3)
ax1.set_ylabel('Heading (deg)', fontsize=16)
ax1.plot(t_arr, np.rad2deg(x_history[:, 2]), 'k',
		 t_arr, np.rad2deg(goal_history[:, 2]), 'g--')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 4)
ax1.set_ylabel('Efforts (N, N*m)', fontsize=16)
ax1.plot(t_arr, u_history[:, 0], 'b',
		 t_arr, u_history[:, 1], 'g')
ax1.grid(True)

ax1 = fig1.add_subplot(2, 4, 5)
ax1.set_ylabel('X Velocity', fontsize=16)
ax1.plot(t_arr, x_history[:, 3], 'k',
		 t_arr, goal_history[:, 3], 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 6)
ax1.set_ylabel('Y Velocity', fontsize=16)
ax1.plot(t_arr, x_history[:, 4], 'k',
		 t_arr, goal_history[:, 4], 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 7)
ax1.set_ylabel('Yaw Rate', fontsize=16)
ax1.plot(t_arr, np.rad2deg(x_history[:, 5]), 'k',
		 t_arr, np.rad2deg(goal_history[:, 5]), 'g--')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

ax1 = fig1.add_subplot(2, 4, 8)
ax1.set_ylabel('State Cost', fontsize=16)
ax1.plot(t_arr, c_history, 'k')
ax1.grid(True)
ax1.set_xlabel('Time (s)')

print("\nClose the plot window to continue to animation.")
plt.show()


# Animation
fig2 = plt.figure()
fig2.suptitle('Evolution', fontsize=24)
plt.axis('equal')

ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_xlabel('- X Position +')
ax2.set_ylabel('- Y Position +')
ax2.grid(True)

radius = 0.02
xlim = (min(x_history[:, 0])*1.1 - radius, max(x_history[:, 0])*1.1 + radius)
ylim = (min(x_history[:, 1])*1.1 - radius, max(x_history[:, 1])*1.1 + radius)
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
			Jmap[i, j] = cost_field.state_cost(np.concatenate(([xval, yval], np.zeros(int(nstates-2)))))
			if Jmap[i, j] < 0:
				print "Negative cost! At ({0}, {1})".format(xval, yval)
	Jmap = Jmap[:-1, :-1]
	plt.pcolor(X, Y, Jmap, cmap='YlOrRd', vmin=np.min(Jmap), vmax=np.max(Jmap))
	plt.colorbar()

graphic_robot = ax2.add_patch(plt.Circle((x_history[0, 0], x_history[0, 1]), radius=radius, fc='k'))
graphic_goal = ax2.add_patch(plt.Circle((goal_history[0, 0], goal_history[0, 1]), radius=radius, fc='g'))

llen = 0.05
lwid = 3
graphic_heading = ax2.plot([x_history[0, 0], x_history[0, 0] + llen*np.cos(x_history[0, 2])],
						   [x_history[0, 1], x_history[0, 1] + llen*np.sin(x_history[0, 2])], color='k', linewidth=lwid)
graphic_goal_heading = ax2.plot([goal_history[0, 0], goal_history[0, 0] + llen*np.cos(goal_history[0, 2])],
								[goal_history[0, 1], goal_history[0, 1] + llen*np.sin(goal_history[0, 2])], color='g', linewidth=lwid)

for p in cost_field.obstacle_positions:
	ax2.add_patch(plt.Circle((p[0], p[1]), radius=radius/2, fc='r'))

def ani_update(arg, ii=[0]):

	i = ii[0]  # don't ask...

	if np.isclose(t_arr[i], np.around(t_arr[i], 1)):
		fig2.suptitle('Evolution (Time: {})'.format(t_arr[i]), fontsize=24)

	graphic_robot.center = ((x_history[i, 0], x_history[i, 1]))
	graphic_heading[0].set_data([x_history[i, 0], x_history[i, 0] + llen*np.cos(x_history[i, 2])],
								[x_history[i, 1], x_history[i, 1] + llen*np.sin(x_history[i, 2])])
	
	ii[0] += int(1 / (dt * framerate))
	if ii[0] >= len(t_arr):
		print("Resetting animation!")
		ii[0] = 0

	return [graphic_robot]

# Run animation
print("\nStarting animation. \nBlack: robot \nRed: obstacles \nGreen: goal \nHeat Map: state cost\n")
animation = ani.FuncAnimation(fig2, func=ani_update, interval=dt*1000)
plt.show()
