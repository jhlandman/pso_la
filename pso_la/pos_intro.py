"""
This intro is from https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy

# * --------------------------------------------------------
# * ------------- Intro with obj func ----------------------
# * --------------------------------------------------------

def f(x, y):
    "Objective function"
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)

def setup_grid(f):
# so here we have a 100 x 100 grid in x and y, and the value for z with these two inputs
    x, y = np.array(np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100)))
    z = f(x, y)
    # The x and y values of the z min
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    return x, y, z, x_min, y_min

x, y, z, x_min, y_min = setup_grid(f)
plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color='white')
# we have a global min denoted by the x
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
plt.savefig('./data/countour_f_x_y.png')
plt.close()


# * --------------------------------------------------------
# * ------------- initialising some particles --------------
# * --------------------------------------------------------

def init_particles(n_particles):
    # UNIF [0, 5] 2 sets of n
    np.random.seed(100)
    X = np.random.rand(2, n_particles) * 5
    # NORM(0, 0.1) 2 sets of n
    np.random.seed(200)
    V = np.random.randn(2, n_particles) * 0.1
    return X, V

x, y, z, x_min, y_min = setup_grid(f)
X, V = init_particles(20)

plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color='white')
# we have a global min denoted by the x
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
plt.scatter(X[0], X[1])
plt.quiver(X[0], X[1], V[0], V[1])
plt.savefig('./data/countour_f_x_y_particle_init.png')
plt.close()
# ? pretty cool, can see no condition to keep everything [0, 5] x [0, 5] after step

# * --------------------------------------------------------
# * -------------------- gbest and pbest -------------------
# * --------------------------------------------------------

def find_init_pbest_gbest(X, f):
    # assign previous best as current
    pbest = deepcopy(X)
    # find z values
    pbest_obj = f(X[0], X[1])
    gbest = pbest[:, pbest_obj.argmin()]
    gbest_obj = pbest_obj.min()
    return pbest, pbest_obj, gbest, gbest_obj

x, y, z, x_min, y_min = setup_grid(f)
X, V = init_particles(20)
pbest, pbest_obj, gbest, gbest_obj = find_init_pbest_gbest(X, f)

plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color='white')
# we have a global min denoted by the x
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.0f')
plt.scatter(X[0], X[1])
plt.quiver(X[0], X[1], V[0], V[1])
plt.plot([gbest[0]], [gbest[1]], marker='x', markersize=8, color='red')
plt.savefig('./data/countour_f_x_y_particle_init_gbest.png')
plt.close()


# * --------------------------------------------------------
# * -------------- walk forward iteration -------------------
# * --------------------------------------------------------

def walk_forward(X, V, r, pbest, pbest_obj, gbest, gbest_obj):
    V = (
        w * V + # ? dampen the steps?
        c1 * r[0] * (pbest - X) + # ? going to be 0 because pbest and X are the same here
        c2 * r[1] * (gbest.reshape(-1, 1) - X) # ? get everything to walk towards gbest by UNIF[0, 1] * 0.1 approx. 0.05
    )
    X = X + V # ? walk forward from X
    obj = f(X[0], X[1]) # ? see what the objective function values are
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)] # ? assign the ones that walked in the right direction as pbest
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0) # ? assign the lowest values as lowest objective function values
    gbest = pbest[:, pbest_obj.argmin()] # ? find the best of everything
    gbest_obj = pbest_obj.min() # ? find the lowest objective function value
    return X, V, r, pbest, pbest_obj, gbest, gbest_obj

x, y, z, x_min, y_min = setup_grid(f)
X, V = init_particles(20)
pbest, pbest_obj, gbest, gbest_obj = find_init_pbest_gbest(X, f)
c1, c2, w = 0.1, 0.1, 0.8 # ? hyper-parameters
np.random.seed(300); r = np.random.rand(2)
X, V, r, pbest, pbest_obj, gbest, gbest_obj = walk_forward(X, V, r, pbest, gbest)


# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([0,5])
ax.set_ylim([0,5])
 
def animate(i):
    "Steps of PSO: algorithm update and show in plot"
    title = 'Iteration {:02d}'.format(i)
    # Update params
    update()
    # Set picture
    ax.set_title(title)
    pbest_plot.set_offsets(pbest.T)
    p_plot.set_offsets(X.T)
    p_arrow.set_offsets(X.T)
    p_arrow.set_UVC(V[0], V[1])
    gbest_plot.set_offsets(gbest.reshape(1,-1))
    return ax, pbest_plot, p_plot, p_arrow, gbest_plot

anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500, blit=False, repeat=True)
anim.save("PSO.gif", dpi=120, writer="imagemagick")

print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
print("Global optimal at f({})={}".format([x_min,y_min], f(x_min,y_min)))