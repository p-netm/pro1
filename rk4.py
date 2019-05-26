"""
Project1: fourth order Runge Kutta solution for second order ode as 
    a system of first oder ode's 
_x, _y have been used as the names variable for x, y
    respectively as not to override the 
    Symbol('x'), and Symbol('y') variable names which are 
    defined in the global scope as x and y
"""
from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
from random import randint

def f(mu, x, y):
    """ 
    x' = µx - y + xy**2
    """
    return mu * x - y + x * y ** 2


def g(mu, x, y):
    """ 
    y' = x + µy + y**3
    """
    return x + mu * y + y ** 3

def F(Y, _t):
    _x, _y = Y
    mu = 0.1
    return [mu * _x - _y + _x * _y ** 2, _x + mu * _y + _y ** 3]


def rk4(t, _x, _y, h, mu, f1, f2):
    """
    h : the step of the mesh i.e how much we increment the t for each iterative turn
    t: 
    _x: holds value of x in the function's local scope
    _y:
    mu: an arbitrary constant
    f1, f2: resultant first ODES after decomposing the 2 order ODE

    Computes the vriables k1, k2, ..., l3, l4 
    returns a tuple of x, y , and t for each turn its called
    """
    k1 = f1(mu, _x, _y)
    l1 = f2(mu, _x, _y)

    x = _x + h * k1 / 2
    y = _y + h * l1 /2
    k2 = f1(mu, x, y)
    l2 = f2(mu, x, y)
    
    x= _x + h * k2 / 2
    y= _y + h * l2 /2
    k3 = f1(mu, x, y)
    l3 = f1(mu, x, y)
    
    x= _x + h * k3
    y= _y + h * l3
    k4 = f1(mu, x, y)
    l4 = f1(mu, x, y)

    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    l = (l1 + 2 * l2 + 2 * l3 + l4) / 6

    _x += h * k
    _y += h * l
    t = t + h
    return t, _x, _y

def grid_plot(mu):
    """
    Returns a quiver plot with labels and a mesh grid
    over which the runge kutta solutions will be plotted
    """
    x = np.linspace(-2.0, 2.0, 20)
    y = np.linspace(-2.0, 2.0, 20)
    X, Y = np.meshgrid(x, y)
    u, v = np.zeros(X.shape), np.zeros(Y.shape)
    ni, nj = X.shape

    for i in range(ni):
        for j in range(nj):
            x = X[i, j]
            y = Y[i, j]
            yprime = F([x, y], t)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]

    Q = plt.quiver(X, Y, u, v, color='r')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    return plt

def _plot(plot, initial_conditions):
    """
    :params: plot: a pyplot object that is already created as a quiver
    with a meshgrid . This function retrieves plot points data
    from the aggregator and embedds them to the quiver plt object
    :params: initial_conditions: a list of pregenerated initial conditions
    """
    for condition in initial_conditions:
        t, _x, _y, step_size, steps = condition
        mini_dict = aggregator(t, _x, _y, step_size, mu, steps)
        x_points = mini_dict['x']
        y_points = mini_dict['y']
        plot.plot(x_points, y_points, 'b-') # path
    return plot

def generate_init_conditions(steps=10, step_size=0.1):
    """Generates one instance of the initial conditions 
    
    returns x(t) = x0, y(t) = y0, h(step_size) in range [0, 1]
    i.e. t, initial_x, initial_y, h, steps
    """
    K, L = randint(0, 20), randint(0, 20)
    _x = -1 + 0.1 * K
    _y = -1 + 0.1 * L
    return list(map(Fraction, [0, _x, _y, step_size, steps]))

def aggregator(t, _x, _y, h, mu, steps):
    """
    calls the rk4 function iteratively for a single value of mu
    and aggregates the result into a dictionary containing sequences
    of plot data that can be consumed by the _plot function
    """
    from decimal import Decimal, localcontext
    mini_dict = {
        't': [t],
        'x': [_x],
        'y': [_y],
        'mu': mu
    }
    for i in range(30):
        response = rk4(mini_dict['t'][i], mini_dict['x'][i], mini_dict['y'][i], h, mu, f, g)
        mini_dict['t'].append(response[0])
        mini_dict['x'].append(response[1])
        mini_dict['y'].append(response[2])
    return mini_dict


def main(n=10):
    """
    Generates initial conditions n times then aggregates the plot
    data for a specific initial condition and then passes the
    data on the custom plot function 
    """
    # a list of initial conditions 
    inits = [generate_init_conditions() for _ in range(n)]
    
    # i suggest we create a function that is given the plot object
    # and the plot data and returns a plt object with the data embedded
    for mu in [-1.0, -0.5, -0.2, 0.0, 0.1, 0.5, 1.0]:
        # quiver plot
        plt = grid_plot(mu)
        # Embed plot points data
        plt = _plot(plt, inits)
    plt.show()

main()