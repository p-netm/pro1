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
# from scipy.integrate import odeint
# from sympy import Symbol


# x = Symbol('x')
# y = Symbol('y')

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

# def _plot(variables_list):
#     """
#     :params:variables_list: contains aggregated information for creating a single plot
#     the plot will have several paths for several mu's constant plotted on the 
#     x-y coordinate using list type variables inside the dictionary that 
#     repectively correspond to either values of x or values of y
#     variables_dictionary_structure:
#         list[{
#             'x': [],
#             'y': [],
#             'mu': <mu>
#         },
#         {}]
#     """
#     legends = []
#     plot = plt.plot
#     for entry in variables_list:
#         plot(entry['x'], entry['y'])
#         legends.append(entry['mu'])
#     # import ipdb;ipdb.set_trace()
#     return plt

def generate_init_conditions():
    """Generates one instance of the initial conditions 
    
    returns x(t) = x0, y(t) = y0, h(step_size) in range [0, 1]
    i.e. t, initial_x, initial_y, h, steps
    """
    K, L = randint(0, 20), randint(0, 20)
    _x = -1 + 0.1 * K
    _y = -1 + 0.1 * L
    return list(map(Fraction, [0, _x, _y, 0.1, 10]))

def aggregator(t, _x, _y, h, steps):
    """
    calls the rk4 function iteratively and aggregates the result
    into sequences that can be consumed by the _plot function
    """
    from decimal import Decimal, localcontext
    mu_s = [0.1]
    results_list = []
    for mu in mu_s:
        mini_dict = {
            't': [t],
            'x': [_x],
            'y': [_y],
            'mu': mu
        }
        with localcontext() as ctx:
            ctx.prec = 100
            for i in range(30):
                response = rk4(mini_dict['t'][i], mini_dict['x'][i], mini_dict['y'][i], h, mu, f, g)
                mini_dict['t'].append(response[0])
                mini_dict['x'].append(response[1])
                mini_dict['y'].append(response[2])
            results_list.append(mini_dict)
    return results_list

for y20 in [-1, -0.5, -0.2, 0.0, 0.1, 0.5, 1.0]:
    tspan = np.linspace(0, 50, 200)
    y0 = [0.0, y20] # initial conditions
    # so i think below: i think its purpose is to create the list of
    # points that are the immediate iterative solutions of 
    # x and y. am suggesting that we could probably create this solutions 
    # using our runge kutta implementation.
    # ys = odeint(f, y0, tspan)
    t, _x, _y, step_size, steps = generate_init_conditions()
    variables_result = aggregator(t, _x, _y, step_size, steps)
    mini_dict = variables_result[0]
    x_plots = mini_dict['x']
    y_plots = mini_dict['y']
    # import ipdb; ipdb.set_trace()
    plt.plot(x_plots, y_plots, 'b-') # path
    # plt.plot([ys[0,0]], [ys[0,1]], 'o') # start
    # plt.plot([ys[-1,0]], [ys[-1,1]], 's') # end

plt.show()
def main(n=4):
    """Generates initial conditions n times then aggregates the plot
    data for that specific initial condition and then passes the
    data on the custom plot function 
    It does so so that we can several subplots within the same plot figure
    """
    # for _ in range(n):
    #     # import ipdb;ipdb.set_trace()
    #     t, _x, _y, step_size, steps = generate_init_conditions()
    #     variables_result = aggregator(t, _x, _y, step_size, steps)
    #     plot = _plot(variables_result)
    # plot.show()
    y1 = np.linspace(-2.0, 2.0, 20)
    y2 = np.linspace(-2.0, 2.0, 20)

    Y1, Y2 = np.meshgrid(y1, y2)

    t = 0

    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

    ni, nj = Y1.shape

    for i in range(ni):
        for j in range(nj):
            x = Y1[i, j]
            try:
                y = Y2[i, j]
            except IndexError:
                import ipdb; ipdb.set_trace()	
            yprime = F([x, y], t)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]

    Q = plt.quiver(Y1, Y2, u, v, color='r')

    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])

main()