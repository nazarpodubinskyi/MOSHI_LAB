import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
# find exact value of given point for given function
def f_value_in_point(f, p):
    return f(p)

# calculates abs error
def abs_error(approx_val, abs_val):
    return approx_val - abs_val

# calculates approximate error
def approx_error(approx_val, abs_val):
    return 100 * np.abs((abs_val - approx_val) / abs_val)

# params:
# f - function to integrate,
# a - lower limit,
# b - upper limit.
def exact(f, a, b):
    return quad(f, a, b)[0]

# params:
# f - function to integrate,
# N - number of points to generate randomly
# a - lower limit,
# b - upper limit.
def montecarlo(f, a, b, N):
    # to ensure that upper y bound is larger than largest y of eq (supremum of function)
    M = 1.4 * max(f(np.linspace(a, b)))

    # generate dots
    under = []
    above = []
    for i in range(N):
        # generate rand pt
        x_val = np.random.rand(1) * (b - a) + a
        y_val = np.random.rand(1) * M

        # compare random against the curve
        fx = f(x_val)

        # logic statement
        if y_val < fx:  # under the curve - blue
            under.append([x_val, y_val])
        else:  # above the curve - red
            above.append([x_val, y_val])

    # plotting
    plot_points(under, above)

    # Integral value Monte Carlo Calculation
    return len(under) / N * (M * (b - a))

def plot_points(under_curve, above_curve):
    under_curve = np.array(under_curve)
    above_curve = np.array(above_curve)

    plt.plot(above_curve[:, 0], above_curve[:, 1], 'ro', markersize=3, markerfacecolor='r')
    plt.plot(under_curve[:, 0], under_curve[:, 1], 'bo', markersize=3, markerfacecolor='b')
    plt.title('Monte Carlo Integration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['above','under'])
    plt.show()
# case #1 (test: trigonometrical positively defined function a = 0, b = 1)
print('Case 1 ')
print('f = 10 + 5*sin(5*x)  a=0 b=1')
f1 = lambda x: (10 + 5 * np.sin(5 * x)) ** 2
Monte_Carlo = montecarlo(f1, 0, 1, 1000)
Exact_Value, _ = quad(f1, 0, 1)
print('Monte_Carlo   Exact_Value     Abs_Error   Approx_Error')
print(
    f'{Monte_Carlo}         {Exact_Value}        {abs_error(Monte_Carlo, Exact_Value)}        {approx_error(Monte_Carlo, Exact_Value)}%')
print(f'value in x = 0; y = {f_value_in_point(f1, 0)}')
print('------------------------------------------------')

# case #2 (main)
print('Case 2 ')
print('exp(x).*x.^2.*sqrt(exp(x))   a=1 b=2')
f2 = lambda x: np.exp(x) * x ** 2 * np.sqrt(np.exp(x))
Monte_Carlo = montecarlo(f2, 1, 2, 1000)
Exact_Value, _ = quad(f2, 1, 2)
labels = []
print('Monte_Carlo   Exact_Value     Abs_Error   Approx_Error')
print(
    f'{Monte_Carlo}         {Exact_Value}        {abs_error(Monte_Carlo, Exact_Value)}        {approx_error(Monte_Carlo, Exact_Value)}%')
print(f'value in x = 1; y = {f_value_in_point(f2, 1)}')
print('------------------------------------------------')





