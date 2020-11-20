# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause


import inspect
import numpy as np
import sympy
import torch
import matplotlib.pylab as plt


class calculus(object):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def derivative_using_sympy():
        x = sympy.symbols('x')
        for expr in [x, sympy.sin(x), sympy.exp(x), sympy.log(x), x**x]:
            for i in range(4):
                expr = sympy.Derivative(expr, x)
                print(f"{expr} = {expr.doit()}, when x=1.0, expression = {expr.doit().evalf(subs={x:1,})}")

    @staticmethod
    def derivative_plot_using_sympy():
        x = sympy.symbols('x')
        expr = x ** 3
        deriv = sympy.Derivative(expr, x).doit()
        x_values = np.linspace(-5, 5, 1000)
        expr_values  = np.array([expr.evalf(subs={x:x_value})  for x_value in x_values])
        deriv_values = np.array([deriv.evalf(subs={x:x_value}) for x_value in x_values])
        plt.plot(x_values, expr_values,  label = 'x ** 3')
        plt.plot(x_values, deriv_values, label='derivative (3*x**2)')
        plt.xlabel('x')
        plt.legend()
        plt.title(f"{inspect.getframeinfo(inspect.currentframe()).function}()")
        plt.show()

    @staticmethod
    def integrate_using_sympy():
        x = sympy.symbols('x')
        for expr in [x, sympy.sin(x), sympy.exp(x), sympy.log(x), x**x]:
            for i in range(4):
                expr = sympy.Integral(expr, x)
                print(f"{expr} = {expr.doit()}, when x=1.0, expression = {expr.doit().evalf(subs={x:1,})}")
    
    @staticmethod
    def derivative_using_torch():
        x = torch.tensor(1.0, requires_grad=True)
        expr = torch.exp(x)
        expr.backward()
        print("The dervative of exp(x) at x = 1.0: ", x.grad)

    @staticmethod
    def derivative_plot_using_torch():
        x = torch.linspace(-5, 5, 1000, requires_grad=True)
        expr = x ** 3
        sum_trick = torch.sum(expr) # a sum trick
        sum_trick.backward()
        plt.plot(x.detach().numpy(),   expr.detach().numpy(), label='x ** 3')
        plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative (3*x**2)')
        plt.xlabel('x')
        plt.legend()
        plt.title(inspect.getframeinfo(inspect.currentframe()).function)
        plt.show()


def math_demo():
    calculus.derivative_plot_using_sympy()
    calculus.derivative_plot_using_torch()

    calculus.derivative_using_sympy()
    calculus.integrate_using_sympy()
    calculus.derivative_using_torch()

