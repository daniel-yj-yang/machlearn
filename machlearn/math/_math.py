# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

from sympy import *
import torch
import matplotlib.pylab as plt


class calculus(object):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def derivative_using_sympy():
        x = symbols('x')
        for expr in [x, sin(x), exp(x), log(x), x**x]:
            for i in range(4):
                expr = Derivative(expr, x)
                print(f"{expr} = {expr.doit()}, when x=1.0, expression = {expr.doit().evalf(subs={x:1,})}")

    @staticmethod
    def integrate_using_sympy():
        x = symbols('x')
        for expr in [x, sin(x), exp(x), log(x), x**x]:
            for i in range(4):
                expr = Integral(expr, x)
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
        plt.show()


def demo():
    calculus.derivative_plot_using_torch()
    calculus.derivative_using_sympy()
    calculus.integrate_using_sympy()
    calculus.derivative_using_torch()

