import numpy as np
import sympy as sym
import pandas as pd
import matplotlib.pyplot as plt

sym.init_printing(use_latex=True)
from IPython.display import display, Markdown

# FIXME!!! add docstrings

############################
class Limits:
    def __init__(self, lower: int, upper: int):
        self.lower = lower
        self.upper = upper
    def __repr__(self):
        return f"<Limits: lower={self.lower}, upper={self.upper}>"

############################
# Globals
# FIXME!!! move to __init__ defaults and args as needed
#our_domain = Limits(0, 4)  # colloquial english (not helpful)
func_range = Limits(0, 4)    # formal mathematical english
func_domain = Limits(-1, 15) # formal mathematical english 

############################
class TaylorExpansion:
    def __init__(self):
        # Symbols
        self.x = sym.Symbol('x')
        self.x_0 = sym.Symbol('x_0')

        # Function
        self.f = sym.sqrt(1 + self.x)

        # Derivatives
        self.f_prime = sym.diff(self.f, self.x)
        self.f_double_prime = sym.diff(self.f_prime, self.x)

        # Evaluated at x_0
        self.f0 = self.f.subs(self.x, self.x_0)
        self.f1 = self.f_prime.subs(self.x, self.x_0)
        self.f2 = self.f_double_prime.subs(self.x, self.x_0)

        # Taylor expansion
        self.f_taylor = self.f0 + self.f1 * self.x + (self.f2 / sym.factorial(2)) * self.x**2
        self.f_taylor_at_0 = self.f_taylor.subs(self.x_0, 0)

    def show_derivatives(self):
        display(sym.Eq(sym.Function('f')(self.x), self.f))
        display(sym.Eq(sym.Function("f'")(self.x), self.f_prime))
        display(sym.Eq(sym.Function("f''")(self.x), self.f_double_prime))

    def show_derivation(self):
        display(sym.Eq(sym.Function('f_{taylor}')(self.x), self.f_taylor))
        display(sym.Eq(sym.Function('f_{taylor}')(0), self.f_taylor_at_0))
        display(Markdown(""))

    def __repr__(self):
        return f"<TaylorExpansion: f(x) = sqrt(1 + x), centered at x_0>"


#####################################################
class TaylorSymPlot:
    def __init__(self):
        # Symbolic setup
        self.x = sym.Symbol('x')
        self.f = sym.sqrt(1 + self.x)

        # Derivatives at x = 0
        self.f0 = self.f.subs(self.x, 0)
        self.f1 = sym.diff(self.f, self.x).subs(self.x, 0)
        self.f2 = sym.diff(self.f, self.x, 2).subs(self.x, 0)

        # Taylor approximations
        self.f_taylor_1st_order = self.f0 + self.f1 * self.x
        self.f_taylor_2nd_order = self.f0 + self.f1 * self.x + (self.f2 / sym.factorial(2)) * self.x**2

    def plot(self):
        sym.init_printing(use_latex=True)

        p = sym.plot(
            self.f,
            self.f_taylor_1st_order,
            self.f_taylor_2nd_order,
            (self.x, func_domain.lower, func_domain.upper),
            show=False,
            legend=True,
            xlabel='x',
            ylabel='f(x)',
            title='Function and Taylor Approximations at $x_0 = 0$ (linear)'
        )

        # Labels and colors
        p[0].label = r'$f(x) = \sqrt{1 + x}$'
        p[1].label = '1st order Taylor'
        p[2].label = '2nd order Taylor'

        p[0].line_color = 'blue'
        p[1].line_color = 'green'
        p[2].line_color = 'red'

        p.show()

    def __repr__(self):
        return f"<TaylorPlot: domain={func_domain}>"

############################

class TaylorNumericPlot:
    def __init__(self):
        # Truth function
        self.f = lambda x: np.sqrt(1 + x)

        # Derivatives at x = 0
        self.f0 = self.f(0)
        self.f1 = 1 / (2 * np.sqrt(1))         # f'(0)
        self.f2 = -1 / (4 * (1)**(3/2))        # f''(0)

        # Taylor approximations
        self.f_taylor_1st = lambda x: self.f0 + self.f1 * x
        self.f_taylor_2nd = lambda x: self.f0 + self.f1 * x + (self.f2 / 2) * x**2

        # Domain values
        self.x_vals = np.linspace(func_domain.lower, func_domain.upper, 400)

    def plot(self):
        y_true = self.f(self.x_vals)
        y_t1 = self.f_taylor_1st(self.x_vals)
        y_t2 = self.f_taylor_2nd(self.x_vals)

        plt.figure(figsize=(10, 6))
        plt.loglog(self.x_vals, y_true, label=r'$f(x) = \sqrt{1 + x}$', color='blue', linewidth=2)
        plt.loglog(self.x_vals, y_t1, label='1st order Taylor', linestyle='--', color='green')
        plt.loglog(self.x_vals, y_t2, label='2nd order Taylor', linestyle=':', color='red')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Function and Taylor Approximations at $x_0 = 0$ (loglog)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<TaylorNumericPlot: domain={func_domain}>"


#####################################################
# Consider a sub-class for our approximation logic as this concept developes.
# We had an idea about mode and method as an operational seperation.
# this might not be appearent as a linguistic interpetation of operations.
# Endomorphism
# think about this...
# note to self ;-)

class TaylorTable:
    def __init__(self):
        # Evaluate function at =1+x -> x=a-1
        self.a_vals = [1.5, 2.0, 3.0]
        self.x_vals = np.array(self.a_vals) - 1

        # True function
        self.f = lambda x: np.sqrt(1 + x)

        # Derivatives at x = 0
        self.f0 = self.f(0)
        self.f1 = 1 / (2 * np.sqrt(1))         # f'(0)
        self.f2 = -1 / (4 * (1)**(3/2))        # f''(0)

        # Taylor approximations
        self.f_taylor_1st = lambda x: self.f0 + self.f1 * x
        self.f_taylor_2nd = lambda x: self.f0 + self.f1 * x + (self.f2 / 2) * x**2

        # Evaluate
        self.y_true = self.f(self.x_vals)
        self.y_t1 = self.f_taylor_1st(self.x_vals)
        self.y_t2 = self.f_taylor_2nd(self.x_vals)

        # Build our initial DataFrame
        self.columns = ['x', 'y_true', 'y_t1', 'y_t2']  # think about idempotency
        self.index = [f'test {i+1}' for i in range(len(self.a_vals))]
        self.df = pd.DataFrame(
            data=np.array([self.x_vals, self.y_true, self.y_t1, self.y_t2]).T,
            index=self.index,
            columns=self.columns
        )

    def show(self):
        display(self.df)

    def plot_errors(self):
        # Compute error columns
        self.df['absolute_error_t1'] = np.abs(self.df['y_t1'] - self.df['y_true'])
        self.df['absolute_error_t2'] = np.abs(self.df['y_t2'] - self.df['y_true'])
        self.df['relative_error_t1'] = self.df['absolute_error_t1'] / self.df['y_true']
        self.df['relative_error_t2'] = self.df['absolute_error_t2'] / self.df['y_true']

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        # Absolute error
        self.df.plot(
            ax=axes[0],
            x='x',
            y=['absolute_error_t1', 'absolute_error_t2'],
            kind='line',
            title='Absolute Error',
            xlabel='x',
            ylabel='Absolute Error',
            color=['green', 'red'],
            marker='o',
            grid=True
        )

        # Relative error
        self.df.plot(
            ax=axes[1],
            x='x',
            y=['relative_error_t1', 'relative_error_t2'],
            title='Relative Error',
            xlabel='x',
            ylabel='Error (%)',
             color=['green', 'red'],
            marker='o',
            grid=True
        )

        plt.suptitle('Taylor Approximation Error Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return f"<TaylorTable: {len(self.df)} evaluations over domain={func_domain}>"

# if you've read this far... run lint again!
