import sympy as sp
from sympy.abc import z
from sympy import hermite, chebyshevt
from sympy.stats import Normal, density

def generate_n_herm_expr(c):
    expr = 0
    for i, coef in enumerate(c):
        expr += coef*hermite(i, z)
    
    return expr

def generate_n_cheb_expr(c):
    expr = 0
    for i, coef in enumerate(c):
        expr += coef*chebyshevt(i, z)
    
    return expr

def generate_n_gauss_expr(w, means, sigmas):
    expr = 0
    for i, coef in enumerate(w):
        expr += coef*(1. / sp.sqrt(2 * sp.pi) / sigmas[i]
                * sp.exp(-0.5 * (z - means[i]) ** 2 / sigmas[i] ** 2))
    
    return expr

def eval_n_herm(x, c):
    x2 = x*2
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - c1*(2*(nd - 1))
            c1 = tmp + c1*x2
    return c0 + c1*x2

# a = hermite(1, x)
# b = hermite(2, x)
# c = a+b   
#x = Symbol('x')
#X = Normal("x", 0, 1)