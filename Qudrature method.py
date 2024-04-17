import numpy as np
import numdifftools as nd
import math
import sympy as sp

eps = 0.00001 

a = 0 
b = 1

x = sp.symbols('x')


# для нахождения значения функции в точке х
def f(x):
    return x+0.5
def K(x, s):
    return -1

# для вывода самой формулы
def f1(x):
    return x+0.5

def K1(x, s):
    return -1


#Этап 1

def qudrature_method(n, flag):
    h = (b-a)/(n-1)
    s = []  #список узлов
    A = []  #список коэффициентов
    F = []  #правая часть уравнения
    M = np.zeros([n,n])
    for i in range(n):
        s.append(a + i*h)
        F.append(f(s[i]))
        if i == 0 or i == n-1: #метод трапеции, выбор коэффициентов
            A.append(h/2)
        else:
            A.append(h)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 1 - A[i]*K(s[i], s[i])
            else:
                M[i][j] = - A[j]*K(s[i],s[j])
                
    U = np.linalg.solve(M, F) #нахождение функции u в узлах
    x = sp.symbols('x')
    u_x = 0

    for i in range(n): #получаем выражение
        u_x += A[i]*K1(x, s[i])*U[i]
    u_x = u_x + f1(x)
    
    if flag == 1:
        print("Функция u(x) =", u_x)
    u_x = sp.lambdify(x, u_x) #превращаем в функцию от аргумента x
    return u_x


#Этап 2

def norm(x, n):
    return (qudrature_method(n,0)(x) - qudrature_method(n-1,0)(x))**2

def gaussian_formulas(k, n, a, b): #подсчет приближенного интеграла для вычисления нормы
    if k == 5:
        c = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891]
        x = [0.0000000000000000, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640]
    if k == 6:
        c = [0.6612093864662645, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, -0.9324695142031521, 0.9324695142031521]
        x = [0.3607615730481386, 0.3607615730481386, 0.4679139345726910, 0.4679139345726910, 0.1713244923791704, 0.1713244923791704]
    I = 0
    for i in range(len(c)):
        I += c[i]*norm((b + a) * 0.5 + (b - a) * 0.5 *x[i], n)
    I = I*(b - a)*0.5
    return I


def adaptive_algorithm(n):
    alpha = a
    beta = b
    I = 0
    I_h = 0
    I_h2 = 0
    h = 0
    delta = 0
    k = 0
    k_max = 0
    while alpha < beta:
        h = beta - alpha
        I_h = gaussian_formulas(5, n, alpha, alpha+h)
        I_h2 = gaussian_formulas(6, n, alpha, alpha+h)
        delta = I_h2 - I_h
        if abs(delta) < eps*h/(b - a):
            k = 0
            alpha = beta
            beta = b
            I += I_h2 + delta
        else:
            beta = (alpha + beta)*0.5 
            k = k + 1
            if k > k_max:
                k = 0
                I += I_h2 + delta
                alpha = beta
                beta = b
        print("норма разности",I)
    return I

def check(n):
    a = 0
    b = 1
    while abs(adaptive_algorithm(n)) > eps:
        n += 3
    print("количество узлов =", n)
    return qudrature_method(n,1)

print(check(3))

"""
import matplotlib.pyplot as plt

def u(x): #действительная функция
    return np.cos(2*x)

x = np.arange(-2, 2, 0.05)

plt.plot(x, check(3)(x), color='r', label='приближенный график')
plt.plot(x, u(x), color='g', label='действительный график')
plt.xlabel("x")
plt.legend()
plt.show()
"""
#print(u(5))


