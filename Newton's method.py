import numpy as np 
from sympy import *
import math
import numdifftools as nd

eps = 0.0000001
vec = []
h = []
alpha = []
vec.append(np.array([2., 9.]))# x_0
x_k = 0
n = 0

flag = 0


# определяем функцию
def f(x):
    return x[0]**2 + x[1]**4

#2 нормы

def Norm1(point):
    sum = 0
    sc = point.dot(point)
    return sqrt(sc)

def Norm2(point1, point2):
    return abs(f(point1) - f(point2))

#ищем alpha_k
def found_alpha(x, h_p):
    lambd = 1/2
    mu = 2
    b = 4
    flag = 0
    a = b
    if f(x + a*h_p) < f(x):
        flag = 1
    if flag == 1:
        a = mu*b
        while f(x + a*h_p) < f(x + b*h_p):
            #print(f(x + a*h_p))
            a = a*mu
    else:
        a = lambd*b
        while f(x + a*h_p) >= f(x):
            #print(f(x + a*h_p), a)
            b = a
            a = lambd*b
    return a


#антиградиент
def agrad(x):
	return -nd.Gradient(f)(x)
#print(found_alpha(vec[0], agrad(vec[0])))

# проверка критерия окончания с погрешностью d

def check1(point2, point1, d):
    if Norm1(point2 - point1) < d and Norm2(point2, point1) < d and Norm1(agrad(point2)) < d:
        return 1
    else:
        return 0

def check2(point2, point1, d):
    if Norm1(point2 - point1) < d and Norm2(point2, point1) < d and Norm1(h2(point2)) < d:
        return 1
    else:
        return 0
# ищем уточнение приближения к минимуму


while flag == 0:
    n += 1
    for i in range(n-1, n):
        #print("vec[i]", vec[i]) 
        h.append(agrad(vec[i]))
        #print("h[i]", h[i])
        alpha.append(found_alpha(vec[i],h[i]))
        #print(alpha[i])
        vec.append(vec[i] + alpha[i]*h[i])
        #print(vec[i+1])
        #print(Norm1(vec[i+1] - vec[i]), Norm2(vec[i+1], vec[i]), Norm1(agrad(vec[i+1])),'\n')	
        if check1(vec[i+1], vec[i], sqrt(eps)) == 1:
            x_k = vec[i+1]
            flag = 1

print("x_k = ", x_k)
print("n = ", n)

#этап 2

# ищем матрицу вторых произодных в точке
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np

def hessian(x):
	H_f = jacobian(egrad(f)) 
	return H_f(x)

#print(hessian(vec[0]))

#вектор h_k 
def h2(x):
	b = np.linalg.inv(hessian(x))
	return np.dot(b,agrad(x))

#print(h2(vec[0]))

# метод деления отрезка пополам

def half_interval(x, h_p):
    a = 0
    b = 5
    n = 1
    eps1 = 0.1
    while(b-a-eps1) >= eps1:
        a1 = (a + b - eps1)/2
        a2 = (a + b + eps1)/2
        if f(x + a1*h_p) <= f(x + a2*h_p):
            b = a2
        if f(x + a1*h_p) > f(x + a2*h_p):
            a = a1
        n += 1
    return (a1 + a2)/2

#print(half_interval(vec[0], agrad(vec[0])))

flag = 0

while flag == 0:
    n += 1
    for i in range(n-1, n):
        #print("vec[i]", vec[i]) 
        h.append(h2(vec[i]))
        #print("h[i]", h[i])
        alpha.append(half_interval(vec[i], h[i]))
        vec.append(vec[i] + alpha[i]*h[i])
        #print(alpha[i])
        #print(vec[i+1])
        #print(Norm1(vec[i+1] - vec[i]), Norm2(vec[i+1], vec[i]), Norm1(h2(vec[i+1])),'\n')	
        if check2(vec[i+1], vec[i], eps) == 1:
            x_k = vec[i+1]
            flag = 1

print("x_k = ", x_k)
print("f(x_k) = ", f(x_k))
print("n = ", n)
