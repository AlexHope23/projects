import numpy as np

eps = 0.001
p = 3
s = 4


def runge_kutta1(x, y, h):
    k_1 = h*f(x, y)
    k_2 = h*f(x + h/3, y + k_1/3)
    k_3 = h*f(x + 2*h/3, y + 2*k_2/3)
    return y + (k_1 + 3*k_3)/4

def fun1(x_0, x_n, y_0):
    x = []
    y = []
    h = []

    h_0 = (x_n - x_0)
    h.append(h_0)
    y.append(y_0)
    x.append(x_0)
    i = 0

    while x[i] <= x_n:
        #print(i)
        y_1 = runge_kutta1(x[i], y[i], h[i])
        y_2 = runge_kutta1(x[i]+ h[i], y[i], h[i])
        norm = np.linalg.norm(y_1 - y_2)
        #print(x[i], y[i], h[i], i)
        if norm/(1-2**(-p)) > eps:
            
            h[i] = h[i]/2
            
        else:
            x.append(x[i] + h[i])
            y.append(y_1)
            if norm/(1-2**(-p)) < eps/16:
                h.append(2*h[i])
            if eps/16 < norm/(1-2**(-p)) < eps:
                h.append(h[i])
            i += 1
        
    return  x[i-1], y[i-1], i-1



def runge_kutta2(x, y, h):
    k_1 = h*f(x, y)
    k_2 = h*f(x + h/4, y + k_1/4)
    k_3 = h*f(x + h/2, y + k_2/2)
    k_4 = h*f(x + h/2, y + k_1 - 2*k_2 + 2*k_3)
    return y + (k_1 + 4*k_3 + k_4)/6

def fun2(x_0, x_n, y_0):
    
    x = []
    y = []
    h = []

    h_0 = (x_n - x_0)
    h.append(h_0)
    y.append(y_0)
    x.append(x_0)
    i = 0

    while x[i] <= x_n:
        #print(i)
        y_1 = runge_kutta2(x[i], y[i], h[i])
        y_2 = runge_kutta2(x[i]+ h[i], y[i], h[i])
        norm = np.linalg.norm(y_1 - y_2)
        #print(x[i], y[i], h[i], i)
        if norm > eps:
            
            h[i] = h[i]/2
            
        else:
            x.append(x[i] + h[i])
            y.append(y_1)
            if norm < eps/16:
                h.append(2*h[i])
            if eps/16 < norm < eps:
                h.append(h[i])
            i += 1
    return  x[i-1], y[i-1], i-1


def runge_kutta3(x, y, h):
    k_1 = h*f(x, y)
    k_2 = h*f(x + h/3, y + k_1/3)
    k_3 = h*f(x + h/3, y + (k_1 + k_2)/6)
    k_4 = h*f(x + h/2, y + k_1/8 + 3*k_3/8)
    k_5 = h*f(x + h, y + k_1/2 - 3*k_3/2 + 2*k_4)
    E = (2*k_1 - 9*k_3 + 8*k_4 - k_5)/30
    return y + (k_1 + 4*k_4 + k_5)/6, E

def fun3(x_0, x_n, y_0):
    eps1 = 0.000001
    x = []
    y = []
    h = []

    h_0 = (x_n - x_0)
    h.append(h_0)
    y.append(y_0)
    x.append(x_0)
    i = 0

    while x[i] <= x_n:
        #print(i)
        y_1 = runge_kutta3(x[i], y[i], h[i])[0]
        norm = np.linalg.norm(runge_kutta3(x[i], y[i], h[i])[1])
        #print(x[i], y[i], h[i], i)
        #print(eps1/16, norm,  eps1)
        if norm > eps1:
            #print(1)
            h[i] = h[i]/2
            
        else:
            x.append(x[i] + h[i])
            y.append(y_1)
            if norm < eps1/16:
                h.append(2*h[i])
            if eps1/16 < norm < eps1:
                h.append(h[i])
            i += 1
    return x[i-1], y[i-1], i-1
def f(x, y):
    return np.array([1])

x_0 = 0.
x_n = 1.
y_0 = np.array([0.])
print(fun1(x_0, x_n, y_0))
print(fun2(x_0, x_n, y_0))
print(fun3(x_0, x_n, y_0))
