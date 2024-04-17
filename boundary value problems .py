import numpy as np

#задаем параметры краевой задачи

#точное решение
def f(x, y):
    return x*(x-1)*y*(y-1)

#элемент матрицы K, осталльные равны нулю
def k_11(x, y):
    return 1 + x**2 + y**2

#задаем прямоугольник двумя точками
B= np.array([0., 0.]) #левая нижняя точка
C= np.array([1., 1.]) #правая верхняя точка

#сетка
N_1 = 20 #x
N_2 = 20 #y

#верхняя и нежняя граница k_11(задаем сами)
M = 3
m = 1

#длины прямоугольников сетки
l_1 = C[0] - B[0]
l_2 = C[1] - B[1]
h_1 = l_1/N_1
h_2 = l_2/N_2

#создаем матрицу со значениями в узлах
U = np.zeros((N_1+1, N_2+1))

#сетчатая функция
F = np.zeros((N_1+1, N_2+1))

for i in range(1, N_1):
    for j in range(1, N_2):
        F[i,j] = f(B[0] + h_1*i, B[1] + h_2*j)

#print(F)

#метод Фурье, зависит от правой части
def B(R):
    u = np.zeros((N_1+1, N_2+1))
    mu_k2 = np.zeros((N_1+1, N_2+1))
    mu_k1k2 = np.zeros((N_1+1, N_2+1))
    v_k2 = np.zeros((N_1+1, N_2+1))

    for i in range(1, N_1):
        for k2 in range(1, N_2):
            for j in range(1, N_2):
                mu_k2[i, k2] += R[i, j]*np.sin(k2*np.pi*j/N_2)

    for k1 in range(1, N_1):
        for k2 in range(1, N_2):
            for i in range(1, N_1):
                mu_k1k2[k1, k2] += mu_k2[i, k2]*np.sin(k1*np.pi*i/N_1)
            
    for i in range(1, N_1):
        for k2 in range(1, N_2):
            for k1 in range(1, N_1):
                a1 = 4*(np.sin(k1*np.pi*h_1/(2*l_1)))**2/h_1**2
                a2 = 4*(np.sin(k2*np.pi*h_2/(2*l_2)))**2/h_2**2
                v_k2[i, k2] += mu_k1k2[k1, k2]*np.sin(k1*np.pi*i/N_1)/(a1 + a2)

    for i in range(1, N_1):
        for j in range(1, N_2):
            for k2 in range(1, N_2):
                u[i, j] += (4/(N_1*N_2))*v_k2[i, k2]*np.sin(k2*np.pi*j/N_2)
    return u

#определим оператор A
def A(u):
    L = np.zeros((N_1+1, N_2+1))   
    for i in range(1, N_1):
        for j in range(1, N_2):
            x = i*h_1
            y = j*h_2
            lambda_xx = (u[i+1, j]*(k_11(x, y) + k_11(x + h_1, y))/2 - u[i, j]*(2*k_11(x, y) + k_11(x+h_1, y)+k_11(x-h_1, y))/2 +  u[i-1, j]*(k_11(x, y) + k_11(x-h_1, y))/2)/h_1**2
            lambda_yy = (u[i, j+1] - 2*u[i, j] + u[i, j-1])/h_2**2
            L[i, j] = -(lambda_xx + lambda_yy)
    return L


#print(A, B)

#энергитическая норма и  скалярное произведение
def norm(v1):
    v2 = A(v1)
    s = 0
    for i in range(1, N_1):
        for j in range(1, N_2):
            s += v2[i, j]*v1[i, j]*h_1*h_2
    #print(s)
    return np.sqrt(s)

def sc(v1, v2):
    s = 0
    for i in range(1, N_1):
        for j in range(1, N_2):
            s += v1[i, j]*v2[i, j]*h_1*h_2
    return s

#метод наискорейшего спуска

def alpha(u):
    coef = sc(u, B(u))/sc(A(B(u)), B(u))
    return coef

#оценка погрешности
def check(n):
    if norm(z[n+1]) <= (M - m)/(M + m)*norm(z[n]):
        return 1
    else:
        return 0
    

z = []
r_arr = [] #сюда записываем норму A(U) - F
du = [] #сюда записываем норму U - F
coef = []

eps = 0.01

#задаем нулевое и первое приближение вручную, чтобы можно было оценить погрешность
#дальше по циклу

z.append(U)
r_arr.append(norm(A(U) - A(F)))
du.append(norm(U - F))
coef.append(alpha(A(U) - A(F)))
z.append(U - coef[0]*B(A(U) - A(F)))
U = z[1]
r = r_arr[0]
n = 0

eps = 0.01

while r > eps and check(n)==0:
    b = A(F)
    r = norm(A(U) - b)
    r_arr.append(r)
    du.append(norm(U - F))
    coef.append(alpha(A(U) - b))
    U = U - coef[n+1]*B(A(U) - b)
    z.append(U)
    n += 1
    
print("№,    ||Au - b||,      ||u - F||")
for i in range(n+1):
    print(i, du[i], r_arr[i])
