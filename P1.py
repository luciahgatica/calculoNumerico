import numpy as np
import matplotlib.pyplot as plt 

#%% epsilon de la maquina

eps = np.finfo(float).eps
print(f"el epsilon de la maquina vale {eps}")

#%% 1.a

p = 10**34
q = 1

print(f"{p+q-p}")

#da 0.0

#%% 1.b

p = 100
q = 10**(-15)

print(f"(p+q)+q vale {(p+q)+q}, y p+2q vale {p+2*q}")
print(f"((p+q)+q)+q vale {((p+q)+q)+q}, y p+3q vale {p+3*q}")

#en ambos casos ignora el valor de q a comparacion del de p

#%% 1.c

print(f"{0.1+0.2==0.3}")

#false

#%% 1.d

print(f"{0.1+0.3==0.4}")

#true

#%% 1.e

e = np.e
x = np.linspace(-4*e-8, 4*e-8, 1000)
fx = []

#cerca del 0, f(x) vale 0.5

for i in x:
    fi = (1-np.cos(i))/(i**2)
    fx.append(fi)
    
plt.plot(x,fx)
plt.xlabel("x")
plt.ylabel("f(x)")

#%% 1.f

print(f"{eps/2}")

#%% 1.g

print(f"{(1+eps/2)+eps/2}")

#no suma al epsilon sobre 2

#%% 1.h

print(f"{1+(eps/2+eps/2)}")

# considera al eps/2 al sumarlo a 1

#%% 1.i

print(f"{((1+eps/2)+eps/2)-1}")

# da 0.0

#%% 1.j

print(f"{(1+(eps/2+eps/2))-1}")

# da epsilon

#%% 1.k

j = np.arange(1, 26, 1)
gj = []

for k in j:
    gk = np.sin((10**k)*np.pi)
    gj.append(gk)
    
plt.plot(j, gj)
plt.xlabel("j")
plt.ylabel("g(j)")

# no representa bien a la funcion

#%% 1.l

hj = []

for l in  j:
    hl = np.sin(np.pi/2+(np.pi)*(10**l))
    hj.append(hl)
    
plt.plot(j, hj)
plt.xlabel("j")
plt.ylabel("h(j)")

# no representa bien a la funcion

#%% 7

# Suma de f(k), donde k son los números neturales + el cero, hasta n

def suma_coordenadas(v):
    #return v.sum()
    suma_v = 0
    for i in v:
        suma_v += i
    return suma_v
    
c = np.array([1,5,2,4,65,4,25,1])

suma_coordenadas(c)

print(f'{suma_coordenadas(c)}')
print(suma_coordenadas(c) == 107)

#%% 8
"""
Busco comparar la suma de un r aproximadamente 1, elevado al lugar k de la lista de longitud N 
y comparar los dos metodos Gn y Qn
"""

r = 1-10**(-16)
N = 20
n = 1
Gn = []

while n<=N:
    Gn.append(r**n)
    n += 1

suma_coordenadas(Gn)

Qn = (1-r**(N+1))/(1-r)

print(f"{suma_coordenadas(Gn)} vs {Qn}")

#%% 9

N = np.arange(0,101)
x = np.linspace(-13, -11, num = len(N))
Tn = [np.e**(-12)]

for n in N:
    Ti = ((np.e)**(-12))*((x+12)**n)/(np.math.factorial(n))
    Tn.append(Ti + Tn[n])
    #print(Tn)
    #plt.plot(x,Tn[n])
    
#suma_coordenadas(Tn)

#mejor aproximacion de e⁻¹² con propiedades de límites

M = 10000
aprox_e = (1+1/M)**(-12*M)
print(aprox_e)

#%% 10.b

def f(x):
  return x**2

h = 10**((np.arange(10,181,1)-190)/(10))

for i in h:
  dhfi = (f(1+i)-f(1))/i
  print(f'la derivada discreta de con h = {i} vale {dhfi}')

# los resultados son mas confiables para h entre 10^-15 y 10^-3