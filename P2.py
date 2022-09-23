import numpy as np
import matplotlib.pyplot as plt

#%% 1

'''
busco resolver el sistema
y'(t)=2y(t)-5sen(t)
y(0)=1
con euler
'''

h = -0.51
yt = [1]
N = 10
t = np.linspace(0,2*np.pi,N+1)
xt = 2*np.sin(t)+np.cos(t)

for i in range(N):
  ti = t[i]
  yi = yt[i]
  yi1 = yi*(1 + 2*h) - h*5*np.sin(ti)
  yt.append(yi1)

plt.plot(t,yt,color='red')
plt.plot(t,xt,color='blue')

#%% 2

N = 100
h = 0.01
k = np.arange(0,11,1)
xt = []
t = np.linspace(0,10,N+1)

for i in k:
  xt = [k[i]]
  for n in range(N):
    tn = t[n]
    xn = xt[n]
    xn1 = xn*(1+h*((np.cos(tn)**2)-1/2)) + h*(5/2) - 5*h*(np.cos(tn))**2
    xt.append(xn1)
  plt.plot(t,xt)
  plt.title(f"Euler para x(0) = {k[i]}")
  plt.show()
  
#%% 3

xs = [1]
h = (1.9)*(10**(-2))
N = 100
ts = np.linspace(0, 1, N+1)

for n in range(N):
    tn = ts[n]
    xn = xs[n]
    xn1 = xn*(1-2*h*tn)
    xs.append(xn1)

plt.plot(ts,xs)
plt.title("Euler")
plt.show()
print(xs)

#%% 4

def euler(g,N,ti,tf,y0):
  h = (tf - ti)/N
  t = np.linspace(ti,tf,N+1)
  y = np.zeros(N+1)
  y[0] = y0
  for i in range(N):
    ti = t[i]
    yi = y[i]
    y[i+1] = yi + h*g(ti,yi)
  return h,t,y

#%% 5

# con Euler
H = [0.1, 0.0625, 0.05, 0.025, 0.01]
ehE = []

for h in H:
  N = int(1/h)
  t = np.linspace(0,1,N+1)
  yE = np.zeros(N+1)
  yE[0] = 1
  for i in range(N):
    yEi = yE[i]
    yE[i+1] = yEi*(1+h)
  plt.plot(t,yE)
  plt.title(f'Solucion para Euler con h = {h}')
  print(f'El error de aproximacion en t = 1 con h = a {h} es eh = {abs(np.e-yE[-1])}')
  ehE.append(abs(np.e-yE[-1]))
  plt.show()
  
logh = np.log(H)
logehE = np.log(ehE)

plt.plot(logh,logehE)
plt.title('grafico de log(eh) en funcion de log(h), para Euler')
plt.show()

# con Taylor orden 2

ehT2 = []

for h in H:
  N = int(1/h)
  t = np.linspace(0,1,N+1)
  yT2 = np.zeros(N+1)
  yT2[0] = 1
  for i in range(N):
    yT2i = yT2[i]
    yT2[i+1] = yT2i*(1+h+(h**2)/2)
  plt.plot(t,yT2)
  plt.title(f'Solucion para Taylor orden 2 con h = {h}')
  print(f'El error de aproximacion en t = 1 con h = a {h} es eh = {abs(np.e-yT2[-1])}')
  ehT2.append(abs(np.e-yT2[-1]))
  plt.show()
  
logh = np.log(H)
logehT2 = np.log(ehT2)

plt.plot(logh,logehT2)
plt.title('grafico de log(eh) en funcion de log(h) para Taylor')
plt.show()

#%% 6

lambdas = [1,10,50,100]

for ld in lambdas:
  def f(t,x):
    return (-1)*ld*x
  _,tt,yt = euler(f,50, 0, 20, 1)
  plt.plot(tt,yt)
  plt.title('explicito')
  plt.show()

N = 50
ti = 0
tf = 20
y0 = 1

for ld in lambdas:
  h = (tf - ti)/N
  t = np.linspace(ti,tf,N+1)
  y = np.zeros(N+1)
  y[0] = y0
  for i in range(N):
    ti = t[i]
    yi = y[i]
    y[i+1] = yi*(ld/(ld-h))
  plt.plot(tt,yt)
  plt.title('implicito')
  plt.show()
  
#%% 9

def euler_vect(g,N,M,ti,tf,y0):
  
  h = (tf - ti)/N
  t = np.linspace(ti,tf,N+1)
  y = np.zeros(M+1, N+1)
  y[0, :] = y0

  for m in range(M):
    for n in range(N):
      tn = t[n]
      ymn = y[m,n]
      y[m+1,n+1] = ymn + h*g(tn,ymn)
  return h,t,y

#%% 10.c

a = 0.25
b = 1
c = 0.01
t = np.linspace(0,10,num=10001)
h = 0.01
x0 = 80
y0 = 30
N = 10000
i = 0
x = [x0]
y = [y0]

while i<N:
    
    xi = x[i]
    yi = y[i]
    xi1 = (1-a*h)*xi + c*h*xi*yi
    yi1 = (1+b*h)*yi - c*h*xi*yi
    x.append(xi1)
    y.append(yi1)
    i += 1

plt.plot(t,x, color='red')
plt.plot(t,y, color='green')
plt.show()

plt.plot(x,y)
 
#%% 13

def Runge_Kutta_4(f,N,ti,tf,y0):
    
  h = 0.01
  t = np.linspace(ti,tf,N+1)
  
  x = np.zeros(N+1)
  x[0] = y0[0]
  y = np.zeros(N+1)
  y[0] = y0[1]
  
  for i in range(100):
      
    ti = t[i]
    yi = y[i]
    xi = x[i]

    k1 = f(ti, xi, yi)
    k2 = f(ti+h/2, xi+h*k1[0]/2, yi+h*k1[1]/2)
    k3 = f(ti+h/2, xi+h*k2[0]/2, yi+h*k2[1]/2)
    k4 = f(ti+h, xi+h*k3[0], yi+h*k3[1])

    x[i+1] = xi + h/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    y[i+1] = yi + h/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
   # y.append(yi1)
  return t,x,y

#13-10.c

def f_10(t,x,y):
    
    a = 0.25
    b = 1
    c = 0.01
    d = 0.01
    
    z = np.array([[x],[y],[x*y]])
    t = t
    
    A = np.array([[-a,0,c],[0,b,d]])
    Az = A@z
    
    return sum(np.transpose(Az))
    
x0 = 80
y0 = 30

z0 = np.array([x0,y0])
        
t_10,x_10,y_10 = Runge_Kutta_4(f_10, 100, 0, 10, z0)

plt.plot(t_10,x_10,color='red')
plt.plot(t_10,y_10,color='green')
plt.show()

plt.plot(x_10,y_10)
plt.show()