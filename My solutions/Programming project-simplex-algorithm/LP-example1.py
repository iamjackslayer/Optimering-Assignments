import scipy.optimize
c = [-5,-4,-3]
A = [[2,3,1],
     [4,1,2],
     [3,4,2]]
b = [5,11,8]
print("========================")
print("example 1")
print(scipy.optimize.linprog(c,A,b))

A = [
     [2,1,1,3],
     [1,3,1,2]
]
b = [5,3]
c = [6,8,5,9]

# print(scipy.optimize.linprog(c,A,b))

# example 2
c = [2,1]
A=[[-1,1],[-1,-2],[0,1]]
b=[-1,-2,1]
print("========================")
print("example 2")
print(scipy.optimize.linprog(c,A,b))

# Integer pivoting example
c = [-5,-2]
A=[[3,1],[2,5]]
b=[7,5]
print("========================")
print("integer pivot example")
print(scipy.optimize.linprog(c,A,b))

# exercise 2_1
c = [6,8,5,9]
A=[[2,1,1,3],[1,3,1,2]]
b=[5,3]
print("========================")
print("example 2_1")
print(scipy.optimize.linprog(c,A,b))

# exercise 2_5
c = [-1,-3]
A=[[-1,-1],[-1,1],[1,2]]
b=[-3,-1,4]
print("========================")
print("example 2_5")
print(scipy.optimize.linprog(c,A,b))

# exercise 2_6
c = [-1,-3]
A=[[-1,-1],[-1,1],[1,2]]
b=[-3,-1,2]
print("========================")
print("example 2_6")
print(scipy.optimize.linprog(c,A,b))

# exercise 2_7
c = [-1,-3]
A=[[-1,-1],[-1,1],[-1,2]]
b=[-3,-1,2]
print("========================")
print("example 2_7")
print(scipy.optimize.linprog(c,A,b))

import numpy as np
i = 13603
while True:
     i += 1
     np.random.seed(i)
     n = 200
     m = 200

     c = np.random.randint(-100,100,(n))
     A = np.random.randint(-20,100,(m,n))
     b = np.random.randint(300,1000,(m))
     res = scipy.optimize.linprog(c,A,b)

     if res.status == 0:
          print(res.status)
          print('objective function: {}'.format(res.fun))
          print('decision vars: {}'.format(res.x))
          print('c: {}'.format(c))
          print(i)
          break
     if i % 1000 == 0:
          print(i)