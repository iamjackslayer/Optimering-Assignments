import scipy.optimize
c = [-5,-4,-3]
A = [[2,3,1],
     [4,1,2],
     [3,4,2]]
b = [5,11,8]
print(scipy.optimize.linprog(c,A,b))
