import numpy as np
from fractions import Fraction
from enum import Enum
import math

def example1(): return np.array([5,4,3]),np.array([[2,3,1],[4,1,2],[3,4,2]]),np.array([5,11,8])
def example2(): return np.array([-2,-1]),np.array([[-1,1],[-1,-2],[0,1]]),np.array([-1,-2,1])
def integer_pivoting_example(): return np.array([5,2]),np.array([[3,1],[2,5]]),np.array([7,5])

def exercise2_1(): return np.array([-6,-8,-5,-9]),np.array([[2,1,1,3],[1,3,1,2]]),np.array([5,3])

def exercise2_5(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,4])
def exercise2_6(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[1,2]]),np.array([-3,-1,2])
def exercise2_7(): return np.array([1,3]),np.array([[-1,-1],[-1,1],[-1,2]]),np.array([-3,-1,2])
def random_lp(n,m,sigma=10): return np.round(sigma*np.random.randn(n)),np.round(sigma*np.random.randn(m,n)),np.round(sigma*np.abs(np.random.randn(m)))
class Dictionary:
    # Simplex dictionary as defined by Vanderbei
    #
    # 'C' is a (m+1)x(n+1) NumPy array that stores all the coefficients
    # of the dictionary.
    #
    # 'dtype' is the type of the entries of the dictionary. It is
    # supposed to be one of the native (full precision) Python types
    # 'int' or 'Fraction' or any Numpy type such as 'np.float64'.
    #
    # dtype 'int' is used for integer pivoting. Here an additional
    # variables 'lastpivot' is used. 'lastpivot' is the negative pivot
    # coefficient of the previous pivot operation. Dividing all
    # entries of the integer dictionary by 'lastpivot' results in the
    # normal dictionary.
    #
    # Variables are indexed from 0 to n+m. Variable 0 is the objective
    # z. Variables 1 to n are the original variables. Variables n+1 to
    # n+m are the slack variables. An exception is when creating an
    # auxillary dictionary where variable n+1 is the auxillary
    # variable (named x0) and variables n+2 to n+m+1 are the slack
    # variables (still names x{n+1} to x{n+m}).
    #
    # 'B' and 'N' are arrays that contain the *indices* of the basic and
    # nonbasic variables.
    #
    # 'varnames' is an array of the names of the variables.
    
    def __init__(self,c,A,b,dtype=Fraction,ori_c=None):
        """
        Returns an (n+1)x(m+1) dictionary.
        ori_c is the coefficients of the original obj functions. It is only used for solving two-phase simplex.
        """
        # Initializes the dictionary based on linear program in
        # standard form given by vectors and matrices 'c','A','b'.
        # Dimensions are inferred from 'A' 
        #
        # If 'c' is None it generates the auxillary dictionary for the
        # use in the standard two-phase simplex algorithm
        #
        # Every entry of the input is individually converted to the
        # given dtype.
        m,n = A.shape
        self.ori_c = ori_c # original objective function.
        if self.ori_c is not None: 
            self.ori_z = np.empty(2+len(ori_c),dtype=dtype)
            self.ori_z[0] = dtype(0)
            self.ori_z[-1] = dtype(0)
            for j in range(0,len(ori_c)):
                self.ori_z[j+1]=dtype(ori_c[j])
            
        self.dtype=dtype
        if dtype == int:
            self.lastpivot=1
        if dtype in [int,Fraction]:
            dtype=object
            if c is not None:
                c=np.array(c,np.object)
            A=np.array(A,np.object)
            b=np.array(b,np.object)
        self.C = np.empty([m+1,n+1+(c is None)],dtype=dtype)
        self.C[0,0]=self.dtype(0)
        if c is None:
            self.C[0,1:]=self.dtype(0)
            self.C[0,n+1]=self.dtype(-1)
            self.C[1:,n+1]=self.dtype(1)
        else:
            for j in range(0,n):
                self.C[0,j+1]=self.dtype(c[j]) # fill in coef of obj fn
        for i in range(0,m):
            self.C[i+1,0]=self.dtype(b[i]) # fill in b values
            for j in range(0,n):
                self.C[i+1,j+1]=self.dtype(-A[i,j]) # fill in A values
        self.N = np.array(range(1,n+1+(c is None)))
        self.B = np.array(range(n+1+(c is None),n+1+(c is None)+m)) # Make all slacks basic initially.
        self.varnames=np.empty(n+1+(c is None)+m,dtype=object)
        self.varnames[0]='z'
        for i in range(1,n+1): # original vars
            self.varnames[i]='x{}'.format(i)
        if c is None:
            self.varnames[n+1]='x0'
        for i in range(n+1,n+m+1): # slacks
            self.varnames[i+(c is None)]='x{}'.format(i)
        
        # Initialize termination condition state. The program is considered terminated as soon as
        # one of these three conditions is True.
        self.isInfeasible = False
        self.isOptimal = False
        self.isUnbounded = False

        # Optional state
        self.isDegenerate = False

    def __str__(self):
        # String representation of the dictionary in equation form as
        # used in Vanderbei.
        m,n = self.C.shape
        varlen = len(max(self.varnames,key=len)) # max len of varname
        coeflen = 0
        # get the max len of the coeff by iter thru entire C (LP dictionary)
        for i in range(0,m):
            coeflen=max(coeflen,len(str(self.C[i,0])))
            for j in range(1,n):
                coeflen=max(coeflen,len(str(abs(self.C[i,j]))))
        tmp=[]
        if self.dtype==int and self.lastpivot!=1:
            tmp.append(str(self.lastpivot))
            tmp.append('*')
        tmp.append('{} = '.format(self.varnames[0]).rjust(varlen+3))
        tmp.append(str(self.C[0,0]).rjust(coeflen))
        for j in range(0,n-1):
            tmp.append(' + ' if self.C[0,j+1]>0 else ' - ')
            tmp.append(str(abs(self.C[0,j+1])).rjust(coeflen))
            tmp.append('*')
            tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        for i in range(0,m-1):
            tmp.append('\n')
            if self.dtype==int and self.lastpivot!=1:
                tmp.append(str(self.lastpivot))
                tmp.append('*')
            tmp.append('{} = '.format(self.varnames[self.B[i]]).rjust(varlen+3))
            tmp.append(str(self.C[i+1,0]).rjust(coeflen))
            for j in range(0,n-1):
                tmp.append(' + ' if self.C[i+1,j+1]>0 else ' - ')
                tmp.append(str(abs(self.C[i+1,j+1])).rjust(coeflen))
                tmp.append('*')
                tmp.append('{}'.format(self.varnames[self.N[j]]).rjust(varlen))
        return ''.join(tmp)

    def basic_solution(self):
        # Extracts the basic solution defined by a dictionary D
        m,n = self.C.shape
        if self.dtype==int:
            x_dtype=Fraction
        else:
            x_dtype=self.dtype
        x = np.empty(n-(1 + (self.ori_c is not None)),x_dtype) # We don't want to print out x0's value
        x[:] = x_dtype(0) # by default all zeros, so that those not basic but original vars have values zeros.
        for i in range (0,m-1): #iter m-1 times
            if self.B[i]<n: # orig vars, skip the slacks (slack var have indices >= n) or > n?
                if self.dtype==int:
                    x[self.B[i]-1]=Fraction(self.C[i+1,0],self.lastpivot)
                else:
                    x[self.B[i]-1]=self.C[i+1,0]
        return x

    def value(self):
        # Extracts the value of the basic solution defined by a dictionary D
        if self.dtype==int:
            return Fraction(self.C[0,0],self.lastpivot)
        else:
            return self.C[0,0]

    def pivot(self,k,l):
        # Pivot Dictionary with N[k] entering and B[l] leaving
        # Performs integer pivoting if self.dtype==int
        if self.dtype==int:
            return self.integer_pivot(k,l)
        else:
            return self.normal_pivot(k,l)

    def normal_pivot(self,k,l):
        # save pivot coefficient
        piv_coef = self.C[l+1,k+1]
        ent_ci = k+1 # col index of entering var in C
        lea_ri = l+1 # row index of leaving var in C
        
        b = self.C[l+1,0] # b val for this pivot row
        h, w = self.C.shape
        # Make the coef of entering var to be 1, by dividing the corr row by its neg coef
        self.C[l+1,:] = np.divide(self.C[l+1,:],-piv_coef)
        # swap var, meaning, coef of this cell updated
        self.C[l+1,ent_ci] = np.divide(self.dtype(1),piv_coef)
        # eliminate entering var in all other rows except it's own row
        for i in range(0,h):
            if i == l+1: # skip the row of entering var
                continue
            # coef of entering var in this row
            coef_t = self.C[i, ent_ci]
            # perform elimination on this row
            self.C[i, ent_ci] = self.dtype(0)
            self.C[i, :] += (self.C[l+1,:]*coef_t)
            # Check if the constant satisfies >= 0 constraint, ignore the first row which is obj fn
            if self.C[i,0] < self.dtype(0) and i != 0:
                self.setInfeasible(True)
                break
            # [For debugging purposes] Check degeneracy
            if self.C[i,0] == self.dtype(0):
                self.setDegenerate(True)
        self.C = self.C.astype(self.dtype)
        # If it is two phase, eliminate original entering var in the original obj function.
        if self.ori_c is not None:
            coef_s = self.ori_z[ent_ci] # temp var
            self.ori_z[ent_ci] = self.dtype(0)
            self.ori_z[:] += coef_s*self.C[l+1,:]
            self.ori_z[:] = self.ori_z[:].astype(self.dtype)
        # Update N and B
        t = self.N[k]
        self.N[k] = self.B[l]
        self.B[l] = t
    
    def integer_pivot(self,k,l):
        # save pivot coefficient
        piv_coef = self.C[l+1,k+1]
        ent_ci = k+1 # col index of entering var in C
        lea_ri = l+1 # row index of leaving var in C
        b = self.C[l+1,0] # b val for this pivot row
        h, w = self.C.shape
        #swap entering and leaving vars
        self.C[lea_ri,ent_ci] = self.dtype(-1)
        # Multiply all rows, except the pivot row, by the abs(piv_coef)
        for i in range(0,h):
            if i == lea_ri: #skip the pivot row
                continue
            self.C[i,:] *= self.dtype(abs(piv_coef))
            # coef of entering var in this row
            coef_t = self.C[i, ent_ci]
            # perform elimination on this row
            self.C[i, ent_ci] = self.dtype(0)
            self.C[i, :] += ((self.C[lea_ri,:]*coef_t)/self.dtype(abs(piv_coef))).astype(self.dtype)
            
            # Check if the constant satisfies >= 0 constraint, ignore the first row which is obj fn
            if self.C[i,0] < self.dtype(0) and i != 0:
                self.setInfeasible(True)
                break
            # [For debugging purposes] Check degeneracy
            if self.C[i,0] == self.dtype(0):
                self.setDegenerate(True)
            # Divide all rows, except pivot rows, by the last_pivot (assumed absolutized already)
            self.C[i,:] = (self.C[i,:]/self.dtype(self.lastpivot)).astype(self.dtype)
            self.C[i,:] = self.C[i,:].astype(int)

        # If it is two phase, eliminate original entering var in the original obj function.
        if self.ori_c is not None:
            self.ori_z *= self.dtype(abs(piv_coef))
            coef_s = self.ori_z[ent_ci] # temp var
            self.ori_z[ent_ci] = self.dtype(0)
            self.ori_z[:] += (coef_s*self.C[lea_ri,:]/self.dtype(abs(piv_coef))).astype(int)
            # Divide the orig obj fn by the last pivot
            self.ori_z = (self.ori_z/self.lastpivot).astype(int)
        # Update lastpivot
        self.lastpivot = self.dtype(abs(piv_coef))

        # Update N and B
        t = self.N[k]
        self.N[k] = self.B[l]
        self.B[l] = t
    
    # Some setter functions.
    def setUnbounded(self,value):
        self.isUnbounded = value

    def setInfeasible(self,value):
        self.isInfeasible = value

    def setOptimal(self,value):
        self.isOptimal = value
        
    def setDegenerate(self, value):
        self.isDegenerate = value

# util
def isAuxiliaryRequired(b):
    # It is required only if some b is non-neg
    for i in range(0,len(b)):
        if b[i] < 0:
            return True
    return False

class LPResult(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3

def bland(D,eps):
    # Assumes a feasible dictionary D and finds entering and leaving
    # variables according to Bland's rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    h,w = D.C.shape
    k=l=None
    # Pick the first encountered entering variable
    for j in range(1,w):
        if D.C[0,j] > eps: # change > to < for mini prob
            k = j
            break
    # If all coef in obj fn is <= 0, then it is optimal.
    if k==None:
        D.setOptimal(True)
        return None, None

    # Pick the first encountered leaving variable among the lowest ratios |bi/xj| where xj must be negative.
    s = math.inf # smallest ratio |bi/xj|, initialized to inf.
    for i in range(1,h):
        if D.C[i,k] == D.dtype(0) or abs(D.C[i,k]) <= eps: # skip var with zero coef to prevent ZeroDivision Error
            continue

        ratio = (D.C[i,0]/abs(D.C[i,k]))
        if D.C[i,k] < -eps and ratio < s:
            s = ratio
            l = i
    # If all coefs of entering var in all eqns are non-neg, it is unbounded.
    if l==None:
        D.setUnbounded(True)
        return None, None
    return k-1,l-1
    
def largest_coefficient(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Coefficient rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    h,w = D.C.shape
    k=l=None
    # Pick the largest encountered entering variable
    larg = 0 # largest coef in obj fn, initialized to zero.
    for j in range(1,w):
        if D.C[0,j] > eps and D.C[0,j] > larg: # change > to < for mini prob (for the 1st condition only)
            larg = D.C[0,j]
            k = j
    # If all coef in obj fn is <= 0, then it is optimal.
    if k==None:
        D.setOptimal(True)
        return None, None

    # Pick the first encountered leaving variable among the lowest ratios |bi/xj| where xj must be negative.
    s = math.inf # smallest ratio |bi/xj|, initialized to inf.
    for i in range(1,h):
        if D.C[i,k] == D.dtype(0) or abs(D.C[i,k]) <= eps: # skip var with zero coef to prevent ZeroDivisionError
            continue

        ratio = (D.C[i,0]/abs(D.C[i,k]))
        if D.C[i,k] < -eps and ratio < s:
            s = ratio
            l = i
    # If all coefs of entering var in all eqns are non-neg, it is unbounded.
    if l==None:
        D.setUnbounded(True)
        return None, None
    return k-1,l-1

def largest_increase(D,eps):
    # Assumes a feasible dictionary D and find entering and leaving
    # variables according to the Largest Increase rule.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0
    #
    # Returns k and l such that
    # k is None if D is Optimal
    # Otherwise D.N[k] is entering variable
    # l is None if D is Unbounded
    # Otherwise D.B[l] is a leaving variable
    k=l=None
    h,w = D.C.shape
    # Check optimality (optimal if all coef in the obj fn is negative)
    if (D.C[0,1:w] <= eps).all():
        D.setOptimal(True)
        return None, None
    z = -math.inf # This is the value of the objective function to be maximized. Initialized to negative inf because it's possible that all potential pivots in this iteration results in decreasing z. 
    for j in range(1,w):
        r = None # temporary row index for entering var candidate
        if D.C[0,j] > eps:
            # Find the leaving var which shd be the most restrictive one.
            s = math.inf # smallest ratio |bi/xi|, initialized to zero.
            coef = D.dtype(0) # absolute of coef of entering var in the suitable row.
            for i in range(1,h):
                # skip var with zero coef to prevent ZeroDivisionError
                if D.C[i,j] == D.dtype(0) or abs(D.C[i,j]) <= eps:
                    continue
                ratio = abs(D.C[i,0]/D.C[i,j])
                # found entering var candidate
                if D.C[i,j] < eps and ratio < s:
                    r = i
                    coef = abs(D.C[i,j])
                    s = ratio
            if s == math.inf: # all coefs in this column are non-negative
                D.setUnbounded(True)
                return None, None
            # substitute this leaving var into the obj function
            assert r is not None
            z_new = D.C[0,0] + D.C[0,j]*(np.divide(D.C[r,0],coef))
            if z_new > z:
                z = z_new
                k = j
                l = r
    assert k is not None
    return k-1,l-1

def makeFeasible(D,eps=0):
    """ Given initially infeasible D, convert it into a feasible dictionary by doing
    one pivot with variable x0 entering and the most infeasible (most negative) variable leaving the basis.
    """
    h,w=D.C.shape
    k = D.C.shape[1] - 1 # entering variable is x0, which has col index n+1 in the dict
    l = None
    ben = math.inf # benchmark for the most negative b value
    for i in range(1,h):
        if D.C[i,0] < -eps and D.C[i,0] < ben:
            ben = D.C[i,0]
            l = i
    assert l is not None, 'The dictionary is already feasible to begin with!'
    D.normal_pivot(k-1,l-1)

def dropX0(D):
    """ Drop by making the coef of X0 in the dict to be zero so the it wont be chosen as entering variable.
    Also make all coef of X0 in all rows zero"""
    for ind in range(0,len(D.N)):
        if D.varnames[D.N[ind]] == 'x0':
            D.C[0,ind+1] = 0
            for i in range(0,len(D.C)):
                D.C[i,ind+1] = D.dtype(0)



def lp_solve(c,A,b,dtype=Fraction,eps=0,pivotrule=lambda D: bland(D,eps=0),verbose=False):
    # Simplex algorithm
    #    
    # Input is LP in standard form given by vectors and matrices
    # c,A,b.
    #
    # eps>=0 is such that numbers in the closed interval [-eps,eps]
    # are to be treated as if they were 0.
    #
    # pivotrule is a rule used for pivoting. Cycling is prevented by
    # switching to Bland's rule as needed.
    #
    # If verbose is True it outputs possible useful information about
    # the execution, e.g. the sequence of pivot operations
    # performed. Nothing is required.
    #
    # If LP is infeasible the return value is LPResult.INFEASIBLE,None
    #
    # If LP is unbounded the return value is LPResult.UNBOUNDED,None
    #
    # If LP has an optimal solution the return value is
    # LPResult.OPTIMAL,D, where D is an optimal dictionary.

    # Check if auxiliary dict is required, if True, solve the first phase.
    if isAuxiliaryRequired(b):

        D = Dictionary(c=None,A=A,b=b,ori_c=c,dtype=dtype)
        makeFeasible(D)
        while True:
            k,l = pivotrule(D)
            if D.isInfeasible: # Check this b4 checking optimality becus the algo may setOptimal = True after setInfeable = True. The objc fn may have all coef <= 0 when some constants are negative.
                return LPResult.INFEASIBLE, None
            elif D.isUnbounded:
                return LPResult.UNBOUNDED, None
            elif D.isOptimal:
                D.setOptimal(False) # Optimal for first phase, reset for second phase.
                # Perform another check to make sure z is zero (because for aux prob it's only feasible iff z = 0)
                if D.C[0,0] == D.dtype(0) or abs(D.C[0,0]) <= eps:
                    break
                else:
                    return LPResult.INFEASIBLE, None
            D.pivot(k,l)
        # The dictionary is optimal for the auxiliary problem. We now reintroduce the original obj function.
        D.C[0,:] = D.ori_z[:]
        # Drop x0 by making its coefficient zero
        dropX0(D)
        
    else:
        D = Dictionary(c,A,b,dtype=dtype)
    while True:
        k,l = pivotrule(D)
        if D.isInfeasible: # Check this b4 checking optimality becus the algo may setOptimal = True after setInfeable = True. The objc fn may have all coef <= 0 when some constants are negative.
            return LPResult.INFEASIBLE, None
        elif D.isUnbounded:
            return LPResult.UNBOUNDED, None
        elif D.isOptimal:
            return LPResult.OPTIMAL, D
        D.pivot(k,l)
    return None,None
  
def run_examples():
    # Example 1
    c,A,b = example1()
    D=Dictionary(c,A,b)
    print("==========================================")
    print('Example 1 with Fraction')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print()

    D=Dictionary(c,A,b,np.float64)
    print("==========================================")
    print('Example 1 with np.float64')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print()

    # Example 2
    c,A,b = example2()
    print("==========================================")
    print('Example 2')
    print('Auxillary dictionary')
    D=Dictionary(None,A,b)
    print(D)
    print('x0 is entering and x4 leaving:')
    D.pivot(2,1)
    print(D)
    print('x2 is entering and x3 leaving:')
    D.pivot(1,0)
    print(D)
    print('x1 is entering and x0 leaving:')
    D.pivot(0,1)
    print(D)
    print()

    # Solve Example 1 using lp_solve
    c,A,b = example1()
    print("==========================================")
    print('lp_solve Example 1:')
    res,D=lp_solve(c,A,b,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

    # Solve Example 2 using lp_solve
    c,A,b = example2()
    print("==========================================")
    print('lp_solve Example 2:')
    res,D=lp_solve(c,A,b,dtype=np.float64,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

    # Solve Exercise 2.1 using lp_solve
    c,A,b = exercise2_1()
    print("==========================================")
    print('lp_solve Exercise 2.1:')
    res,D=lp_solve(c,A,b,pivotrule=lambda D: largest_increase(D,0))
    print(res)
    print(D)
    print()

    # Solve Exercise 2.5 using lp_solve
    c,A,b = exercise2_5()
    print("==========================================")
    print('lp_solve Exercise 2.5:')
    res,D=lp_solve(c,A,b,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

    # Solve Exercise 2.6 using lp_solve
    c,A,b = exercise2_6()
    print("==========================================")
    print('lp_solve Exercise 2.6:')
    res,D=lp_solve(c,A,b,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

    # Solve Exercise 2.7 using lp_solve
    c,A,b = exercise2_7()
    print("==========================================")
    print('lp_solve Exercise 2.7:')
    res,D=lp_solve(c,A,b,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

    #Integer pivoting
    c,A,b=example1()
    D=Dictionary(c,A,b,int)
    print('Example 1 with int')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x4 leaving:')
    D.pivot(0,0)
    print(D)
    print('x3 is entering and x6 leaving:')
    D.pivot(2,2)
    print(D)
    print()

    c,A,b = integer_pivoting_example()
    D=Dictionary(c,A,b,int)
    print('Integer pivoting example from lecture')
    print('Initial dictionary:')
    print(D)
    print('x1 is entering and x3 leaving:')
    D.pivot(0,0)
    print(D)
    print('x2 is entering and x4 leaving:')
    D.pivot(1,1)
    print(D)

    # Solve Exercise integer_pivoting_example using lp_solve
    c,A,b = integer_pivoting_example()
    print("==========================================")
    print('lp_solve integer pivoting example:')
    res,D=lp_solve(c,A,b,dtype=int,pivotrule=lambda D: largest_increase(D,eps=0))
    print(res)
    print(D)
    print()

# run_examples()
