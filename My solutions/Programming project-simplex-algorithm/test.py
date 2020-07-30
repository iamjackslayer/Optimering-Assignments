import unittest
import numpy as np
import scipy.optimize
from fractions import Fraction

from simplex import lp_solve, Dictionary, bland, largest_coefficient, largest_increase, LPResult

from time import time
"""
# README
# To run the test, execute the following command line in the directory containing this file:
# > python test.py
# The code is implemented and tested using Python 3.7.3
"""
def timereps(reps, func):
    start = time()
    for i in range(0, reps):
        func()
    end = time()
    return (end - start) / reps

class TestBigLP1(unittest.TestCase):
    def setUp(self):
        np.random.seed(13603)
        n = 200
        m = 200
        self.c = np.random.randint(-100,100,(n))
        self.A = np.random.randint(-20,100,(m,n))
        self.b = np.random.randint(300,1000,(m))
        self.res = scipy.optimize.linprog(self.c,self.A,self.b)
    
    def test_solve_scipy(self):
        reps = 3
        duration = timereps(reps, lambda: scipy.optimize.linprog(self.c,self.A,self.b))
        print("TestBigLP1: It takes {} ms to run scipy".format(round(duration*1000,2)))

    def test_solve_float_bland(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)
    
    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)
    
    def test_solve_float_largest_increase(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)
    
    def test_solve_float_bland_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64))
        print("TestBigLP1: It takes {} ms to run float, bland".format(round(duration*1000,2)))

    def test_solve_float_largest_coefficient_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("TestBigLP1: It takes {} ms to run float, largest coefficient".format(round(duration*1000,2)))

    def test_solve_float_largest_increase_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0)))
        print("TestBigLP1: It takes {} ms to run float, largest increase".format(round(duration*1000,2)))

    def test_solve_fraction_bland_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: bland(D,eps=0)))
        print("TestBigLP1: It takes {} ms to run fraction, bland".format(round(duration*1000,2)))

    def test_solve_fraction_largest_coefficient_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("TestBigLP1: It takes {} ms to run fraction, largest coefficient".format(round(duration*1000,2)))

    def test_solve_fraction_largest_increase_benchmark(self):
        reps = 3
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_increase(D,eps=0)))
        print("TestBigLP1: It takes {} ms to run fraction, largest increase".format(round(duration*1000,2)))

class TestBigLP2(unittest.TestCase):
    def setUp(self):
        np.random.seed(13603)
        n = 20
        m = 20
        self.c = np.random.randint(-100,100,(n))
        self.A = np.random.randint(-20,100,(m,n))
        self.b = np.random.randint(300,1000,(m))
        self.res = scipy.optimize.linprog(self.c,self.A,self.b)

    def test_solve_float_bland(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)
        # self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)

    def test_solve_float_largest_increase(self):
        res,D=lp_solve(-self.c,self.A,self.b,dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -self.res.fun, 1)
    
    def test_solve_float_bland_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64))
        print("TestBigLP2: It takes {} ms to run float, bland".format(round(duration*1000,2)))

    def test_solve_float_largest_coefficient_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("TestBigLP2: It takes {} ms to run float, largest coefficient".format(round(duration*1000,2)))

    def test_solve_float_largest_increase_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D, eps=0)))
        print("TestBigLP2: It takes {} ms to run float, largest increase".format(round(duration*1000,2)))

    def test_solve_int_bland_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int, pivotrule=lambda D: bland(D,eps=0)))
        print("TestBigLP2: It takes {} ms to run int, bland".format(round(duration*1000,2)))

    def test_solve_int_largest_coefficient_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int, pivotrule=lambda D: largest_coefficient(D, eps=0)))
        print("TestBigLP2: It takes {} ms to run int, largest coeffient".format(round(duration*1000,2)))

    def test_solve_int_largest_increase_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int, pivotrule=lambda D: largest_increase(D, eps=0)))
        print("TestBigLP2: It takes {} ms to run int, largest increase".format(round(duration*1000,2)))

    def test_solve_fraction_bland_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: bland(D,eps=0)))
        print("TestBigLP2: It takes {} ms to run Fraction, bland".format(round(duration*1000,2)))

    def test_solve_fraction_largest_coefficient_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_coefficient(D, eps=0)))
        print("TestBigLP2: It takes {} ms to run Fraction, largest coeffient".format(round(duration*1000,2)))

    def test_solve_fraction_largest_increase_benchmark(self):
        reps = 10
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_increase(D, eps=0)))
        print("TestBigLP2: It takes {} ms to run Fraction, largest increase".format(round(duration*1000,2)))

class TestExample1(unittest.TestCase):
    def setUp(self):
        self.c = np.array([5,4,3])
        self.A = np.array([[2,3,1],[4,1,2],[3,4,2]])
        self.b = np.array([5,11,8])

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])
    
    def test_solve_fraction_bland_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b))
        print("It takes {} ms to run fraction, bland".format(round(duration*1000,2)))
    
    def test_solve_float_bland_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64))
        print("It takes {} ms to run float, bland".format(round(duration*1000,2)))

    def test_solve_int_bland_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int))
        print("It takes {} ms to run int, bland".format(round(duration*1000,2)))

    def test_solve_fraction_largest_coefficient_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("It takes {} ms to run fraction, largest_coefficient".format(round(duration*1000,2)))

    def test_solve_float_largest_coefficient_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("It takes {} ms to run float, largest_coefficient".format(round(duration*1000,2)))

    def test_solve_int_largest_coefficient_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int, pivotrule=lambda D: largest_coefficient(D,eps=0)))
        print("It takes {} ms to run int, largest_coefficient".format(round(duration*1000,3)))

    def test_solve_fraction_largest_increase_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=Fraction, pivotrule=lambda D: largest_increase(D,eps=0)))
        print("It takes {} ms to run fraction, largest_increase".format(round(duration*1000,3)))
    
    def test_solve_float_largest_increase_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0)))
        print("It takes {} ms to run float, largest_increase".format(round(duration*1000,3)))
    
    def test_solve_int_largest_increase_benchmark(self):
        reps = 1000
        duration = timereps(reps, lambda: lp_solve(self.c, self.A, self.b, dtype=int, pivotrule=lambda D: largest_increase(D,eps=0)))
        print("It takes {} ms to run int, largest_increase".format(round(duration*1000,3)))
    

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])
    
    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])

    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(13))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(0), Fraction(1)])

    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 13.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 0.0, 1.0])

class TestExample2(unittest.TestCase): #got aux
    def setUp(self):
        self.c = np.array([-2,-1])
        self.A = np.array([[-1,1],[-1,-2],[0,1]])
        self.b = np.array([-1,-2,1])

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(-3))
        self.assertEqual(list(D.basic_solution()), [Fraction(4,3), Fraction(1,3)])

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -3.0)
        self.assertAlmostEqual(list(D.basic_solution()), [1.3333333333333335, 0.3333333333333333])

    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(-3))
        self.assertEqual(list(D.basic_solution()), [Fraction(4,3), Fraction(1,3)])

    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -3.0)
        self.assertAlmostEqual(list(D.basic_solution()), [1.3333333333333335, 0.3333333333333333])

    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(-3))
        self.assertEqual(list(D.basic_solution()), [Fraction(4,3), Fraction(1,3)])

    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), -3.0)
        self.assertAlmostEqual(list(D.basic_solution()), [1.3333333333333335, 0.3333333333333333])

class TestExample2_1(unittest.TestCase): #no aux
    def setUp(self):
        c = [-6,-8,-5,-9]
        A=[[2,1,1,3],[1,3,1,2]]
        b=[5,3]
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(0))
        self.assertEqual(list(D.basic_solution()), [Fraction(0), Fraction(0), Fraction(0), Fraction(0)])


    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 0.0)
        self.assertAlmostEqual(list(D.basic_solution()), [0.0,0.0,0.0, 0.0])

    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(0))
        self.assertEqual(list(D.basic_solution()), [Fraction(0), Fraction(0), Fraction(0), Fraction(0)])

    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 0.0)
        self.assertAlmostEqual(list(D.basic_solution()), [0.0,0.0,0.0, 0.0])
    
    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(0))
        self.assertEqual(list(D.basic_solution()), [Fraction(0), Fraction(0), Fraction(0), Fraction(0)])

    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 0.0)
        self.assertAlmostEqual(list(D.basic_solution()), [0.0,0.0,0.0, 0.0])

class TestExample2_5(unittest.TestCase): #Aux
    def setUp(self):
        c = [1,3]
        A=[[-1,-1],[-1,1],[1,2]]
        b=[-3,-1,4]
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(5))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(1)])

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 5.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 1.0])

    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(5))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(1)])

    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 5.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 1.0])

    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(5))
        self.assertEqual(list(D.basic_solution()), [Fraction(2), Fraction(1)])

    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), 5.0)
        self.assertAlmostEqual(list(D.basic_solution()), [2.0, 1.0])

    def test_solve_integer_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=int, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertAlmostEqual(D.value(), Fraction(5))
        self.assertAlmostEqual(list(D.basic_solution()), [Fraction(2), Fraction(1)])

class TestExample2_6(unittest.TestCase): #Aux
    def setUp(self):
        c = [1,3]
        A=[[-1,-1],[-1,1],[1,2]]
        b=[-3,-1,2]
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)
    
    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)
    
    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)

    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)
    
    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.INFEASIBLE)
        self.assertIsNone(D)

class TestExample2_7(unittest.TestCase): #Aux
    def setUp(self):
        c = [1,3]
        A=[[-1,-1],[-1,1],[-1,2]]
        b=[-3,-1,2]
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)

    def test_solve(self):
        res,D=lp_solve(self.c,self.A,self.b)
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)

    def test_solve_float(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64)
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)

    def test_solve_fraction_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)
    
    def test_solve_float_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_coefficient(D,eps=0))
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)

    def test_solve_fraction_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)
    
    def test_solve_float_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=np.float64, pivotrule=lambda D: largest_increase(D,eps=0))
        self.assertEqual(res, LPResult.UNBOUNDED)
        self.assertIsNone(D)

class TestIntegerPivotingExample(unittest.TestCase):
    def setUp(self):
        c = [5,2]
        A=[[3,1],[2,5]]
        b=[7,5]
        self.c = np.array(c)
        self.A = np.array(A)
        self.b = np.array(b)

    def test_solve_integer_bland(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=int)
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(152,13))
        self.assertEqual(list(D.basic_solution()), [Fraction(30,13), Fraction(1,13)])
    
    def test_solve_integer_largest_coefficient(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=int, pivotrule=lambda D: largest_coefficient(D, eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(152,13))
        self.assertEqual(list(D.basic_solution()), [Fraction(30,13), Fraction(1,13)])
    
    def test_solve_integer_largest_increase(self):
        res,D=lp_solve(self.c,self.A,self.b, dtype=int, pivotrule=lambda D: largest_increase(D, eps=0))
        self.assertEqual(res, LPResult.OPTIMAL)
        self.assertIsNotNone(D)
        self.assertEqual(D.value(), Fraction(152,13))
        self.assertEqual(list(D.basic_solution()), [Fraction(30,13), Fraction(1,13)])
    



if __name__ == '__main__':
    unittest.main()
