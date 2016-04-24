import numpy as np
import pandas as pd
import random as rnd
#import openpyxl
#from openpyxl import load_workbook

def readProblem(i):
#Read in a single problem, and then
#return A,b,c as numpy arrays
	
	filename = 'ProjectProblems.xlsx'
	sheetname = 'Sample Problems'
	
	xls = pd.ExcelFile(filename)
	b = xls.parse(sheetname, skiprows= ((i-1)*7), index_col=None, parse_cols=[5,6,7,8,9], header=None, na_values=['NA'])
	b = b.ix[0,:]
	b = b.as_matrix()
	c = xls.parse(sheetname, skiprows=(((i-1)*7) + 1), index_col=None, header = None, na_values=['NA'])
	c = c.ix[0,:]
	c = c.as_matrix()
	A = xls.parse(sheetname, skiprows=(((i-1)*7) + 2), index_col=None, header = None, na_values=['NA'])
	A = A.ix[0:4,:]
	A = A.as_matrix()

	return A, b, c


def feval(c,x):
#return objective function value at some point 'x'
	return np.dot(c,x)

def isFeasible(A,b,x):
#return boolean value representing feasibility of x	
	return (np.all(np.less_equal(np.dot(A,x),b)))

def generateTABUCandidates(x):
	swaps = []
	while np.count_nonzero(swaps) <= 2:
		swaps =  np.random.randint(2, size=50);
	cand = np.logical_xor(x, swaps)
	return cand
	
def antColony(c,x):
#Ant Colony (?) - insert docstring here

	return x

def TABUsearch(c,x):
#TABU Search- docstring here

	return x

def SimulatedAnnealing(A, b, c, x)
#takes model parameters A, b,c and an initial solution x
#returns best found x

#temperature -> t
#iterations at some temperature t -> r(t)

	change = True

	while change:
		change = False
		




	return x


#/--------------------/
#         MAIN
#/--------------------/

#will loop over all problems, here just #1
(A, b, c) = readProblem(1)

#Call heuristics -- 

#Evaluate a random x
x = np.random.randint(2, size=50)
print(isFeasible(A,b,x))
z = feval(c,x)
print(z)

y = generateTABUCandidates(x)
print(isFeasible(A,b,y))
z = feval(c,y)
print(z)

#Evaluate a trivial solution, 0 vector
x = np.zeros(50)
print(isFeasible(A,b,x))
z = feval(c,x)
print(z)


