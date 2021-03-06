import numpy as np
import pandas as pd
import random as rnd
from heapq import nlargest
import pprint
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

def SimulatedAnnealing(A, b, c, x):
#takes model parameters A, b,c and an initial solution x
#returns best found x

#temperature -> t
#iterations at some temperature t -> r(t)
	change = True

	while change:
		change = False
		
	return x

def GAf(A,b,c,x):
	if all(np.dot(A,x.T)<=b):
		return np.dot(c,x.T)
	else: #penalize infeasibility
		return max(0,np.dot(c,x.T)-(max(c)*(sum(np.dot(A,x.T)-b))))
	


def GAInit(A,b,c,popsize):			
	x = np.random.choice([0,1], size=(popsize, len(c)))	
			
	f={}	
	for i in xrange(popsize):
		f[i]=GAf(A,b,c,x[i,:])
	return x, f

def GASearch(A,b,c):
	popsize=100
	survivalpop = 90
	kids = popsize - survivalpop

	(x,f)=GAInit(A,b,c,popsize)
	
	best_generation = np.argmax(f)
	total_generation = sum(f.values())
	fitness = np.array(f.values())/total_generation

	#fitness.update((k, v/total_generation) for k,v in fitness.items())
	#np.divide(f,total_generation)
	xnew = np.random.choice(xrange(popsize),size=popsize-kids, p=fitness)	
	xnew = x[xnew,:]

	child = np.empty(len(c))
	breakpoints = np.random.randint(0,len(c),kids)
	for breakpoint in breakpoints:
		parents = np.random.choice(xrange(popsize),size=2, p=fitness)
		parents = x[parents,:]
		child[:breakpoint] = parents[0,:breakpoint]
		child[breakpoint:] = parents[1,breakpoint:]
		np.reshape(child,(1,len(c)))		
		xnew = np.vstack((xnew, child))
	print(xnew.shape)
	#for i in xrange(popsize):
	#	f[i]=GAf(A,b,c,sol)
	
	max(f.values())
	return x, f	


#/--------------------/
#         MAIN
#/--------------------/

#will loop over all problems, here just #1
(A, b, c) = readProblem(1)

(x,f) = GASearch(A,b,c)



#Call heuristics -- 

#Evaluate a random x
x = np.random.randint(2, size=50)
print(isFeasible(A,b,x))
z = feval(c,x)
print(z)
print(GAf(A,b,c,x))


y = generateTABUCandidates(x)
print(isFeasible(A,b,y))
z = feval(c,y)
print(z)
print(GAf(A,b,c,y))

#Evaluate a trivial solution, 0 vector
x = np.zeros(50)
print(isFeasible(A,b,x))
z = feval(c,x)
print(z)
print(GAf(A,b,c,x))

