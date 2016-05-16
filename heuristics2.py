import numpy as np
import pandas as pd
import random as rnd
from heapq import nlargest
from tqdm import tqdm
import pprint
import matplotlib.pyplot as plt
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
	return x

def GAEval(A,b,c,x):				
	f={}	
	for i in xrange(x.shape[0]):
		f[i]=GAf(A,b,c,x[i,:])
	return f

def GAGen(A,b,c,x,f,popsize,kids,mrate):
	mutationrate = mrate
	total_generation = sum(f.values())	
	while total_generation <=0:
		x = GAInit(A,b,c,popsize)
		f = GAEval(A,b,c,x)
		total_generation = sum(f.values())	
	best_generation = np.argmax(f)		

	fitness = np.array(f.values())/total_generation	
	fitness[1] = fitness[1] + (1-sum(fitness))	
	
	#fitness.update((k, v/total_generation) for k,v in fitness.items())
	#np.divide(f,total_generation)
	xnew = np.random.choice(xrange(x.shape[0]),size=x.shape[0]-kids-1, p=fitness)	
	xnew = np.vstack((x[best_generation],x[xnew,:]))
	child = np.empty(len(c))
	breakpoints = np.random.randint(0,len(c),kids)
	for breakpoint in breakpoints:
		parents = np.random.choice(xrange(popsize),size=4, p=fitness)
		tuples = []		
		for p in parents:
			tuples.append((p,(fitness[p])))
		parentlist = sorted(tuples,key=lambda tuples:-tuples[1])	
		parents = [parentlist[0][0],parentlist[1][0]]

		parents = x[parents,:]
		child[:breakpoint] = parents[0,:breakpoint]
		child[breakpoint:] = parents[1,breakpoint:]
		np.reshape(child,(1,len(c)))		
		xnew = np.vstack((xnew, child))

	for mutation in range(int(mutationrate*popsize)):	
		mutationcol = np.random.randint(0,len(c))
		mutationrow = np.random.randint(0, popsize)
		x[mutationrow,mutationcol] = int((1+x[mutationrow,mutationcol])%2)

	x = xnew
	f = GAEval(A,b,c,x)
	bsf = max(f.values())
	bs = x[np.argmax(f)]
	
	return x,f,bs,bsf

def GASearch(A,b,c):
	popsize=100
	kids = 20
	survivalpop = popsize-kids

	x=GAInit(A,b,c,popsize)
	f=GAEval(A,b,c,x)
	bestsofar = max(f.values())
	bestso = x[np.argmax(f),:]
	bs = np.zeros(len(c))
	bsf = 0
	hist = []
	hist.append(bestsofar)	
	mrate = .05
	for gen in tqdm(xrange(1000)):
		(x,f,bs,bsf) = GAGen(A,b,c,x,f,popsize,kids,mrate)
		if bsf > bestsofar:
			bestsofar = bsf
			bestso = bs
		if gen % 100 == 0:
			#print("Generation "+str(gen))			
			hist.append(bestsofar)		
	
	return bestso, bestsofar	


#/--------------------/
#         MAIN
#/--------------------/


for prob in xrange(1,12):
	(A, b, c) = readProblem(prob)
	print("GA Iteration Problem " + str(prob))
	(x,f) = GASearch(A,b,c)
	print(x)
	print(f)




