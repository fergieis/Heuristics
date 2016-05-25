import numpy as np
import pandas as pd
#import random as rnd
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

def GAf(A,b,c,x):
#Evaluates objective for a single population member, penalizing down to zero by level of infeasibility if infeasible.  Takes LP-like model components in standard form for a maximization as inputs (with the exception of x, assumed to be a row-vector instead of a column vector)
	if all(np.dot(A,x.T)<=b):
		return np.dot(c,x.T)
	else: #penalize infeasibility
		return max(0,np.dot(c,x.T)-(max(c)*(sum(np.dot(A,x.T)-b))))
	


def GAInit(A,b,c,popsize):
#Generates and returns a population of random solutions, by rows.  Assumes objective function will penalize infeasibility, so that the entire space becomes feasible.  Does not require restriction to a convex feasible region, however algorithm will recall this function if no feasible solutions are found.
	x = np.random.choice([0,1], size=(popsize, len(c)))		
	return x

def GAEval(A,b,c,x):		
#Wrapper function for GAf that calls for each member of a population for some table of population members by rows.  Returns a vector of penalized objective function values for the entire population.		
	f={}	
	for i in xrange(x.shape[0]):
		f[i]=GAf(A,b,c,x[i,:])
	return f

def GAGen(A,b,c,x,f,popsize,kids,mrate):
#Performs all necessary operations for a single generation. Returns surviving population, its fitness/object fcn values, and the best-so-far values (both fitness and actual solution vectors).
	mutationrate = mrate
	total_generation = sum(f.values())	
	
	#If no population members are feasible or are at least "close" to feasible, then 	randomize the population to look for one (use random search rather than hope that 	some alleles are valuable in the current population)
	while total_generation <=0:
		x = GAInit(A,b,c,popsize)
		f = GAEval(A,b,c,x)
		total_generation = sum(f.values())	
	
	best_generation = np.argmax(f)			
	fitness = np.array(f.values())/total_generation

	xnew = np.random.choice(xrange(x.shape[0]),size=x.shape[0]-kids-1, p=fitness)	

	#ensure survival of best in generation to the next
	xnew = np.vstack((x[best_generation],x[xnew,:]))
			
	#generate the necessary number of children from the best 2 of 4 population members
	#Children are generated using a single point cross-over of genes
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

	#Mutation might be varied, but currently static
	for mutation in range(int(mutationrate*popsize)):	
		mutationcol = np.random.randint(0,len(c))
		mutationrow = np.random.randint(0, popsize)
		x[mutationrow,mutationcol] = int((1+x[mutationrow,mutationcol])%2)

	x = xnew
	f = GAEval(A,b,c,x)
	bsf = max(f.values())
	bs = x[np.argmax(f)]
	
	return x,f,bs,bsf

def GASearch(A,b,c,popsize,kids,mrate):
#Conducts a search using a Genetic Algorithm with the parameters given below.  These could be adjusted to tune the heuristic in an attempt to improve the solution. Returns a value (bestsofar) of the objective function and a vector (bestso) representing the best solution found by the algorithm upon terminating
	#popsize = 500		    #number of members in population
	#kids = 25		    #number of children to generate between generations
	#mrate = .05		    #mutation rate
	numgenerations = 1000	    #max number of generations
	max_accept_stagnation = 100 #max generations with no change in best value

	survivalpop = popsize-kids
	x=GAInit(A,b,c,popsize)
	f=GAEval(A,b,c,x)
	bestsofar = max(f.values())
	bestso = x[np.argmax(f),:]
	bs = np.zeros(len(c))
	bsf = 0
	hist = []
	hist.append(bestsofar)	
	counter = 0

	for gen in tqdm(xrange(numgenerations)):
	#for gen in xrange(numgenerations):  #to mute status bar
		(x,f,bs,bsf) = GAGen(A,b,c,x,f,popsize,kids,mrate)
		if bsf > bestsofar:
			bestsofar = bsf
			bestso = bs
			counter = 0
		else:
			counter += 1
		if counter > max_accept_stagnation:
			return bestso,bestsofar	

		#if gen % 100 == 0:		
		#	hist.append(bestsofar)		
	
	#could add hist to return history of every nth generation's bestsofar value
	return bestso, bestsofar	


#/--------------------/
#         MAIN
#/--------------------/

#print("prob"+"\t"+"popsize"+"\t"+"kids"+"\t"+"f")
for prob in xrange(1,11):
	(A, b, c) = readProblem(prob)
	print("GA Problem " + str(prob))
	(x,f) = GASearch(A,b,c,popsize=500,kids=50,.05)
	print(f)		



