import numpy as np
import pandas as pd
#import random as rnd
from heapq import nlargest
from tqdm import tqdm
#import pprint
#import matplotlib.pyplot as plt
import timeit
from gurobipy import *
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
		return max(0,np.dot(c,x.T)-abs(max(c)*(sum(np.dot(A,x.T)-b))))
	


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

	try: 
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
	except ValueError:
		#print("Numpy issue with cdf.")
		return x,f,x[0],f[0]	
	#print("bs:"+str(bs)+"\tbsfar:"+str(bsf))	
	return x,f,bs,bsf	

def GASearch(A,b,c,popsize,kids,mrate):
#Conducts a search using a Genetic Algorithm with the parameters given below.  These could be adjusted to tune the heuristic in an attempt to improve the solution. Returns a value (bestsofar) of the objective function and a vector (bestso) representing the best solution found by the algorithm upon terminating
	#popsize = 500		    #number of members in population
	#kids = 25		    #number of children to generate between generations
	#mrate = .05		    #mutation rate
	numgenerations = 500	    #max number of generations
	max_accept_stagnation = 50 #max generations with no change in best value

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
	count2 = 0
	
	#for gen in tqdm(xrange(numgenerations)):
	for gen in xrange(numgenerations):  #to mute status bar
		(x,f,bs,bsf) = GAGen(A,b,c,x,f,popsize,kids,mrate)
		if bsf > bestsofar:
			bestsofar = bsf
			bestso = bs
			counter = 0
			count2 = 0
		else:
			counter += 1
		if counter > max_accept_stagnation:
			mrate += .05	
			counter = 0
		#if gen % 100 == 0:		
		#	hist.append(bestsofar)		
	
	#could add hist to return history of every nth generation's bestsofar value
	return bestso, bestsofar	

def SAf(A,b,c,x):
#Evaluates objective for a single solution, penalizing down to zero by level of infeasibility if infeasible.  Takes LP-like model components in standard form for a maximization as inputs (with the exception of x, assumed to be a row-vector instead of a column vector)
	#if all(np.dot(A,x)<=b):
	return np.dot(c,x)
	#else: #penalize infeasibility
	#	return np.dot(c,x)-(max(c)*(sum(np.dot(A,x)-b)))

def KPQuickInit(A,b,c):
#More lazy than greedy...
	x = np.zeros(len(c))
	for i in xrange(len(c)):
		x[i] = 1
		if any(np.dot(A,x)<=b):
			x[i] = 0
	x = SARepairup(A,b,x)
	return x

def SARepairdown(A,b,x):
	while any(np.dot(A,x)>b):
		delta = np.random.randint(0,len(x))
		if x[delta]==1:
			x[delta]=0	
	return x

def SARepairup(A,b,x):
	s = x
	while all(np.dot(A,x)>b):
		delta = np.random.randint(0,len(x))
		if x[delta]==0:
			x[delta]=1
			if all(np.dot(A,x)<= b):
				s = x	
	return s


def SASearch(A,b,c,x,temp,alpha, final_temp,M):		
	x = KPQuickInit(A,b,c)	
	f_x = SAf(A,b,c,x)
	bso = x
	bsofar = f_x
	while temp>= final_temp:
		temp *= alpha # cooling schedule
		#if t % 1 == 0:		
		#	print("Temp:"+str(temp-t)+"\t Best so far:"+str(bsofar))
		for i in xrange(M):		
			s = x
			bit = np.random.randint(0,len(x))			
			r = np.random.uniform()		
			if s[bit]==0:
				s[bit]=1			
				s = SARepairdown(A,b,s)			
				f_s = SAf(A,b,c,s)
				delta = f_x - f_s
				if (delta <= 0) or (r<np.exp(delta/temp)):
					x = s
					f_x = f_s
					if f_s > bsofar:
						bso = s
						bsofar = f_s
			else:
				s[bit]=0
				s = SARepairup(A,b,s)
				delta = f_x - f_s
				if (delta <= 0) or (r<np.exp(delta/temp)):
					x = s
					f_x = f_s
					if f_s > bsofar:
						bso = s
						bsofar = f_s
		
	return (bso,bsofar)


#/--------------------/
#         MAIN
#/--------------------/

R = open("Results.csv","w")

R.write("Problem, Method, Param1, Param2, Param3, SolutionValue, Time\n")
GAParam1 = [100,200,500,750]
GAParam2 = [10,25,50,100]
GAParam3 = [.01,.05,.1]

SAParam1 = [500,750,1000,2000]
SAParam2 = [.85,.9,.95,.99]
SAParam3 = [1e-5,1e-3,1e-1]


for prob in xrange(1,11):
	(A, b, c) = readProblem(prob)
	print("Problem "+str(prob))
	for i in xrange(4):
		for j in xrange(4):
			for k in xrange(3):
				print(str(i)+"\t"+str(j)+"\t"+str(k))
				#Record each datapoint for GA Algorithm
				start = timeit.default_timer()
				(x,f) = GASearch(A,b,c,popsize=GAParam1[i],kids=GAParam2[j],mrate=GAParam3[k])		
				t = timeit.default_timer() - start
				R.write(str(prob)+", GA,"+str(GAParam1[i])+","+str(GAParam2[j]) + ","+str(GAParam3[k])+","+str(f)+","+str(t)+"\n")		

				#Record each datapoint for SA Algorithm
				start = timeit.default_timer()
				x_init = KPQuickInit(A,b,c)
				(x,f) = SASearch(A,b,c,x_init,SAParam1[i],SAParam2[j],SAParam3[k],len(c))
				t = timeit.default_timer() - start
				R.write(str(prob)+", SA,"+str(SAParam1[i])+","+str(SAParam2[j]) + ","+str(SAParam3[k])+","+str(f)+","+str(t)+"\n")	







