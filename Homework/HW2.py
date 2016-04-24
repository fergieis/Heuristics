

import numpy as np

#User Inputs
n=50 		#actually, |C| - number of DVs in instance
R = 100		#Range, R, from Pisinger
reps = 10	#number of reps

r_min = 1	#Memory of lowest correlation so fat

#repeat this the given number of times...
for i in xrange(0,reps-1):

	#weights w_j ~ uniform(1,R)
	w_j = np.random.uniform(1, R, (1,n))

	#benefits p_j are "weakly correlated"
	p_j = np.random.uniform(w_j-R/10, w_j+R/10, (1,n))
	r = np.corrcoef(w_j, p_j)
	print(r[0,1])
	if r[0,1]<r_min:
		r_min = r[0,1]
print("Minimum correlation found after %i repetitions was %.3f.\n" % (reps, r_min))

import matplotlib.pyplot as pp

Data = np.zeros((reps, 99))
for R in xrange(2,101):
	for i in xrange(0,reps-1):

		#weights w_j ~ uniform(1,R)
		w_j = np.random.uniform(1, R, (1,n))
	
		#benefits p_j are "weakly correlated"
		p_j = np.random.uniform(w_j-R/10, w_j+R/10, (1,n))
		Data[i, R-2] = np.corrcoef(w_j, p_j)[0,1]
		
pp.boxplot(Data, notch=True)		
	

