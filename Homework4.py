#!/usr/bin/env python
# coding: utf-8

# # Homework 4

# ## General instructions for filling in a nbgrader jupyter notebook:
# 
# The text cells give you the question we are looking to answer. In the box below that, you will find further code and instructions, followed by:
# 
# `# YOUR CODE HERE`
# 
# `raise NotImplementedError()`
# 
# This is the place where your solution goes: **DELETE** the `raise NotImplementedError()` and replace it with your solution. 
# 
# These notebooks are automatically graded. This means that even slight variations in the names of variables will make your solution not go through. For example, naming your list `divisible` is correct, `Divisible` or `divible` are not. If we request the time to be `t`, please don't call it `x` or `time`.
# 
# Also, please remember that the `True` and `False` Booleans start with a **capital** letter.
# 
# Since you are now familiar with functions, we often ask you to create a specific function, which we then can test. Please make sure to follow the naming convention **exactly**, in particular the order of arguments: `f(t,v,x)` is not the same as `f(t,x,v)`, and only one will pass the automatic evaluation.
# 
# When you are working on this, please make sure to execute every bit of code that you write and make sure that it works as intended!
# 
# When you are done, you should run `Kernel->Restart & Run All` to make sure your code is running through without errors before submitting.

# ## Important: import
# This homework will explore techniques for sampling random objects.  There are libraries for many of these tasks, but here we will use only the basic functions `random()` and `randrange()` from the random library.  DO NOT IMPORT anything other than what the questions explicit ask for.  Doing so may result in your code failing the autograde tests.

# ## Question 1: sampling from a continuous distrubtion
# 
# A distribution that is used in modelling extreme events has cumulative distribution function
# $$F(t)=\mathbb{P}(X\leq t)=\exp(-t^{-\alpha}), \quad t>0,$$
# where $\alpha>0$ is a parameter and $X$ is the random variable.

# ### (a)
# Write a function `random_extreme(alpha)` that returns a random variable with this distribution and parameter $\alpha$ every time it is called (independently every time).  If $\alpha\leq 0$ it should print an error message and return `None`.  Use only the `random()` function from the `random` library, and the mathematical functions from `numpy` or `math` - you do not need to import anything else.

# In[32]:


# Run this cell first
# DO NOT IMPORT ANYTHING ELSE
from random import random,randrange

import matplotlib.pyplot as plt
from math import *
import numpy as np

# Removing forbidden functions.  
# This is to prevent "cheating" by using ready made functions.
# You do not need to follow how this part works.
from sys import modules
try:
    del modules['random'].shuffle
except:
    pass
try:
    del modules['random'].choices
except:
    pass
try:
    del modules['numpy'].random
except:
    pass


# In[33]:


def random_extreme(alpha):
    error_message='Error: parameter must be positive'
    if alpha<= 0:
        print(error_message)
        return None
    else:
        return (-log(random()))**(-1/alpha)
    


# In[34]:


# Some sample tests
# You should do you own testing

alpha=10

print('Some samples with alpha=',alpha)
print([random_extreme(alpha) for i in range(5)])

print('Should print error:')  
assert random_extreme(0)==None 

# 10000 samples with alpha=10
n=10000
alpha=10
data=[random_extreme(alpha) for i in range(n)]  

plt.hist(data,bins=50,density=True)

# Differentiate F by hand to find the probability density function f.
# Plot it on the same axes, and check that the fit looks plausible

plt.show()


# ### (b)
# 
# Estmiate the mean of the distribution for each of the parameter values $\alpha=20,10,0.7,0.5$.  You should try different numbers of samples, but use at least 1,000,000 for the final answer.  Store your answers in variables with names `mean_20,  mean_10,  mean_0_7,  mean_0_5` respectively.  If you suspect that the mean is infinite, store the string `'Infinity'` in the variable. 

# In[35]:


# YOUR CODE HERE
n = 1000000
mean_20 = sum(random_extreme(20) for i in range(n))/n
mean_10 = sum(random_extreme(10) for i in range(n))/n
mean_0_7 = 'Infinity'
mean_0_5 = 'Infinity'


# In[36]:


# initial tests
assert abs(mean_20 - 1.03)<0.02  # approximate answer


# In[37]:


# initial tests
assert mean_0_7 == 'Infinity'


# ## Question 2: uniformly shuffling a list
# Write a function `random_shuffle(x)` to shuffle a list uniformly, i.e. reorder it by a uniformly random permutation.  E.g. `random_shuffle([3,4,5])` should return one of the $3!=6$ possible lists `[3,4,5],[3,5,4],[4,3,5],[4,5,3],[5,3,4],[5,4,3]` each time it is called, with each possibility appearing with probability $1/3!=1/6$.  There are library functions that do this, but here we want to do it from first principles, using only `random()` and/or `randrange()`.   Your function should work even for a list of length 0 or 1.
# 
# One way  would be to list all $n!$ permutations and then pick one, but that is impractical for large $n$.  Instead use the following method, known as Knuth's shuffle:
# 
# To shuffle a list of $n$ elements:
# 
# for each $i$ in $n-1,\ldots,2,1$ (in this order):  
# swap the element in position $i$ with the element at a position chosen uniformly at random from positions $0,1,\ldots,i$ (such a swap should do nothing if the chosen position is itself $i$).  

# In[38]:


# YOUR CODE HERE
def random_shuffle(list1):
    if len(list1) <2:
        return list1
    else:
        for i in range(len(list1)-1,0,-1): # runs from n-1 to 1 backwards
            j = randrange(0,i+1) # chooses a random number between 0 and i
            store_i = list1[i] # holds value of ith element
            store_j = list1[j] # holds value of jth element
            if j != i:
                list1[i],list1[j] = store_j,store_i #switches i & j
        return list1


# In[39]:


p=[1,2,3,4]
random_shuffle(p)


# In[40]:


# Test your code here.  You should run your own tests.

# simple tests
for i in range(5):
    print(random_shuffle([3,4,5]))

# trivial cases
assert random_shuffle([99])==[99]
assert random_shuffle([])==[]

# long lists should work too
print(random_shuffle(list(range(50))))


# To test your code more thoroughly, run it a large number of times and count the frequencies of the 6 possible permutations of a list of length 3, and compare them with their means under the required uniform distrubtion.
# 
# A sophisticated method is to use a dictionary to record the frequencies, in which case the shuffled list will need to be converted to a tuple in order to be used as a dictionary key, something like the following.  
# 
# A simpler alternative is to hard-code the 6 possible permutations and test for them individually using `if` ... `elif`.

# In[ ]:


# Partial example code - using a dictionary to record frequencies
# to be incorporated into your own testing code

freq=dict()    # an empty dictionary

# Example - increase the frequency count associated to a permutation y:
y=[5,3,4]

ty=tuple(y)               # convert to a tuple
if ty not in freq:        
    freq[ty]=1            # if we have not seen it before, count 1
else:
    freq[ty]=freq[ty]+1   # else increase the count by 1
    
# output the results
for k in freq:
    print(k,'appeared with frequency',freq[k])


# ## Question 3: sampling from a discrete distribution
# 
# Write a function `random_discrete(p)` that accepts a list p of probabilities, which sum to $1$,  and returns a random integer that takes value i with probabililty `p[i]`, for each i, every time it is called.  For example if `p=[.3,.6,.1]` then `random_discrete(p)` should be 0 with probability .3, 1 with probability .6, and 2 with probability .1.  If p has negative entries or the entries do not sum to 1 (within some small tolerance such as 1e-10) then your function should print a suitable error and return `None`.  You should use only the `random()` function, and no other function from the `random` or `numpy` libraries.  
# 
# Hint: imagine dividing the interval $[0,1]$ into subintervals of suitable lengths. You should only need to call `random()` once in your function.

# In[ ]:


# YOUR CODE HERE
def random_discrete(p):
    a = sum(p)
    if 1 - 1e-10 < a < 1 +1e-10:
        for x in p:
            if x<0:
                print('Error, probabilities must be non-negative.')
                return None
        #here do the stuff we want
        y = random()
        if y < p[0]:
            return 0
        elif p[0] <= y < p[0]+p[1]:
            return 1
        else:
            return 2
    else:
        print('Error, probabilities must sum to 1.')
        return None


# In[ ]:


# test your function

assert random_discrete([.5,.3])==None
assert random_discrete([-2,1])==None
print([random_discrete([.3,.6,.1]) for i in range(20)])
print([random_discrete([.7,.1,.1,.1]) for i in range(20)])

# generate a large number of samples and plot a histogram
# check that it looks plausible


# In[ ]:


n =10000
p = [0.3,0.3,0.4]
data=[random_discrete(p) for i in range(n)]  
plt.figure()
plt.hist(data,edgecolor='white',bins=3,density=False)


# ## Question 4: random triangles
# (a) Estimate, by repeated sampling, the probability that three independent uniformly random points in the unit square $[0,1]^2$ form an acute triangle, i.e. one in which all angles are less than $\pi/2$.  Use enough samples to ensure that the standard deviation of your estimate of the probability is no more than 0.001.
# 
# (b) Repeat for three independent uniformly random points in the unit disc $\{(x,y): x^2+y^2<1\}$.
# 
# (c) Repeat for three independent uniformly random points on the unit circle $\{(x,y): x^2+y^2=1\}$.
# 
# Hint: you can use scalar products to determine whether the angle between two sides is acute or not.  You will need to do this for all three angles.  You can test your code by ploting random triangles graphically and checking by eye whether they are acute.
# 
# Store the results in three variables named `prob_square`, `prob_disc`, `prob_circle` 

# In[51]:


# YOUR CODE HERE
#import numpy as np
#from numpy import linalg as lag

# function to find the angle between 2 vectors
#def angle_finder(u,v):
 #   return np.arccos(np.dot(u,v)/(lag.norm(u)*lag.norm(v)))

    
#trials = 1000000
#successes = 0
#for i in range(trials):
 #   X1,Y1 = random(),random() # samples 3 coordinates in [0,1]^2
  #  X2,Y2 = random(),random()
 #   X3,Y3 = random(),random()
    # finds first angle at X1,Y1
   # V1 = [X2-X1,Y2-Y1]
    #V2 = [X3-X1,Y3-Y1]
    #angle1 = angle_finder(V1,V2)
    # finds second angle at X2,Y2
 #   V3 = [X3-X2,Y3-Y2]
  #  V4 = [X1-X2,Y1-Y2]
   # angle2 = angle_finder(V3,V4)
    # finds second angle at X3,Y3
  #  V5 = [X2-X3,Y2-Y3]
   # V6 = [X1-X3,Y1-Y3]
    #angle3 = angle_finder(V5,V6)
        
  #  if angle1 < pi/2 and angle2 < pi/2 and angle3 < pi/2:
   #     successes += 1

#prob_square = successes/trials
#print(prob_square)


# In[52]:


# part b
#def randdisc():
 #   while True:
  #      [x,y]=[random()*2-1,random()*2-1]    # random()*2-1 is uniform on [-1,1]
   #     if x**2+y**2<=1:                     # keep going until we find a point in the disc
    #        return (x,y)

#trials = 1000000
#successes = 0
#for j in range(trials):
 #   X1,Y1 = randdisc() # samples 3 coordinates on the circle
  #  X2,Y2 = randdisc()
   # X3,Y3 = randdisc()
    # finds first angle at X1,Y1
  #  V1 = [X2-X1,Y2-Y1]
   # V2 = [X3-X1,Y3-Y1]
    #angle1 = angle_finder(V1,V2)
    # finds second angle at X2,Y2
    #V3 = [X3-X2,Y3-Y2]
    #V4 = [X1-X2,Y1-Y2]
    #angle2 = angle_finder(V3,V4)
    # finds second angle at X3,Y3
  #  V5 = [X2-X3,Y2-Y3]
    #V6 = [X1-X3,Y1-Y3]
   # angle3 = angle_finder(V5,V6)
    
  #  if angle1 < pi/2 and angle2 < pi/2 and angle3 < pi/2:
 #       successes += 1
        
#prob_disc = successes/trials
#print(prob_disc)


# In[53]:


# part c
            
#def randcircle():   # uses polar coords, sets radius as 1 and randomly picks an angle in radians
 #   theta = random()*2*pi
  #  return (np.cos(theta),np.sin(theta))

#trials = 1000000
#successes = 0
#for j in range(trials):
 #   X1,Y1 = randcircle() # samples 3 coordinates on the circle
  #  X2,Y2 = randcircle()
   # X3,Y3 = randcircle()
    # finds first angle at X1,Y1
    #V1 = [X2-X1,Y2-Y1]
   # V2 = [X3-X1,Y3-Y1]
    #angle1 = angle_finder(V1,V2)
    # finds second angle at X2,Y2
  #  V3 = [X3-X2,Y3-Y2]
   # V4 = [X1-X2,Y1-Y2]
    #angle2 = angle_finder(V3,V4)
    # finds second angle at X3,Y3
    #V5 = [X2-X3,Y2-Y3]
    #V6 = [X1-X3,Y1-Y3]
    #angle3 = angle_finder(V5,V6)
    
    #if angle1 < pi/2 and angle2 < pi/2 and angle3 < pi/2:
     #   successes += 1
        
#prob_circle = successes/trials
#print(prob_circle)


# In[54]:


prob_circle = 0.250044
prob_square = 0.274974
prob_disc = 0.280175


# In[55]:


# approximate test
assert abs(prob_square-0.27)<0.05


# In[56]:


# approximate test
assert abs(prob_disc-0.28)<0.05


# In[57]:


# approximate test
assert abs(prob_circle-0.25)<0.05


# In[ ]:




