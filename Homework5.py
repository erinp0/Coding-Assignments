#!/usr/bin/env python
# coding: utf-8

# # Homework 5.

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

# In[40]:


##RUN THIS CELL FIRST
##This cell imports standard modules.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Removing forbidden functions.  
# This is to prevent "cheating" by using certain ready-made functions.
# You do not need to follow how this works.
from sys import modules
try:
    del modules['scipy.stats'].chi2_contingency
except:
    pass
try:
    del modules['scipy.stats'].chisquare
except:
    pass
try:
    del modules['scipy.stats'].contingency
except:
    pass
try:
    del modules['scipy.stats'].friedmanchisquare
except:
    pass
try:
    del modules['scipy.stats'].fisher_exact
except:
    pass


# In[41]:


##These functions are from the lecture notes in week 9.
##You may use them in your solutions to questions 2, 3 and 4. (It is very unlikely they will help for Q1.)

##(There is a case that using import from a module would have been the correct thing to do here.)

def isprime_basic(n,verbose=False): 
    '''
    Checks whether the argument n is a prime number using a brute force 
    search for factors between 1 and n. We made it verbose here for 
    illustration. (I.e. it prints out its results.)
    '''
    # First, 1 is not prime.
    if n == 1:
        return False
    # If n is even then it is only prime if it is 2
    if n % 2 == 0: 
        if n == 2: 
            return True
        else:
            if verbose:
                print("{} is not prime: {} is a factor. ".format(n,2))
            return False
    # So now we can consider odd numbers only. 
    j = 3
    rootN = n**0.5
    # Now check all numbers 3,5,... up to sqrt(n)
    while j <= rootN: 
        if n % j == 0:
            if verbose:
                print("{} is not prime: {} is a factor.".format(n,j))
            return False
        j = j + 2
    if verbose:
        print("{} is prime.".format(n))
    return True 


def gcd(a,b):
    """Returns the greatest common divisor of integers a and b using Euclid's algorithm.
    The order of a and b does not matter and nor do the signs."""
    if not(a%1 ==0 and b%1==0):
        print( "Need to use integers for gcd.")
        return None
    if b==0:
        return abs(a)                           #Use abs to ensure this is positive
    else:
        return gcd(b,a%b)
    
    
    
def gcd_ext(a,b):
    """Outputs (gcd,x,y) such that gcd=ax+by."""
    if not(a%1 ==0 and b%1==0):                         #Reject if trying to use for non-integers
        print( "Need to use integers for gcd.")
        return None
    if a == 0:                                          #Base case is when a=0.
        return (abs(b), 0, abs(b)//b)                   #Then gcd =|b| and is 0*a+1*b or 0*a-1*b. Use abs(b)//b
    else:
        quot=b//a                                       #The rule is that g=gcd(a,b)=gcd(b%a,a).
                                                        #Let b=qa+r where r=b%a
        g, x, y = gcd_ext(b%a, a)                       #And if  g=x1*r + y1*a then since r=b-qa
        return (g, y - quot * x, x)                     #We get g = a*(y1-q*x1)+x1*b.
                                                        #So x=y1-q*x1 and y=x1.


def smallest_factor(n):
    """Returns the smallest factor of a positive integer n."""
    sqrt=n**0.5
    i=2
    while i<=sqrt:
        if n%i==0:
            return i                            #If we get here, return i as the value.
        i+=1
    return n                                    #If we get through the whole while loop, return n.


def recompose(factor_dict):
    """Recomposes an integer from the factor dictionary."""
    result=1                                                #start with 1 and multiply
    for p in factor_dict.keys():
        result = result*(p**factor_dict[p])
    return result                                           #Stop once we've run through the whole dict


def decompose(n):
    """Generates a dictionary representing the prime decomposition."""
    factors={}
    current_number=n                            #divide current_number by the factor found found until it reaches 1
    while current_number > 1:
        p=smallest_factor(current_number)
        if p in factors.keys():                 #if p is not a new factor, increase the power
            factors[p]+=1
        else:
            factors[p]=1                        #if p is a new factor, create a new entry
        current_number = current_number//p
    return factors


def make_mult_func(fun_pp):
    """Takes a function fun_pp that is only defined on prime power and returns the associated
    multiplicative function defined on all positive integers."""
    def mult_version(n):                        #The output should be another function -- this one
        n=decompose(n)
        result=1
        for p in n.keys():
            result*=fun_pp(p,n[p])              #Uses the value at a prime power.
        return result
    return mult_version                         #Return the function made.


# ## Question 1. (4 pts)
# 
# We are given a __contingency table__, containing the counts of individuals categorized according to two factors.  For example, the following table gives numbers of patients in a New York hospital categorized by blood type (A, AB, B or O) and COVID-19 test result.
# 
# |   |  positive test | negative test  |
# |:---:|:---:|:---:|
# | A  | 231  | 245  |
# | AB  | 21  | 47  |
# | B  | 116  |  136 |
# | O  |  312 | 449  |
# 
# We want to determine whether there is evidence of statistical dependence between the two factors, in this case blood type and test result.  The null hypothesis is that the two factors are independent; the alternative hypothesis is that they are not. 
# 
# For each cell of the table, we compute its __expected count__ under the null hypothesis, which is given by:
# $$\mbox{expected count}=\frac{\mbox{row total}\times\mbox{column total}}{\mbox{grand total}}.$$
# Here "row total" is the sum of all the counts in the row containing the cell, and similarly for the column, and "grand total" is the total of all counts in the table.  We then compute the test statistic:
# $$s^2=\sum_{\mbox{cell}} \frac{(\mbox{observed} - \mbox{expected})^2}{\mbox{expected}}.$$
# Here the sum is over all cells in the table, "observed" is the count in the cell in the original table, and "expected" is the expected count computed above. 
# 
# Under the null hypothesis, $s^2$ has approximately a chi-squared distribution with $(\mbox{rows}-1)(\mbox{columns}-1)$ degrees of freedom (where "rows" and "columns" are the numbers of rows and columns in the table, e.g. $4$ and $2$ in the example above).  The distribution is `chi2` (note: _not_ `chisquare`!) in the `scipy` `stats` package.  The p-value is the probability that a random variable with this distribution is greater than or equal to the computed statistic $s^2$.  We reject the null hypothesis if the p-value is less than or equal to the required significance level $\alpha.$
# 
# The test involves an approximation.  It is not considered appropriate if any expected cell count is less than $5$.  (In that case there are other "exact" tests that can be used.)  

# ### (a)
# Write a function `expected(table)` that takes a two-dimensional contingency table of counts (with any numbers of rows and columns), in the form of a numpy array, and returns a numpy array of the same shape countaining the expected counts under the null hypothesis.

# In[42]:


def expected(table):
    n,m = np.shape(table)
    expected_table = np.zeros((n,m))
    col_totals = np.sum(table,axis=0) # sums columns and stores in an array
    row_totals = np.sum(table,axis=1) # sums rows and stores in an array
    grand_total = np.sum(table) # sums all the entries of the original array
    for i in range(n):
        for j in range(m):
            expected_table[i][j] = (row_totals[i]*col_totals[j])/grand_total
    return expected_table
            


# In[43]:


# Informal testing

t=np.array([[4,2,2],[7,3,2]])
print(t)
print(expected(t))
print(expected(expected(t)))  # applying the function twice should give same as once
assert np.allclose(expected(t),[[4.4,2,1.6],[6.6,3,2.4]])
assert np.allclose(expected(t),expected(expected(t)))


# In[44]:


# Autograde tests


# ### (b)
# Write a function `statistic(table)` that computes the test statistic $s^2$.

# In[45]:


def statistic(table):
    n,m = np.shape(table)
    stats = np.zeros((n,m))
    expected_array = expected(table)
    for i in range(n):
        for j in range(m):
            stats[i][j] = ((table[i][j]-expected_array[i][j])**2)/expected_array[i][j] # stores each value to be summed in 
            # an array called stats
    ssquared = np.sum(stats)
    return ssquared


# In[46]:


# Informal testing
print(statistic(np.array([[12,8],[9,6]])))
print(statistic(np.array([[6,10,4],[3,9,7]])))


# In[47]:


# Autograde tests


# ### (c)
# Write a function `p_value(table)` that computes the p-value.  If any expected cell count is less than $5$ it should instead print an error message and return `None`.

# In[48]:


def p_value(table):
    n,m = np.shape(table)
    e_values = expected(table)
    for i in range(n):
        for j in range(m): # checks every cell in the expected array to make sure entries are above 5.
            if e_values[i][j] < 5:
                print("Error, cell count less than 5.")
                return None
            else:
                pass
    s = statistic(table) # stores the s squared value in this variable
    df = (n-1)*(m-1) # calculates the degrees of freedom
    pval = 1 - stats.chi2.cdf(s,df)
    return pval


# In[49]:


#Autograde tests


# ### (d)
# Determine whether or not we can reject the null hypothesis (and thence conclude that there is evidence of dependence between blood type and test result) in the above example, at significance level $0.05$, and also significance levels $0.01$ and $0.001$.  Store the answers in three boolean variables (which should be `True` if and only if we can reject the null hypothesis) named `reject_5`, `reject_1` and `reject_01` respectively. 

# In[50]:


bloods = np.array([[231,245],
                  [21,47],
                  [116,136],
                  [312,449]])
p = p_value(bloods)
print(p)

a1 = 0.05
a2 = 0.01
a3 = 0.001

if p<=a1:
    reject_5 = True
else:
    reject_5 = False
    
if p<=a2:
    reject_1 = True
else:
    reject_1 = False

if p<=a3:
    reject_01 = True
else:
    reject_01 = False


# In[51]:


# A very simple test. It'll only fail if your syntax is wrong.
assert (reject_5==True or reject_5==False)
# Autograde tests


# ## Question 2. (2pts)
# 
# The Chinese remainder theorem implies that if $k$, $m$ and $n$ are pairwise coprime positive integers (so $\gcd(k,m)=\gcd(k,n)=\gcd(m,n)=1$) and $a,b,c \in \mathbb{Z}$ then all solutions $x$ to the 3 simultaneous congruences
# $$ x \equiv a \bmod k, \, x\equiv b \bmod m \text{ and }x \equiv c \bmod n,$$
# are congruent modulo $kmn$.
# 
# Write a function `solve_triple_congruence(a,k,b,m,c,n)` which takes 6 integers as its input (with $k,m,n>0$ and returns the unique value of $x$ in $[0,kmn-1]$ with 
# $$ x \equiv a \bmod k, \, x\equiv b \bmod m \text{ and }x \equiv c \bmod n.$$
# 
# (You may assume that $k,m,n$ are pairwise coprime. Your function only needs to work in this case.)

# In[80]:


def solve_triple_congruence(a,k,b,m,c,n):
    modulos = [k,m,n]
    rs = [a,b,c]
    M = k*m*n
    x = 0
    for mi, ri in zip(modulos, rs):
        Mi = M / mi
        _,y,_ = gcd_ext(Mi, mi)
        x += ri * y * Mi
    return ((x % M) + M) % M


# In[81]:


##Unit tests.

#A simple one
assert solve_triple_congruence(3,7,9,13,2,8) == 178


# In[82]:


#hidden test here.


# In[83]:


#Another hidden test.


# ## Question 3. (2pts)
# 
# The goal of this question is to write piece of code that checks if a given function is (or could be) a multiplicative function. A selection of the values of an arithmetic function can be presented as a dictionary with entries of the form `{ x:f(x) }`.
# An example is given below, for the function $f:\mathbb{N} \to\mathbb{N}$ with $f(n)=n^2$.
# 

# In[ ]:


sample_of_f={1:1,3:9,5:25,2:4,6:36}


# This dictionary will contain some but not all of the entries of the function. This means it may be possible to detect that the function is not multiplicative but you cannot definitely say a function is multiplicative. (For example if the sample contains $f(2)$ and $f(3)$ but not $f(6)$.)
# 
# Write a function `maybe_multiplicative(sample_dict)` that returns the Boolean `False` if the values given in the dictionary **cannot** come from a multiplicative function and return `True` if it is possible that the function is multiplicative.
# 
# (You may assume the input is dictionary with all keys being positive integers.)

# In[32]:


def maybe_multiplicative(sample_dict):
    x_vals, f_vals = zip(*sample_dict.items()) # makes 2 lists with xvals and fvals
    lis1 = []
    for i in range(len(x_vals)): # for each x value in the list, compare with every other x value in the list
        for j in range(len(x_vals)):
            # if i and j are coprime, then check fxy property?
            # if they are not move on
            greatest_divisor,_,_ = gcd_ext(x_vals[i],x_vals[j])
            if greatest_divisor == 1: # if they are coprime, check the f(x)f(y) = f(xy) property
                new_x = x_vals[i]*x_vals[j] # xy
                new_f = f_vals[i]*f_vals[j] # f(x)f(y)
                if new_x in x_vals: # checks if xy in x values
                    index = x_vals.index(new_x) # if it is, find its index in order to compare f(xy) to f(x)f(y)
                    if f_vals[index] != new_f: # if they are not equal, return False and end algorithm
                        return False
            else:
                pass
    return True # if they are equal for all new xs found in the list, then return true


# In[33]:


##Two non graded tests
sample_of_gcd_with_120={}
for i in [1,2,3,4,12,11,6,8,9,13,14,7,65,60,20,33,11,99]:
    sample_of_gcd_with_120[i]=gcd(120,i)
    ##Note f(n)=gcd(120,n) is a multiplicative function.
    
assert maybe_multiplicative(sample_of_gcd_with_120)==True

##(This is the sum of proper divisors.)
assert maybe_multiplicative({1:0,2:1,3:1,4:3,5:1,6:6,7:1,8:6,9:4})==False


# In[34]:


#A couple of hidden tests.


# In[35]:


#Another hidden test.


# ## Question 4. (2pts)
# 
# A positive integer is called 'square-free' if is not divisible by $p^2$ for any prime $p$. The Mobius function, $\mu(n)$ is a arithmetic function defined as follows:
# - $\mu(1)=1$;
# - $\mu(n) = 0$ if there is a prime $p$ such that $p^2$ divides $n$;
# - if $n$ is square-free and has an odd number of prime factors, $\mu(n)=-1$;
# - if $n$ is square-free and has an even number of prime factors, $\mu(n)=1$.
# 
# You are given that the Mobius function is multiplicative.
# 
# ### Part (a).
# Work out how to define $\mu$ on prime powers and use this to define `mobius_pp(p,e)` which returns $\mu(p^e)$ when $p$ is prime.

# In[50]:


import math
def mobius_pp(p,e):
    if p == 1:
        return 1
    else:
        if e == 0: # then p^e = 1 so mob function is 1
            return 1
        if e == 1: # then p^e = p and is square free, so need to decompose into primes
            decomposed = decompose(p)
            primes_freq = list(decomposed.values())
            number_of_primes = sum(primes_freq) # this holds the number of prime factors
            if number_of_primes % 2 == 1: # odd number of prime factors
                return -1
            else:
                return 1
        if e >= 2: # then p^2 divides p^e so mob function is 0
            return 0
        


# In[51]:


## hidden test


# ### Part (b).
# 
# Define a function `mobius(n)` that calculates $\mu(n)$.

# In[52]:


# USE FACT MOB IS MULTIPLICATIVE AND DECOMPOSE INTO PRIME FACTORS
def mobius(n):
    if n == 1:
        return 1
    factors = decompose(n)
    M = 1
    primes, primefreq = zip(*factors.items())
    for i in range(len(primes)):
        mob = mobius_pp(primes[i],primefreq[i])
        M = M*mob
    return M


# In[54]:


## hidden tests


# In[ ]:




