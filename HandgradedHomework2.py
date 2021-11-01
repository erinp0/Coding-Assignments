#!/usr/bin/env python
# coding: utf-8

# # Handgraded Homework 2
# In this first hand-graded homework, you will be solving three questions.
# 
# In addition to some automated checks to make sure it works, we will be grading the quality and clarity of your code. Therefore, make sure that your code is clean and well commented. The criterion to apply is: If I give this to my classmate who has never seen my code before, can they follow what my code is doing and why?

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
# When you are working on this, please make sure to execute every bit of code that you write and make sure that it works as intended!
# 
# When you are done, you should run `Kernel->Restart & Run All` to make sure your code is running through without errors before submitting.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# You will need the following three imported functions in this homework
from cryptography_functions import is_prime, modular_inverse, convert_to_text

# The following two lines of code give you an overview of the content of the 
# module cryptography_functions as a printout 
import cryptography_functions 
help(cryptography_functions)

# You can also simply inspect the file cryptography_functions.py in jupyter


# In[3]:


# So you can now use the functions is_prime, modular_inverse, convert_to_text
# As an easy example let's print out the list of primes less that 100
print([p for p in range(100) if is_prime(p)])


# ## Question 1. (10pts)
# 
# In number theory, the prime number theorem gives an approximation of the number of primes up to a given integer. Let $n$ be a natural number and define $\Pi: \mathbb{N}\to\mathbb{N}$ so $\Pi(n)$ is the number of positive prime numbers which are less than or equal to $n$. Thus $\Pi(2)=1$, $\Pi(7)=4$ etc.
# 
# The prime number theorem states that for large enough $n$, $\Pi(n)$ is well-approximated by the following two functions:
# $$f(n) = \frac{n}{\log n} \text{ and } Li(n) = \int_2^n{\frac{1}{\log t} \,dt}.$$
# 
# Formally, the prime number theorem states that $$\lim_{n\to\infty}\frac{\Pi(n)}{f(n)}=1$$
# and $$\lim_{n\to\infty}\frac{\Pi(n)}{Li(n)}=1.$$
# 
# In this question you will graph these functions in the range where $n<3,000$ to numerically check the prime number theorem and compare the two approximations.

# In[ ]:


##Here is a graph of 1/log t.
x=np.linspace(2,300,1000)
y=1/np.log(x)
plt.plot(x,y)
plt.title('Graph of y=1/log(x) between x=2 and x=300')
plt.show()
##This is the derivative of Li(x)


# ### Part (a).
# 
# The function $Li$ above is called the logarithmic integral. The integrand $\frac{1}{\log t}$ does not have an elementary anti-derivative but can be integrated numerically.
# 
# Decide on a reasonable scheme to numerically integrate to calculate $Li$ and define a function `Li(n)` which  approximates the logarithmic integral. In choosing this method you should consider the following factors:
# - The function has to be a "close enough" approximation to allow you to compare $Li(n)$ to $\Pi(n)$. It will be "good enough" if it allows you to draw reasonable graphs in the questions below.
# - The function should not take too long to calculate. You will need to evaluate $Li(n)$ many times to draw a graph. There is a speed test cell below to help you judge this.
# - It may be useful to know that your estimate is either an overestimate or an underestimate of the true value.
# 
# In addition to defining the function, use the box below to explain the reasoning behind the numerical integration scheme you have chosen. (Please do not use LaTeX symbols in you answer as Noteable does not process them correcly.)
# 
# **Note:** Your numerical approximation does not need to be perfect. It's more important that you can justify any shortcuts that simplfy the numerical integral.

# --Explain your integration scheme. You may need to double click to be allowed to type.--
# --Please avoid using LaTeX symbols.--
# 
# I have decided to numerically integrate Li(n) using Simpson's rule, over the trapezoidal and midpoint methods. This is because they all have similar numbers of calculation steps (so similar speeds) however the Simpson method has a higher degree of accuracy. In fact the error bound for Simpson's rule is less than (M(n-2)^5)/180x^4, where M is the maximum size of the 4th derivative of Li(n) on (2,n) and x is the number of intervals chosen for the approximation. Unfortunately, with Simpson's Rule, you cannot infer whether the estimation is over or under the true value.

# In[ ]:


##Define a function for the logarithic integral.
##(You can define the function and experiment before writing the justification.)

# Here we will define a function Li(n) which will approximate the logarithmic function with Simpson's rule. As n<3000, 
# we will want choose a large enough number of intervals so that our approximation is accurate enough.

def Li(n):
    intervals = 2000 # This stores the number of intervals we want. 2000 is large enough to give us reasonable accuracy
                     # but doesn't make the code run too slowly.
    h = (n-2)/intervals # this line of code determines the step length of our approximation
    Sn = 0
    for i in range(intervals):  # For each interval we calculate the 1/log value and store it in f
        f = 1/np.log(2+(i*h))
        if i == 0 or (intervals-1):  # Using simpson's rule, the first and last f values (x0, xn-1) are multiplied by 1
            f1 = f
        if i%2 == 1: # Using simpson's rule, any odd term f values (ie x1, x3, etc) are multiplied by 4
            f1 = 4*f
        else:      # Using simpson's rule, any even term f values (ie x2, x4, etc) are multiplied by 4
            f1 = 2*f
        Sn += f1  # Here we sum all the f values after they are multiplied
    Sn = (Sn*h)/3  # Lastly, the sum is multiplied by 1/3 of the step width
    return Sn
    


# In[ ]:


##A unit test. You may use this to help gauge the level of precision required.
assert Li(3000) > 441 and Li(3000) < 443


# In[ ]:


##This is a speed test. Run this cell to see how long it takes to find Li(i) for all i between 2 and 3000.
##This tells you something about how long it will take to draw the graph below.
##Your code should take less than 20 seconds on average. (It could be quite a lot less.)
##This will depend on your computer but if it takes much more than that, see if there is a way to speed up the calculation.
def test_Li():
    for i in range(2,3000):
        Li(i)
        
get_ipython().run_line_magic('timeit', 'test_Li()')


# In[ ]:


#Other tests


# ### Part (b).
# 
# 
# Next you will need a way to find $\Pi(n)$. It will be enough to have a list of primes. Create a list (or a numpy array) called `primes` which contains all primes between $1$ and $3000$ in numerical order. (You may use the `is_prime` function from the auxiliary `cryptography_functions` module.)

# In[ ]:


primes=[]
for i in range(1,3001): # Inputs each integer 1 to 3000 (incl) into the is_prime function, adds to the prime list if True is returned
    if is_prime(i,num_wit=50) == True:
        primes.append(i)


# In[ ]:


assert primes[0]==2
assert primes[3]==7
assert len(primes)==430


# ### Part (c).
# 
# Draw a graph (with title, labelled axes, legend) of $n$ against
#  - $\Pi(n)$, the prime counting function;
#  - $f(n)=\frac{n}{\log n}$, the asymptote;
#  - $Li(n)$, the logarithmic integral.
#  
# Your graph should go up to $n=3,000$.
# 
# Your graph needs to be precise enough to compare these functions. You may use an x-range of just primes (i.e. the list from the previous part) rather than all integers and the resolution will be fine -- you should still have about $\Pi(3000)=430$ data points.

# In[ ]:


# YOUR CODE HERE
# This list will store the Li(n) values for all the primes
# in the range [0,3000]. This is so we can plot them against the primes later
li_values =[]
for i in primes:
    li_values.append(Li(i))

# The range is 1,431 as the first prime is 2, and the number of
# primes up to and including 2 is 1. 
prime_counting = np.arange(1,431,1)
    

# Plot of the the 3 graphs
plt.figure(figsize=(8,6))
# This plots the prime counting function against the primes list. 
plt.plot(primes,prime_counting,'-b',lw=2,label='$\prod$(n)')
# This plots the f(n) = n/log(n) function against the primes list
plt.plot(primes,primes/np.log(primes),'-r',lw=2,label='f(n)')
# This plots the Li(n) approximation against the primes list
plt.plot(primes,li_values,'-g',lw=2,label='Li(n)')
plt.xlabel('Natural Numbers (n)')
plt.ylabel('Number of primes up to and including n')
plt.legend()
plt.title('Approximations for the number of primes up to and inlcuding n')
plt.show()





# ### Part (d).
# 
# Draw a graph (with title, labelled axes, legend) of $n$ against
#  - $\frac{\Pi(n)}{Li(n)}$;
#  - $\frac{\Pi(n)\log n}{n}$;
#  - a constant function with value $1$.
#  
# Your graph should go up to $n=3,000$.
# 
# (Suggestion: As in the previous question, it is good enough to use an x-range of just primes rather than all integers.)

# In[ ]:


# This list stores the values of the prime counting function divided by the Li function, so that we can plot them later
# Because Li(2) = 0, we do not use the first prime 2 in the list, as dividing by 0 is undefined. Hence the first x entry is 3
a_list = [prime_counting[i]/li_values[i] for i in range(1,430)]

# This list stores the values of the prime counting function divided by the f(n)= n/logn function
# For continuity purposes, we will ignore n=2 again
b_list = [(prime_counting[i]*np.log(primes[i]))/primes[i] for i in range(1,430)]


plt.figure(figsize=(12,8))
plt.plot(primes[1:430],a_list,'-b',lw=2,label=r'$\frac{\prod(n)}{Li(n)}$')
plt.plot(primes[1:430],b_list,'-r',lw=2,label=r'$\frac{\prod(n)log(n)}{n}$')
plt.axhline(y=1, color='g', linestyle='-', label = 'y=1') # This line of code plots a horizontal line at y=1
plt.xlabel('n')
plt.ylabel('Function of n')
plt.legend()
plt.title('Graphs using the prime counting function')
plt.show()


# ### Part (e).
# What conclusion can you draw from these graphs about the relative rate of convergence of these two estimates?
# 
# *Note:* This part is worth 1 point so you don't need to write much.

# The prime counting function divided by Li values converges quicker to one, as shown by the graph and how close the blue line gets closer to the green horizontal. Perhaps if we ran for a larger n, the red line would also converge more to 1.

# ## Question 2. (10pts)
# 
# A bag contains several balls, each either red or blue.  At each step $n=1,2,3,\ldots$, we choose a ball uniformly at random from the bag (that is, in such a way that each ball in the bag has equal probability of being chosen).  Then, we put the chosen ball back in the bag __together with a new ball of the same color__.
# 
# For example, if the bag initially contains 2 red and 3 blue balls, and if at step 1 we happen to choose one of the 3 blue balls, then after step 1 the bag will contain 2 red and 4 blue balls.  Then, if on step 2 we choose a red ball, then the bag will then contain 3 red and 4 blue balls, and so on.
# 
# Let $X_n$ be __the proportion of red balls in the bag after $n$ steps__ (which is a random variable).  For instance, in the above example we would have $(X_0,X_1,X_2,\ldots)=(\tfrac25,\tfrac26,\tfrac37,\ldots)$.

# ### (a)
# It is known that $X_n$ converges to a limit $X_\infty$ as $n\to\infty$, and the limit $X_\infty$ is itself random.  Here you will produce a plot to illustrate this phenomenon, for a bag that __initially contains $1$ red and $1$ blue ball__.  You should simulate the random process for a reasonably large number of steps (between $100$ and $1000$, say), and plot a graph of the resulting proportions $X_n$ as a function of $n$.  You should do this for several independent random runs of the entire process from the same initial conditions, plotting all the resulting curves on the same axes. (Use at least $5$ runs, but not so many that the plot becomes too crowded).
# 
# You must ensure that the upper and lower limits of the vertical axis reflect the correct full range of possible values for a proportion (use `ylim` from `matplotlib.pyplot`), and your plot should have appropriate axis labels and title.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from random import *

R = 1 # Here we are setting up variables to act as counters for our numbers of each ball colour.
B = 1
n =500 # This is the number of balls chosen by us
plt.figure(figsize=(15,12))
colours = ['-b','r','g','k','y'] # These are a list of different colours for the graph plots
for j in range(5): # This line of code runs the whole n steps 5 times in total, to produce 5 lines
    X =[1/2] # This list will store the proportion of red balls after each step
    for i in range(n):
        chosen = random() # This assigns chosen a uniformly random number in [0,1]
        # According on the current proportion of red balls in the bag, the interval [0,1] is split into red and blue
        # So if chosen is equal to or falls below R/number_balls, then effectively we have selected a red ball and we 
        # put it back in the bag along with another red ball. Similarly, if it is greater than R/number_balls then
        # we have chosen blue.
        if chosen <= R/(R+B): 
            R +=1
        else:
            B += 1
        X.append(R/(R+B)) # This adds the new proportion to the list above
    # We then plot the n step process and select a colour from the list above
    plt.plot(np.arange(0,n+1,1),X,colours[j],lw=2,label='Run'+str([j]))
plt.xlabel('Number of selections')
plt.ylabel('Proportion of red balls after n selections')
plt.legend()
plt.ylim(0,1) # The proportion must be between (0,1]  
plt.title('Proportion of red balls at each step Xn')
plt.show()


# ### (b)
# 
# Investigate the distribution of the random limiting proportion $X_\infty$.  To do this, fix a fairly large $n$ (e.g. about $100$), sample $X_n$ a large number of times (e.g. a few thousand) and plot a histogram of the results.  You can assume that the distribution of $X_n$ is close to that of $X_\infty$.  Since $X_n$ can only take a finite set of possible values (because it is a fraction with a fixed denominator), you need to adjust the `bins` argument (and possibly also $n$) so that each bin contains (at least approximately) the same number of these possible values, so that the histogram appears reasonably smooth.
# 
# By examining the histogram, formulate a conjecture (that is, an informed guess) for the formula for the probability density function (pdf) of $X_\infty$, and plot a graph of your conjectured pdf on the same axes to show that it is a good fit.  Remember that a pdf of a random variable that takes values in $[0,1]$ is a non-negative function $f$ that satisfies $\int_{0}^{1} f(x) dx=1$.
# 
# Do all this for a bag that initially contains $r$ red and $b$ blue balls, for each of the three cases $$(r,b)=(1,1),(2,1),(2,2).$$  You answer should consist of three separate plots.  The first two pdf's are very simple continuous functions that are easy to guess. The third is a bit harder - it is a quadratic function with integer coefficients.
# 
# Your plots should be clearly labelled.  __Include your conjectured formula__ for $f$ in the labels or legend (you can use LaTeX in `pyplot` labels).

# In[ ]:


# Similar set up to question 2a
def proportion_n_steps(r,b): # creates a function which outputs the proportion after 200 steps, inputs are the number of
    # red and blue balls initially
    R = r # Here we are setting up variables to act as counters for our numbers of each ball colour.
    B = b
    for i in range(200): # Takes n balls out etc... as in 2a
        chosen = random() 
        if chosen <= R/(R+B): 
            R +=1
        else:
            B += 1
    X = R/(R+B) # After n steps, this is the proportion of red balls in the bag
    return X 

# For 1 red and 1 blue ball
x = [proportion_n_steps(1,1) for i in range(1000)]  # We run the function 1000 times and store each sample in the list x
plt.hist(x,edgecolor='white', bins=10, density=True)
plt.axhline(y=1, color='g', linestyle='-', label = 'f = 1') # This line of code plots unif[0,1] distribution
plt.title('Histogram for 1 red and 1 blue ball')
plt.ylabel('Density')
plt.xlabel('Proportion of red balls in the bag')
plt.legend()
plt.show()

# For 2 red and 1 blue balls
x2 = [proportion_n_steps(2,1) for i in range(1000)]  
plt.hist(x2,edgecolor='white', bins=10, density=True)
t = np.arange(0,1.1,0.1) 
plt.plot(t,2*t, label = 'f = 2x')  # This function has an area of 1 underneath, to satisfy pdf properties
plt.title('Histogram for 2 red and 1 blue ball')
plt.ylabel('Density')
plt.xlabel('Proportion of red balls in the bag')
plt.legend()
plt.show()

# For 2 red and 2 blue balls
x3 = [proportion_n_steps(2,2) for i in range(1000)]  
plt.hist(x3,edgecolor='white', bins=10, density=True)
t2 = np.arange(0,1.01,0.01) # To make the quadratic smoother, I have used more x values
plt.plot(t2,(-6*t2)*(t2-1),label='f = $-6x^2 +6x$')
plt.title('Histogram for 2 red and 2 blue ball')
plt.ylabel('Density')
plt.xlabel('Proportion of red balls in the bag')
plt.legend()
plt.show()


# ### (c)
# 
# Let $T$ be the number of steps until the first time a red ball is chosen.  For example, in the scenario described in part (a), the first chosen ball is blue and the second is red, so $T=2$ in this case.  Use repeated sampling to estimate the mean of the random variable $T$.  Do this for a bag that initially contains $r$ red and $b$ blue balls, for each of the four cases $$(r,b)=(1,1),(1,2),(2,1),(2,2).$$ 
# In each case, formulate a conjecture for the exact mean of $T$, based on your experiments.  In each case the exact answer is either an integer or infinity.  You should store __your conjectured exact answers__ in four variables `mean_1_1`, `mean_1_2`, `mean_2_1`, `mean_2_2` respectively.  If you believe the answer is infinity then you should set the variable to be the string `'Infinity'`. 

# In[ ]:


# YOUR CODE HERE
# This function uses a while loop to count how many steps it takes until the first red is drawn
def until_red(r,b):
    ball = True
    T = 0 # This is a counter for the number of steps until first red ball
    R = r # Initial number of red balls
    B = b # Initial number of blue balls
    while ball == True:
        chosen = random() 
        T += 1
        if chosen <= B/(R+B): # Selects blue proportionaly to how may blue are already in the bag
            B +=1 # Increases the blue counter by 1 if a blue ball is selected
        else: # If red is chosen, end the while loop
            break 
    return T


# For 1 red and 1 blue
x1 = [until_red(1,1) for i in range(1000)] # here we call our waiting time function 1000 times and store each T value in a list
mean_1_1_sample = (sum(x1))/len(x1) # this finds the mean waiting time
mean_1_1 = 'Infinity' # After testing the sample mean multiple times, the mean was very volatile, indicating it was likely
# to be infinity
# Theoretically the E(T) is a sum of 1/(n+1) from n=1 to n=inf, the summation does not converge, hence supports my thoughts 
# that the mean is infinite.

# For 1 red and 2 blue
x2 = [until_red(1,2) for i in range(1000)]
mean_1_2_sample = (sum(x2))/len(x2)
mean_1_2 = 'Infinity' # The sample mean for 1000 varied a lot between 20 and 200, the lack of consistency indicated the
# that the mean is Infinity.

# For 2 red and 1 blue
x3 = [until_red(2,1) for i in range(1000)]
mean_2_1_sample = (sum(x3))/len(x3) # This consistently produced around 1.8 to 2, so we set the mean to 2
mean_2_1 = 2

# For 2 red and 2 blue
x4 = [until_red(2,2) for i in range(1000)]
mean_2_2_sample = (sum(x4))/len(x4)  # This consistently produced around 2.8 to 3.3, so we set the mean to 3
mean_2_2 = 3


# In[ ]:


# checking the answers have the right form
for ans in [mean_1_1,mean_1_2,mean_2_1,mean_2_2]:
    assert ans=='Infinity' or ans==int(ans) 


# ## Question 3. (10pts)
# Alice has a problem. She would like Bob to send her an encrypted  message using the techniques from Lecture 3 of Week 10. However her computer only generates primes $p$ of at most $16$ binary bits (i.e. $p < 2^{16}$). She still wants a public key $(N,e)$ where ($e = 65537$ and) $N$ is a composite integer of at least $1024$ bits... as she thinks that this will be secure. So she decides to compute several (for example 15-30) primes of at most $16$ bits, and to randomly use one or more of each of these primes to make up the prime decomposition of an integer of bit length at least $1024$. She will then, as in the RSA protocol,  publish her public key $(N,e)$. 
# 
# **Reminder.** In the RSA protocol Alice would have had private key $(p,q)$ where $p$ and $q$ are primes such that $N = p \cdot q$ (i.e. $p^1 \cdot q^1$ is the prime decompositon of $N$). Here Alice will have as private key the dictionary that encodes the prime decomposition of $N$. So her key will be of the form 
# $\{p_1\text{:}\;e_1, p_2\text{:}\; e_2, \dots ,p_m\text{:}\; e_m\}$ where $\prod^m_{i=1} p_i^{e_i}$ is the prime decomposition of $N$.  
# 
# What Alice does not realise is that her modification to the RSA protocol makes it highly insecure. Indeed Eve, who has been eavesdropping (i.e. listening in) on the line of transmission between Bob and Alice, thinks she can now hack the ciphertext integer $c$ that Bob transmits to Alice so as to extract Bob's original message. 
# 
# **Note 1.** Bob proceeds precisely as in Lecture 3 of Week 10: given a message in the form of a string he uses the function `convert_to_integer` to convert this string to an integer $m$. He then computes the ciphertext   
# 
# $$
# c \,=\, m^e \;\left( \mathrm{mod}\; N  \right)\, 
# $$
# 
# and transmits $c$ to Alice (via the line on which Eve is eavesdropping). 
# 
# **Note 2.** You may use the functions imported from the module `cryptography_functions` in the cell at the beginning of this notebook. (So you need to run that cell.)  Make sure that you are conversant with the functions in this module as also the other functions used in lectures. 
# 

# ## Part (a).
# Eve needs a function that, on input `N`, outputs the prime decomposition of `N`. She decides to develop her own algorithm instead of using the function `decompose` from Week 9 which she thinks will have problems with very large integers (which it does in its present form). Her algorithm does the following. 
# 
# 1. It starts by setting variables `current_N` = `N` and `current_decomp` = `{}` (the empty  dictionary).   
# 2. It iterates through stages `p` = $2,3,4,5,\dots$ as follows. 
# 3. At stage `p` if it is not the case that `current_N` > 1 then the algorithm returns `current_decomp`. 
# 4. Otherwise at stage `p`, if `p` is prime, and `current_N` is divisible by `p` then, during stage `p` - supposing that `e` is the greatest integer such that `current_N` is divisible by `p**e` - the value of `current_N` is divided by `p**e` and the key, value pair `p: e` is added to `current_decomp`. The algorithm then proceeds to stage `p + 1`. 
# 5. If at stage `p` neither the conditions in 3 hold nor those in 4 hold, then nothing is done and the algorithm proceeds directly to stage `p + 1`. 
# 
# Implement Eve's algorithm as the function `eves_decompose`. Your function should take as input a positive integer `N` and output the dictionary representing the prime decomposition of `N`. 
# Your function should use the primality test `is_prime` from the module `cryptography_functions`.
# 
# **Note.** You may assume that `N` is a positive integer. I.e$.$ your function does not need to handle incorrect input. 

# In[17]:


# All your code must be contained in the definition of eves_decompose
def eves_decompose(N):
    current_N = N # These are our initial variables 
    current_decomp = {}
    i = 2 # As instructed we start at stage p = 2
    while current_N >1: # The while loop terminates when current_N = 1, meaning we have decomposed it fully into prime factors
        if is_prime(i) == True: # This checks whether i is a prime
            if current_N%i == 0: # If it is prime, it checks whether current N is divisible by i
                s = 1 
                while current_N%(i**s)==0: # This while loop calculates the larger power that i can be raised to, and is
                    s+=1 # still a factor of the current N. When the remainder is not equal to 0, we have found the highest power
                current_decomp[i] = s-1 # It stores the value i in the dictionary, with its corresponding power s-1
                # We use s-1 as our loops terminates when i**s does not divide into current N with no remainder, meaning the power
                # one before s is the largest.
                current_N = current_N//(i**(s-1)) # Here we update the current N value to the current one divided by the prime
                # factor i to the power s-1. The double slash speeds up the code dramatically as it tells python that the division
                # will produce an integer (which we already know)
        i += 1 # We then move to the i+1 stage
    return current_decomp


# In[18]:


# Testing area, using a 1030 bit number N as input  
N = int('''
83706564861599899653596475840424730458044600907771647007334050372336
09892280960094075716733849713226033323457029628024552980324599333370
00350684058261595028326433198358461905686764348015800175804034215489
64422648032179381441885225061637776302818897970437444034711022068655
35792022518811024903672288999022155493'''.replace('\n',''))
N_dict = {8273: 6, 9173: 4, 10847: 4, 11311: 4, 11621: 4, 11953: 1, 
          12479: 2, 12923: 3, 13331: 4, 17209: 6, 17827: 1, 22643: 3, 
          25693: 2, 28279: 5, 30059: 3, 40763: 1, 41149: 4, 42589: 6, 
          55529: 1, 56299: 3, 56659: 5}
# Your function should pass this test (and a hidden test)
assert eves_decompose(N) == N_dict
# There is one hidden test using a similarly large number as input


# In[19]:


# This cell is for your  own tests and rough work
# Here's an easy example. 
K = 3**4 * 17**2 * 23*5 * 37*1
eves_decompose(K)


# ## Part (b).
# In Lecture 3 or Week 10 we developed the function `rsa_decrypt` as defined in the cell below. 
# 
# **Warning.** Note that `pow(c,f,N)` is used below to compute $c^f$ (mod $N$). We do **not** use  `c**f % N` as this does not handle large numbers correctly and so can lead to errors.

# In[20]:


# Function from Lecture 3 of Week 10 for your information. 
def rsa_decrypt(c,p,q,N,e): 
    '''
    Given input (c,p,q,N,e) returns the RSA decryption of ciphertext
    c using private key (p,q) and public key (N,e). (We input N as a 
    parameter to avoid having to recompute N = p*q.)
    '''
    totient = N - (p + q) + 1       # This is (p-1)*(q-1)
    f = modular_inverse(e,totient)  # Note: f * e = 1 (mod totient)
    return pow(c,f,N)               # This is m = c**f (mod N) where c = m**e


# The point is that if Bob transmits $c \,=\, m^e \;\left( \mathrm{mod}\; N  \right)\,$ and $N = p \cdot q$ where $p$ and $q$ are prime, then `rsa_decrypt(c,p,q,N,e)` computes the integer $m$. Eve's idea is to modify this function so that it works for any composite positive number $N$ by replacing the two input parameters $p$, $q$ by a dictionary representing the prime decomposition of $N$. 
# 
# Implement Eve's idea as the function `eves_decrypt` which takes as input (`c`, `N_decomp`, `N`, `e`) where `c` is the transmitted ciphertext, (`N`,`e`) is Alice's public key and `N_decomp` is the dictionary representing the prime decomposition of `N`. Under the assumption that the first input has value $c \,=\, m^e \;\left( \mathrm{mod}\; N  \right)\,$ your function should return $m$. 
# 
# **Note 1.** You may assume that the input is as described in the preceding paragraph. I.e$.$ your function does not need to handle incorrect input. 
# 
# **Note 2.** You should use the function `modular_inverse` from the module `cryptography_functions` in your function `eves_decrypt`. However you should not use any other functions developed in lectures or tutorials. 

# In[21]:


# All your code must be contained in the definition of eves_decrypt
def eves_decrypt(c,N_decomp,N,e): 
    # In order to find the modular inverse f, we need to use the totient function on the prime decomposition
    # of N. The totient function is multiplicative and if p is prime, the totient function of p^e is equal to
    # (p^(e-1))(p-1).
    totient = 1
    for key in N_decomp: # Takes the product of all the totient functions of the prime factors of N
        totient = totient*((key-1)*(key**((N_decomp[key])-1)))
    f = modular_inverse(e,totient) # Where f*e = 1 mod (tot(N))
    return pow(c,f,N) # This is m = c**f(mod N) where c = m**e


# In[22]:


# This cell is for your  own tests and rough work
# An easy example its easy to come up with others... 
N = 3**4 * 13**6 * 19**2 * 53**8 
N_decomp = eves_decompose(N)
e = 65537 
m = 53458936789543654   
c = pow(m,e,N)
m_new = eves_decrypt(c,N_decomp,N,e)
assert m_new == m


# In[23]:


# Testing area using large integers. 
# The ciphertext transmitted by Bob is c = m**e (mod N1) for some m. 
c = int('''
44040993505419961079485909296703212068236418038487210630216568513457921
38336402714823913994482821220533646724844746539049498179119183080513149
31408838044424151820320215981324447571417572264574845615546729136893812
49131244618204208416492118915226940262437591117873293250397449419056765
862894569167485210222637280212955530851785230'''.replace('\n',''))
# Alice's public key is (N1,e). 
e = 65537
N1 = int('''
10230611115167627701967833246600529852209634787210723189242636200865741
18444371403714267935391853062750090843150978074427150943772945195279171
00280972327473328434991888940897882646502852180501590304145861856226481
21545694830420218027450318764061548271599311224935464351312228384098752
0276059163143000753961230876956414639120841839'''.replace('\n',''))
# The prime decomposition of N1 is N1_decomp. 
N1_decomp = {8237: 1, 8377: 6, 10343: 1, 12941: 4, 14563: 3, 14939: 1, 
            15991: 7, 16691: 5, 20011: 1, 20627: 1, 22039: 5, 22111: 2,
            25633: 7, 26189: 5, 32603: 6, 50159: 7, 52259: 3, 54721: 3, 
            62383: 7}
m_decrypt = 30786253411008024821942804353038798977485124256255665198
# This is true - i.e. c = m_decrypt**e (mod N1) - since m_decrypt == m
assert c == pow(m_decrypt,e,N1)
# This is the test of your function 
assert eves_decrypt(c,N1_decomp,N1,e) == m_decrypt
# The hidden test is similar to this. 


# ## Part (c).
# Eve has now got the necessary tools to try to hack Bob's ciphertext $c$, given that the key $(N,e)$ is public. Her idea is to develop an algorithm that uses the functions `eves_decompose` and `eves_decrypt` to compute $m$ and then converts $m$ back in to the original message using `convert_to_text` from Lecture 3 of Week 10 (and imported above). 
# 
# Implement this as the function `eves_hack` which takes as inputs either  (`c`, `N`, `e`, `verbose`) or simply (`c`, `N`, `e`) 
# where `c` is the ciphertext,  (`N`,`e`) is Alice's public key and `verbose` is a boolean variable used as a switch.  Your function should return the original message sent by Bob as a string. 
# 
# When the optional input parameter `verbose` is set to `True` your function should produce a nicely formatted print out (similar to those given in Lecture 3 or Week 10) showing the various 
# elements used and processed during the computation. 
# 
# **Note.** You can assume that the inputs are as described above. I.e$.$ your function does not need to handle incorrect inputs. 

# In[37]:


# All your code must be contained in the definition of eves_hack
def eves_hack(c,N,e,verbose=False): 
    # We want c^f (mod N) = m as m is our decrypted number, which we can then convert back into plaintext
    N_decomp = eves_decompose(N) # This calls our function from part a
    m = eves_decrypt(c,N_decomp,N,e)
    if verbose: # If true is entered as an input, the extra explanation is printed out, along with the decoded message
        print('Eve know\'s Alice\'s public keys, N and e, which respectively are\n\n'+str(N)+' and '+str(e))
        print('\nShe has also intercepted the ciphertext c, sent from Bob to Alice:\n\n'+str(c))
        print('\nWe first decompose N into it\'s prime factors:')
        print(N_decomp)
        print('\nWith this prime factorisation and the fact that the quotient function is multiplicative, we decrypt the ciphertext c to produce m:')
        print(str(m)+'\n\n')
        print('Finally, we use the convert_to_text function to convert m into plaintext:\n')
    return convert_to_text(m) # This outputs m as readable characters using a pre existing function
    


# In[38]:


# This cell is for your own tests and rough work


# In[39]:


# The output of this cell - i.e. the printout generated by your function 
# (since verbose=True below) will be hand graded.

# The ciphertext transmitted by Bob is c2. 
c2 = int('''
13765857727387973232479930349817413548909598342616602193701631284325656267
80701134620002785255145626349800021032002509266271558856266934655194975687
59307106770371716547021307786879941777280361623176540814670229006368058329
84911920670968761587938063429816954180041385102041952823773836621293529317
2747559925006181429813854'''.replace('\n',''))

# Alice's public key is (N2,e). 
e = 65537
N2 = int('''
45826872776221747553965304038569114157321898032045936109324798759092548505
31613377090447584756132759059985898756678241036420212547809537827143541861
23768838634031792685862961283953201133968017357971449925795092740936013316
34869263468578803287892075018530064166921208113922227263664667767248577655
3724508783300940735066371
'''.replace('\n',''))

# Eve now hacks ciphertext c2
eves_hack(c2,N2,e,True)


# In[40]:


# This cell contains a hidden test which checks that your function computes 
# and returns the original message (as a string) in another example. 

