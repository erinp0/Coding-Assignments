#!/usr/bin/env python
# coding: utf-8

# # Handgraded Homework 1
# In this first hand-graded homework, you will be solving three questions. The first [..]. In the second question, you will be finding the minima of several functions, and then plot some of these functions as well. In the third question, you will be computing the powers of a matrix using several methods. 
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

# ## Question 1: $m$-ary Conversion.
# 
# **(a)** Write a function `convert` that, given a pair of integers `(number,base)` as input, with `number` being a non-negative integer and `base` being a member of the set $\{2,\dots,32\}$, returns a string representing `number` in `base`-ary notation. 
# Note that 2-ary is just binary and that, for example, the 5-ary string representation of `43` is `'133'` since $43 = 1 \cdot 5^2 + 3 \cdot 5^1 + 3 \cdot 5^0$. 
#  To represent the numbers $10,11,\dots,31$ in your string representation you should use the characters `'a','b',...,'u','v'`. (So, for example, `convert(59,12)` should return `'4b'`.) 
#  
# **Hint.** If instead of working with a `base` in the set $\{2,\dots,32\}$ you were asked to work with a `base` in the set $\{2,\dots,5\}$ then your `base`-ary digits can be found in the string `'01234'` as seen in the cell below. Also remember that integer division is implemented using `//`. For example `14 // 4` = `3` whereas `14 / 4` = `3.5`. With this in mind the cell below also gives you a hint on how to process the `number` you are trying to convert. 

# In[1]:


# Use a string of digits for number digits
base_digits = '01234'
b_string = base_digits[1] + base_digits[1] + base_digits[0] + base_digits[1]
f_string = base_digits[1] + base_digits[3] + base_digits[3]
print("13 in binary is " + b_string)
print("43 in 5-ary is " + f_string)
# Hint on how to process the first input of your function
print("The last digit of 43 in 5-ary is given by 43 % 5 =", 43 % 5)
print("Also 43 // 5 = ", 43 // 5)
print("But 8 is 13 in 5-ary! ")


# **Note.** You can assume that `number` is a non-negative integer and that `base` is a number in the set $\{2,\dots,32\}$.  I.e. there is no need to deal with exceptions.

# In[2]:


#This line of code allows us to access functions in the pre-coded python module called string. We will need this to create a
#list with the alphabet in
import string
#This list contains all the digits our function can produce initially, before we convert 10-31 into letters
raw_numbers = list(range(0,32))
#This creates a list with elements corresponding to the alphabet
alphabet1 = list(string.ascii_lowercase)
#This then restricts the alphabet to the first 22 letters as we only need letters corresponding to numbers 10 to 31
alphabet2 = alphabet1[0:22]
#This list holds the digits 0-9 as we don't need to assign them letters. then we convert all the integers into strings. 
#This becomes important later
first_10 = list(range(0,10))
string_first_10 = [str(x) for x in first_10]
#Here we combine 2 lists to form the final selection of digits for our function, this is a combination of letters and numbers
#and will be what our outputted number will be made up of
digitslist = string_first_10 + alphabet2

#Here we are defining a function which converts an inputted integer from decimal base to any given base up to 32-ary
def convert(number,base):
    #This part of the code makes sure that the number to be converted is non-negative and an integer
    #it also checks that the base chosen is between 2 and 32, if these conditions are not satisfied then
    #an error message is displayed and the function returns None
    if base not in range(33) or number<0 or type(number) is not int:
        print("Error, the number or base is in incorrect format.")
        return None
    #This part of the code runs if the number and base inputted are of the correct format
    else:
        #Here we create an empty list which is going to hold the initial digits for our converted number
        converted=[]
        #For the special case where number = 0, we have an elif statement as 0 is 0 in all base systems
        if number == 0:
            return '0'
        else:
            #The while function calculates the last digit and works backwards
            while number>0:
                z = number%base
                #This next line uses the value of z as an index and finds the corresponding digit in our digitslist to replace
                #z with.
                #If z is between 0-9 then it will stay the same, but if z is between 10-31, it will be replaced with a letter
                z2 = digitslist[z]
                #This next line adds the number/letter to the start of our list which is holding the digits 
                #for our converted number
                converted.insert(0,z2)
                number = number//base
                #When number equals 0, our while function stops as we have all the digits we need to represent the number inputted
            #This next line allows our variable answer to be a global variable, i.e. accessed outside of our function 
            global answer
            #Here we are joining the string elements in our list to form one string using the built in join function
            #For instance it changes '1','2','3' to '123' by joining the strings together
            answer = ''.join(converted)
        #This next line outputs the converted number when we run our function
        return answer

        


# In[3]:


# Casual testing area. You can run this cell multiple times.
# This cell is NOT graded. 
from random import randint
max_num = 100000
base = randint(2,32)
print("Input  Base  Converted")
print("=====  ====  =========")
for i in range(10): 
    n = randint(0,max_num)      # Define max in the previous cell 
    d = convert(n,base)     # Define base in the previous cell 
    print("{:<7}{:<6}{:<11}".format(n,base,d))


# In[4]:


# In each of the tests (including the hidden tests) convert(number,base) is such that 
# number is a non-negative integer and base is an integer in the interval [2,32]
assert convert(0,7) == '0'
assert convert(17,8) == '21'
assert convert(854,2) == '1101010110'
assert convert(183,5) == '1213'
assert convert(388,11) == '323'
assert convert(3496,32) == '3d8'
assert convert(5961,28) == '7gp'
assert convert(78932148,25) == '821gan'


# **(b)** Write a function `revert` that, given pair `(num_string,base)` as input, where `base` is a number in the set  $\{2,\dots,32\}$  and `num_string` is  a string of digits in `base`-ary notation,  returns the integer (in decimal) of which `num_string` is the `base`-ary representation. (So, for example, `revert('4b',12)` should return `59`.) 
# 
# **Note.** You can assume that `base` is a number in the set $\{2,\dots,32\}$ and that `num_string` is the string representation of a `base`-ary number.  I.e. there is no need to deal with exceptions.

# In[5]:


#The general method for my revert function is to split the string into a list of strings. Then find the index of 
#string element in our character list. The string will be replaced with this index. I.e. 'a' will be replaced
#by 10. Then it will convert these numbers into the original denary number by using exponents and summation.

import string
#Here we are setting up our two lists which holds all the digits possible, 0-9 and a-v, as in the above function
#This is so when we are provided with a number string that contains a letter, we can convert it to it's 2 digit number 
#i.e. 10 = a, before performing the maths behind the conversion back to denary. Then we join the two lists to make our 
#final list of characters.
raw_numbers = list(range(0,32))
#This is a list of all lowercase letters in the alphabet
alphabet1 = list(string.ascii_lowercase)
#Here we only need the first 22 letters so we slice our list
alphabet2 = alphabet1[0:22]

first_10 = list(range(0,10))
string_first_10 = [str(x) for x in first_10]
digitslist = string_first_10 + alphabet2

def revert(num_string,base): 
    #assume str input is non neg
    if base not in range(33):
        print("Error, the number or base is in incorrect format.")
        return None
    else:
        #Here we deal with the special case if num_string is 0 using an elif statement
        if num_string == '0':
            return 0
        else:
            #This next line of code converts the string number inputted into a list of strings
            number_as_list = list(num_string)
            #The variable length stores the number of digits of the original number
            length = len(number_as_list)
            #This for loops takes each string digit and finds it in our digitslist which contains 0-9 and a-v
            #The index of the position it was found at is stored in index.
            #We then assign the string digit a new value, equivalent to a number 0-31 using the index variable
            for i in range(length):
                index = digitslist.index(number_as_list[i])
                number_as_list[i] = index
            #Now we have all the characters as integers, we reverse the list, so we can easily sum the exponents 
            #of the digits in ascending power order.
            number_as_list.reverse()
            
            total = 0
            #Using the summation technique described in the question, we add together all the elements
            #multiplied by the base to the power of their index position.
            #This sum gives us our final answer which we output.
            for i in range(length):
                total += number_as_list[i]*(base**i)
            return total


# In[6]:


revert('133',5)


# In[7]:


# Casual testing area. You can run this cell multiple times.
# This cell is NOT graded. 
from random import randint
base = randint(2,32)
max_num = 5000
print("Input  Base  Converted  Reverted")
print("=====  ====  =========  ========")
for i in range(10): 
    n = randint(0,max_num)
    d = convert(n,base)     # Define base in the previous cell
    k = revert(d,base)
    print("{:<7}{:<6}{:<11}{:<8}".format(n,base,d,k))


# In[8]:


# In each of the tests (including the hidden tests) revert(num_string,base) is such that 
# num_string is a valid base-ary string representation. Also base is an integer in [2,32].
assert revert('0',5) == 0
assert revert('1101010110',2) == 854
assert revert('1213',5) == 183
assert revert('323',11) == 388
assert revert('3d8',32) == 3496
assert revert('7gp',28) == 5961
assert revert('e2d89c5',17) == 341893567


# ## Question 2: Finding the minimum of a function
# 
# For an arbitrary continuous one-dimensional function $f(x)$, given a bracket of points $[a,b]$, find the minimum of $f(x)$ in this bracket. Your function should take the form `minimise(func,a,b,tol)`, where `func` is the arbitrary function, `a` and `b` are the beginning and end point of the bracket, and `tol` is the desired precision of the result. Optionally, add an argument `verbose` with default value `True` that suppresses output when it is `False`.
# 
# Your function should return the position of the minimum and the value of the function, i.e. $x_{min}$ and $f(x_{min})$. Please see below for an example of a function that takes a function as an argument.

# In[9]:


# Example of a function taking a function as an argument
import numpy as np
import matplotlib.pyplot as plt

# Here one function
def myFunc1(x):
    return x**2-1
# Here is another
def myFunc2(x):
    return 3*x**0.5

# Here is a plotting function that takes
# an arbitrary function func as argument and plots it between a and b
def plotFunc(func,a,b):
    x = np.linspace(a,b,100)
    y = func(x)
    plt.figure()
    plt.plot(x,y,'r')

# Calling the plotting function for both of my functions
plotFunc(myFunc1,-2,2)
plotFunc(myFunc2,0,1)


# ### Part a: Detailed instructions:
# - Starting from the bracket `[a,b]`, compute the midpoint between `a` and `b`, `m`.
# - Define a new point `x` which is midway between `m` and `b`. If `f(x)>f(m)`, we can replace the bracket endpoint `b` by `x`. Else, we can replace the bracket start point `a` by `m`. 
# - Recompute the new midpoint `m` from the new bracket `[a,b]`
# - Repeat until the bracket width is below the tolerance
# 
# Things to verify:
# - Make sure your code works.
# - Does your code work even if there is no minimum between a and b? What is the correct output then?
# - Since $f'(x)=0$ at the minimum so that around it, for $x=x_{min}+\epsilon$, we have $f(x) \approx f(x_{min}) + 0.5 \epsilon^2 f''(x_{min})$. Then your best tolerance is approximately $\sqrt{10^{-16}}=10^{-8}$. Please confirm this numerically.

# In[10]:


#We will need this library later on so we import it.
import math

def minimise(func,a,b,tol):
    #We want our algorithm to run until the interval a to b is smaller than the tolerance so we use a while function.
    #Once the difference between b and a is below the tolerance, our algorithm will output our x value and f(x) where
    #the minimum lies
    #For comparison later, we shall store the inital interval [a,b] in the below variables
    a0 = a
    b0 = b
    while b-a >= tol:
        #We find the midpoint of a and b, m, and then find the value of our function at m
        m=(a+b)/2
        fm = func(m)
        #Similarly, we find the midpoint of m and b, x, then the value of our function at this midpoint x
        x=(m+b)/2
        fx = func(x)
        #Here we use an if else statement to compare the values of the function at the 2 different midpoints.
        #If fx is greater than fm, then we replace the bracket end point with x. So we assign b the value of x.
        #Otherwise we assign a, the start point bracket, the value of m.
        if fx>fm:
            b = x
        else:
            a = m
    #We are now going to compare the minimum value that the algorithm has found. If there is no local minimum in the 
    #interval a to b, then the algorithm will converge to one of the end points a or b. Therefore we use the math module
    #function isclose() to compare the algorithm output and the original endpoints (which we will store in a0 and b0). 
    #The function isclose will return True if the output of the algorithm is deemed very close (within the tolerance 1e-05) 
    #to the endpoint a (or b in the case for equal_b). 
    #Therefore we can use a Boolean comparison and and if-else statement to see if there is no local minima in the interval.
    #If both equal_a and equal_b are False, then the algorithm found a local minima and we output this value.
    equal_a = math.isclose((a+b)/2, a0, rel_tol=1e-9)
    equal_b = math.isclose((a+b)/2, b0, rel_tol=1e-9)
    if equal_a == True or equal_b == True:
        print("There is no local minima in the given interval.")
        return None
    else:
        return (a+b)/2, func((a+b)/2)
    

    
    
#In order to find the best tolerance, I tried different powers of 10 until the algorithm no 
#longer worked, as the precision was too great, i.e. too many decimal places. We want the smallest possible tolerance 
#in order to be as accurate as possible, but not too small that the algorithm struggles with the computational limit 
#(number of decimal places python can store). Any smaller tolerance and our cell would be stuck computing forever as
#the interval would not be small enough to end the algorithm.


# In[11]:


# A simple function with a minimum at 1
def testfunc(x):
    return (x-1)**2

# Checking that your function finds it. Allowing for verbose option or its absence
try:
    x,fmin = minimise(testfunc,-2,2,1e-8,False)
except:
    x,fmin = minimise(testfunc,-2,2,1e-8)
print("Solutions: ", x,fmin)
assert(abs(x-1)<1e-8)
assert(abs(fmin)<1e-14)


# ### Part b: Find all the minima and plot
# Using your minimising function from part a, find all the minima of the function $f(x) = - (x-0.5)^2 e^{-x^2}$, and create a plot that shows:
# - The function
# - A colour marker for each minimum 
# - A text label at each minimum with the values of $(x_{min},f(x_{min}))$; please use `plt.text`.
# - appropriate axes labels and a title

# In[12]:


import numpy as np
import matplotlib.pyplot as plt

def func1(x):
    ans = (-(x-0.5)**2)*(np.exp(-(x**2)))
    return ans

#Here we are finding the first minimum of the function
min1 = minimise(func1,-0.6,-1,10**(-8))

#Here we are finding the other minimum of the function
min2 = minimise(func1,1,1.5,10**(-8))

#We are creating two lists, one to hold the x value of the minimum points and the other to hold the f(x)
#values of the minimum points, so we can plot them against one another on our graph as markers.
xmins = [min1[0],min2[0]]
fxmins = [min1[1],min2[1]]
#Here we create a figure and the parameters (6,6) specify the size.
plt.figure(figsize=(6,6))

#Using arange to construct x values for the x axis
#Where the syntax means : arange(xmin,xmax,step)
xvalues = np.arange(-1, 2, 0.01)

#We plot the x against f(x) and also plot the markers for the minimum points. We also add axis labels and a title for
#our plot.
plt.plot(xvalues, func1(xvalues),'k',xmins,fxmins,'x')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) and its minimum points')

#Here we are adding text labels to our plot, next to the markers for the minimum points.
plt.text(-0.65, -0.9, min1, fontsize=8)
plt.text(0.8, -0.15, min2, fontsize=8)


# ### Part c: Two dimensional contour plot
# Create a two dimensional `plt.contourf` filled contour plot of the function
# \begin{equation} f(x,y) = e^{-(x^2+y^2)}(x-0.5)(y-0.5). \end{equation}
# Make sure that all minima and maxima of the function are clearly visible with sufficient resolution. Please choose a colour map and levels that reflect that $\lim_{x,y\rightarrow \infty} f(x,y) = 0$. If that means cutting off other parts of the values of $f$, that is not a problem.
# 
# The contourf and colormap references are as follows: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html
# https://matplotlib.org/stable/tutorials/colors/colormaps.html

# In[13]:


#In order to pick colours for our plot, we need to import the colourmap library
import matplotlib.cm as cm

#Here we are generating x and y values between -3 and 3 with spacing 0.01
x = y = np.arange(-3.0, 3.0, 0.01)

#Here we are creating an x-y grid for us to compute our z values on
X, Y = np.meshgrid(x, y)

#This is our f(x,y) function
z = np.exp(-((X**2)+(Y**2))*(X-0.5)*(Y-0.5))

#For our contours, we need different levels to colour different colours. Here we are saying we want
#30 evenly spaced out levels between 0 and 10
levels = np.linspace(0,10,30)

#Here we give our figure a specific size and a title
plt.figure(figsize=(8,8))
plt.title('Contourf of minima for f(x,y)')

#We use the countourf function to plot a contour map with the levels we specified earlier
#Our x and y values are along the axes and the z value is represented by a colour. We add a colourbar to
#highlight the different z values, showing where the minima of f(x,y) are.
cf = plt.contourf(x,y,z, levels=levels,cmap=cm.RdYlGn)
plt.colorbar()


# ## Question 3: powers of matrices 
# 
# The `numpy.linalg` library contains function `matrix_power` that on input `(M,k)` such that `M` is a square matrix and `k` is an integer computes `M` raised to the power `k`, i.e$.$ `M` multiplied with itself `k` times. Your task in this question is to develop  in two ways your own version of the `matrix_power` function for the case when `k` is a non-negative integer. Firstly you will develop a simple brute force function in which you carry out matrix multiplication `k` times. Secondly you will develop a clever function that computes `M` raised to the power `k` very efficiently (at least for large `k`) provided that `M` is diagonalisable.
# 
# **Reminder 1.** Given $n \times n$ matrix $M$, $M^0 = I$ where $I$ is the $n \times n$ identity matrix. Also of course $M^1 = M$. <br />
# **Reminder 2.** Matrix $M$ is diagonalisable if the set of eigenvectors $\{\mathbf{v}_1,\dots,\mathbf{v}_n\}$ of $M$ is linearly independent. <br />
# **Reminder 3.** An integer $k$ is *non-negative* if $k \ge 0$.   
# 
# **(a)** Write a function `brute_matrix_power` which takes as inputs a square `numpy` matrix `M` and a non-negative integer `k` and computes and returns `M` raised to the power `k`.  Note that your function should deal with the case  `k` = `0` appropriately. For matrix multiplication you should use the `dot` function from the `numpy` library. (Note that, since we have imported `numpy` under the alias `np`, to compute the matrix multiplication `AB` we can use `np.dot(A,B)`.)  
# 
# **Note.** You can assume that input `M` is a square matrix and input `k` is a non-negative integer. I.e. no need to deal with exceptions. 

# In[14]:


# Run this cell first to make sure you have the appropriate libraries
import numpy as np
import numpy.linalg as lag


# In[15]:


def brute_matrix_power(M,k): 
    #Here we are using numpy to establish the dimensions of the matrix M, we store the number of rows in n_rows and number
    #of columns in n_cols. Then we create a new matrix called identity and let it be the n_rows x n_rows identity matrix
    #NB: n_rows = n_cols as M is a square matrix
    n_rows,n_cols = np.shape(M)
    identity=np.eye(n_rows)
    #Here we deal with the special case where k = 0. If so, M^0 is the identity matrix, so we set our answer to be the 
    #identity matrix created above
    if k == 0:
        answermatrix = identity
    else:
        answermatrix = identity
        #If k does not equal 0, we use a for loop to matrix multiply M by itself k times
        #We let the answer matrix be the identity matrix initially so that for k=1, the matrix
        #multiplication of the identity and M is M.
        for i in range(k):
            answermatrix = np.dot(answermatrix,M)
    #Here we output our answer in matrix format
    return answermatrix


# In[16]:


# Some initial tests. You should do your own testing.
M1 = np.array([[1.0,2.0],[3.0,4.0]])
M1_exp = brute_matrix_power(M1,7)
M2 = np.array([[0.92740639, 0.25351063, 0.24551659],
               [0.98468568, 0.17874981, 0.95155668],
               [0.17727058, 0.79925759, 0.90888965]])
M2_exp = brute_matrix_power(M2,19)
assert np.allclose(M1_exp, lag.matrix_power(M1,7))
assert np.allclose(M2_exp, lag.matrix_power(M2,19))
# In the hidden tests we check the cases k = 0, k = 1, and k > 1
# for n x n matrices with n > 1. 


# **(b)(i)** A matrix $D = (d_{ij})$ is *diagonal* if it is of the form
# 
#  $$                                                                                                               
#     \begin{pmatrix}                                                                           
#       d_{11} & 0 & \cdots  & 0 \\                                                             
#       0  & d_{22} & \cdots & 0  \\                                                             
#       \vdots & \vdots & \ddots  & \vdots \\                                                   
#       0 & 0 & \cdots  & d_{nn}                                                                 
#     \end{pmatrix} \,,                                                                                              
#     $$    
# i.e. if the only non-zero components of $D$ lie on the diagonal. Now, it is  is easy to see that, for any non-negative integer $k$, 
# 
#  $$
#  D^k \;=\;
#     \begin{pmatrix}                                                                           
#       d_{11}^k & 0 & \cdots  & 0 \\                                                                 0  & d_{22}^k & \cdots & 0  \\                                                           
#       \vdots & \vdots & \ddots  & \vdots \\                                                   
#       0 & 0 & \cdots  & d_{nn}^k                                                               
#     \end{pmatrix} \, ,                                                                                             
# $$
# 
# i.e. that $D^k$ is the diagonal matrix whose diagonal components correspond to the diagonal components of $D$ raised to the power $k$. Using this idea write a function `diag_exp` that takes as inputs a diagonal `numpy` matrix `D` and a non-negative integer `k` and computes and returns the matrix `D` raised to the power `k`. Your function `diag_exp` should not carry out any matrix multiplication. It should also not carry out any matrix exponentiation. (I.e. `D**k` is not permitted.) 
# 
# **Note.** You can assume that input `D` is a diagonal matrix and input `k` is a non-negative integer. I.e. no need to deal with exceptions. 

# In[19]:


def diag_exp(D,k):
    #Here we are establishing the dimensions of the matrix D. As it is an nxn matrix, we are only interested in
    #the number of rows
    n , _ = np.shape(D)
    
    #This for loop repeats n times, as the matrix D is an nxn matrix. It changes the ith row and ith column entry to 
    #whatever it's previous value was to the power of k. We store these new values in a new matrix Dk
    #As we are referring to the ith row and ith column, we will only be changing the values along the diagonal of the matrix
    #All other entries not on the diagonal will stay as they were.
    Dk = np.zeros((n,n))
    for i in range(n):
        Dk[i][i]=(D[i][i])**k
    #Here we output our new matrix 
    
    return Dk


# In[20]:


#Here I am testing the diagonal matrix D2 from below. My outputs are D2, then D2^6 using the built in lag.matrix_power 
#function, and then my diag_exp function for finding D2^6. As you can see, both my function and the built in function
#return the same matrix, however in the next cell, my code creates an assertion error. This seems strange as my algorithm 
#outputs the correct answer.

D2 = np.diag([4.08801819, 7.62779907, 7.43214746, 3.62364886])
print(D2)
print(lag.matrix_power(D2,6))
print(diag_exp(D2,6))


# In[22]:


# Some initial tests. You should do your own testing.
D1 = np.diag([1.0,2.0,3.0])
D1_zero = diag_exp(D1,0)
D1_exp = diag_exp(D1,12)
D2 = np.diag([4.08801819, 7.62779907, 7.43214746, 3.62364886])
D2_exp = diag_exp(D2,6)
assert np.allclose(D1_zero,np.eye(3)) 
assert np.allclose(D1_exp,lag.matrix_power(D1,12))
assert np.allclose(D2_exp,lag.matrix_power(D2,6))
# In the hidden tests we check for cases k >= 0
# for n x n diagonal matrices with n > 1. 


# **(b)(ii)** Suppose that $M$ is an  $n \times n $ matrix which  is diagonalisable with eigenvalues $\lambda_1,\dots,\lambda_n$ and eigenvectors $\mathbf{v}_1,\dots,\mathbf{v}_n$ such that $M \mathbf{v}_j = \lambda_j \mathbf{v}_j$ for $j = 1,\dots,n$. 
# Then we know (from first year linear algebra) that $M = NDN^{-1}$ where $D = (d_{ij})$ is the diagonal matrix whose diagonal components are the eigenvalues of $M$  and $N$ is a  $n \times n$  matrix whose columns are the eigenvectors of $M$. Accordingly for each $j = 1,\dots, n$, we have that $d_{jj} = \lambda_j$  and the  $j$th column of $N$ is the eigenvector  $\mathbf{v}_j$ associated with eigenvalue $\lambda_j$. Notice also that 
# 
# $$
# M^2 = NDN^{-1}NDN^{-1} = ND^2N^{-1}
# $$ 
# 
# so that we can easily see by induction that $M^k = ND^kN^{-1}$ for *any* integer $k \ge 0$. With this in mind your task here is to write a function `d_matrix_exp` that takes as input a diagonalisable `numpy` matrix `M` and a non-negative integer `k` and  computes and returns `M` to the power `k`. The idea of course is to exploit the fact that we have just discussed. Therefore you must follow instructions 1-4 carefully. 
# 
# 1. Your function `d_matrix_exp` should carry out only two matrix multiplication operations. 
# 2. In the definition of your function you must use the function `diag_exp` that you defined above. 
# 3. You may use the functions `diag` and `dot` from the `numpy` library and the functions `eig` (or alternatively `eigvals`) and  `inv` from the `numpy.linalg` library. 
# 4. You may **not** use any other (library) functions in the definition of your function `d_matrix_exp`.  
# 
# **Note.** You can assume that input `M` is a diagonalisable matrix and input `k` is a non-negative integer. I.e. no need to deal with exceptions.
# 

# In[ ]:


def d_matrix_exp(M,k):
    nrows , _ = np.shape(M)
    #Here we deal with the special case for when k=0, we tell the algorithm to return the nxn identity matrix
    if k == 0:
        return np.eye(nrows)
    else:
        #Here we are using the linear algebra module and its function eig() to extract the eigenvalues and eigenvectors of our 
        #diagonisable matrix M. We assign them to the variables below
        eig_values, eig_vectors = lag.eig(M)
        #We create our diagonal matrix D by having diagonal entries of each of the eigenvalues, and the rest of the entries as 0.
        D = np.diag(eig_values)
        #Next we create our matrix N using the eigenvectors from above. Each column of the array eig_vectors is an eigenvector,
        #so we can set N equal to this array. We then find the inverse of N using a built in python function lag.inv
        N = eig_vectors
        Ninv = lag.inv(N)
        #Using our function from part bi, we find D^k
        Dk = diag_exp(D,k)
        #Then we perform 2 matrix multiplications using the built in np.dot function to get M^k. This is our output.
        X = np.dot(N,Dk)
        Mk = np.dot(X,Ninv)
        return Mk


# In[ ]:


# Initial tests. You should do your own testing.
E1 = np.array([[0.28106291, 0.51904762, 0.64148366, 0.34731372],
               [0.26979199, 0.26259106, 0.06760985, 0.37285164],
               [0.52170856, 0.06242276, 0.48040518, 0.05721432],
               [0.46227145, 0.61229525, 0.36713064, 0.72517594]])
E1_zero = d_matrix_exp(E1,0)
E1_exp = d_matrix_exp(E1,10)
# In the hidden tests we check for cases k >= 0
# for n x n diagonal matrices with n > 1.


# In[ ]:





# In[ ]:




