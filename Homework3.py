#!/usr/bin/env python
# coding: utf-8

# # Homework 3

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

# ## Question 1: applications of Gaussian elimination. 
# 
# Suppose that a system of linear equations can be represented in the form $A \mathbf{x} = \mathbf{b}$ where 
# $A = (a_{ij})$ is a $n \times n$ matrix and $\mathbf{x} = (x_1,\dots,x_n)^T$ and $\mathbf{b} = (b_1,\dots,b_n)^T$. 
# Then we know from first year linear algebra that, if the system has a (unique) solution (which is equivalent to saying that $A$ is non-singular), then we find this solution by reducing the augmented matrix 
# $$                                                                                                                 
#     \begin{pmatrix}                                                                                               
#       a_{11} & \cdots  & a_{1n} & b_1\\                                                                           
#       \vdots  & \ddots & \vdots & \vdots \\                                                                       
#       a_{n1}  & \cdots  & a_{nn} & b_n                                                                             
#     \end{pmatrix}
# $$                       
# to a matrix of the form 
# $$
#    \begin{pmatrix}                                                                                               
#       1 & \cdots  & 0 & c_1\\                                                                           
#       \vdots  & \ddots & \vdots & \vdots \\                                                                       
#       0  & \cdots  & 1 & c_n                                                                             
#     \end{pmatrix}\, ,                                                                                             
# $$
# i.e. where the first $n$ columns are the $n \times n$ identity matrix, using elementary row operations. We then note that, relative to the original system  this represents 
# 
# \begin{align*}                                                                                                     
#   x_1 + 0 + \cdots + 0 \;&=\; c_1 \\                                                                               
#   0  + x_2 + \cdots + 0 \;&=\; c_2 \\                                                                             
#                           &\hspace{2mm}\vdots \\ 
#   0 + 0 + \cdots + x_n  \;&=\; c_n                                                                                 
# \end{align*}  
# 
# so that the solution of the system of equations is given by the vector $\mathbf{c} = (c_1,\dots,c_n)^T$. In a similar vein, if $n \times n$ matrix $B = (b_{ij})$ is invertible (or equivalently if $B$ is non-singular) we can  find $B^{-1}$ as follows. We first form an $n \times 2n$ augmented matrix by juxtaposing $B$ with the $n \times n$ identity matrix $I$ - for which we use the notation  $[B \,|\, I]$- and perform elementary row operations until we obtain $[I\,|\,B^{-1}]$ and hence $B^{-1}$ itself.    
# 
# The purpose of the present question is to define both a function `my_solve` to  solve systems of linear equations, of the type described above, as also a function `my_inverse` to compute the inverse of square matrices. We shall do this in modular fashion by designing a function `reduce_matrix` that, given an $n \times m$ matrix $M$ such that $m \ge n$, performs elementary row operations on $M$ in such a way that (if successful) the first $n$ columns of $M$ end up forming the $n \times n$ identity matrix. We shall do this via parts (a), (b) and (c) below. We shall then use use `reduce_matrix` to develop the functions `my_solve` and `my_inverse` in parts (d) and (e) respectively. 
# 
# **Notation.** To avoid confusing notation, for $n \times m$ matrix `M` we use python style notation to refer to its rows and columns and components. Accordingly `M[i,:]` denotes the row with index `i`  and `M[:,j]` denotes the column with index `j`. Thus for example `M[0,:]` and `M[n-1,:]` are respectively the first and last rows of `M`. Likewise `M[i,j]` denotes the component of `M` belonging to to row `M[i,:]` and column `M[:,j]`. So, as another example, `M[0,0]` is the first component  of the first row of `M` and `M[n-1,0]` is the first component of the last row of `M`.  For simplicity we will use the terminology *row* `i` and *column* `j` to refer to the row with index `i` and the column with index `j` (i.e. `M[i,:]` and `M[:,j]` respectively). 

# In[1]:


# Make sure that you have the appropriate library functions available before proceeding
import numpy as np
import numpy.linalg as lag


# **(a)** Write a function `pivot` that takes as input a $n \times m$ `numpy` matrix `M`, and an iteger `i` and accomplishes the following. 
# 1. If all the components in column  `i` below  and including `M[i,i]` are zero, print the message `"Error: pivot is zero"`  and return `i`. 
# 2. Otherwise ensure that `M[i,i]` has greatest absolute value relative to the components below it in  column `i`. To do this  perform a swap of row `i` with an appropriate row below row `i`,  if necessary. Then divide row `i` by `M[i,i]` - so that after this operation we have `M[i,i]` = `1.0` - and return `-1` (which signals that the computation has been successful).      
# 
# **Example.** If `M` is the matrix
# $
# \begin{pmatrix}                                                                                                   
#   -1 & 2 & -3 & 0 \\                                                                                               
#   -3 & 4 & -1 & 2 \\                                                                                               
#   5 & 2 & 0 & -2 \\                                                                                               
#   3 & 7 & 2 & 5                                                                                                   
# \end{pmatrix}
# $
# then `pivot(M,1)` will modify `M` so that it becomes 
# $
# \begin{pmatrix}                                                                                                   
#   -1 & 2 & -3 & 0 \\                                                                                               
#   3/7 & 1 & 2/7 & 5/7 \\                                                                                           
#   5 & 2 & 0 & -2 \\                                                                                               
#   -3 & 4 & -1 & 2                                                                                                 
# \end{pmatrix}    
# $
# and will return `-1`. On the other hand if `N` is the matrix 
# $
# \begin{pmatrix}                                                                                                   
#   -1 & 2 & -3 & 0 \\                                                                                               
#   -3 & 0 & 4  & 1 \\                                                                                               
#   2 & 0 & -1 & 5 \\                                                                                               
#   1 & 0 & -2 & 3                                                                                                   
# \end{pmatrix}  
# $
# then `pivot(N,1)` will not modify `N` but will print `"Error: pivot is zero"` and return  `1`. 
# 
# **Note.** To test whether a component is zero you should use the function `is_zero` given below. Note also that you can suppose that $n \le m$ and  `0` $\le$ `i` $<$ `n` . I.e. there is no need to test for incompatible input. 

# In[2]:


# You should start by running this cell
def is_zero(x): 
    return abs(x) < 1e-14


# In[3]:


# the function is_zero returns false if not equal to zero and true it is
def pivot(M,i):
    error_message = "Error: pivot is zero"
    n,m = np.shape(M)
    col_entries = np.array(M[i:n,i])
    max_entry = np.amax(abs(col_entries))
    if is_zero(max_entry) == True:
        print(error_message)
        return i
    else:
        holder1 = []
        holder2 = []
        x = np.argmax(abs(col_entries))
        holder1 = np.copy(M[i,:])
        holder2 = np.copy(M[i+x,:])
        M[i,:] = holder2
        M[i+x,:] = holder1
        M[i,:] = M[i,:]/M[i,i]
        return -1
        
        
    


# In[4]:


# Some initial tests. You should do your own testing. 
# B = np.random.randint(-10,30,size=(5,6)) * 1.0 used to generate B
B = np.array([[ 8., 14.,  -7.,  0., -5.,  8.],
              [ 4.,  8.,  25., -2.,  6., 27.],
              [21.,  0.,   3., -2., -7., 16.],
              [13.,  3.,   6., 21.,  4., 23.],
              [20.,  0.,  -8., 12.,  4., -2.]])
C = np.array([[ 8., 14.,  -7.,  0., -5.,  8.],
              [ 4.,  8.,  25., -2.,  6., 27.],
              [21.,  0.,   0., -2., -7., 16.],
              [13.,  3.,   0., 21.,  4., 23.],
              [20.,  0.,   0., 12.,  4., -2.]])
# Now the pivoting
B_val = pivot(B,2)
C_val = pivot(C,2)
# B should have been modified and should be identical to B_result below
B_result = np.array([[ 8.  , 14.  , -7.  ,  0.  , -5.  ,  8.  ],
                     [ 4.  ,  8.  , 25.  , -2.  ,  6.  , 27.  ],
                     [-2.5 ,  0.  ,  1.  , -1.5 , -0.5 ,  0.25],
                     [13.  ,  3.  ,  6.  , 21.  ,  4.  , 23.  ],
                     [21.  ,  0.  ,  3.  , -2.  , -7.  , 16.  ]])
# C should not have been modified. C should be identical to C_result below. 
C_result = np.array([[ 8., 14., -7.,  0., -5.,  8.],
                    [ 4.,  8., 25., -2.,  6., 27.],
                    [21.,  0.,  0., -2., -7., 16.],
                    [13.,  3.,  0., 21.,  4., 23.],
                    [20.,  0.,  0., 12.,  4., -2.]]) 
assert B_val == -1
assert np.allclose(B,B_result)
assert C_val == 2
assert np.allclose(C,C_result)
# Note pivot(C,2) generates message "Error: pivot is zero" below. This is correct behaviour


# **(b)** Write a function `gauss` that takes as input a $n \times m$ numpy matrix `M` and an integer `i` - where it is assumed that `M[i,i]` = `1` - and peforms elementary row operations on the rows of `M` in such a way that, following this operation, all the components of column `i` except  `M[i,i]` are zero. Your function should return `-1`.  
# 
# **Example.** Suppose that `M` is the matrix 
# $
# \begin{pmatrix}                                                                                                   
#   -1 & 2 & -3 & 0 \\                                                                                               
#   3/7 & 1 & 2/7 & 5/7 \\                                                                                           
#   5 & 2 & 0 & -2 \\                                                                                               
#   -3 & 4 & -1 & 2                                                                                                 
# \end{pmatrix}
# $. 
# Then `gauss(M,1)` will modify `M` so that it becomes 
# 
# $$
# \begin{pmatrix}                                                                                                   
#   -1 - 6/7 & 2-2 & -3-4/7 & 0-10/7 \\
#   3/7 & 1 & 2/7 & 5/7 \\                                                                                           
#   5- 6/7 & 2-2 & 0-4/7 & -2-10/7 \\
#   -3-12/7 & 4-4 & -1-8/7 & 2-20/7                                                                                 
# \end{pmatrix}
# \;=\;
# \begin{pmatrix}                                                                                                   
#   -13/7  & 0 & -25/7 & -10/7 \\
#   3/7 & 1 & 2/7 & 5/7 \\                                                                                           
#   29/7 & 0 & -4/7 & -24/7 \\
#   -33/7 & 0 & -15/7 & -6/7                                                                                 
# \end{pmatrix} \,.
# $$
# 
# **Note.** We will only implement `gauss(M,i)` on a matrix that has already been successfully processed by our function `pivot(M,i)`. This is why we can assume that `M[i,i]` = `1`. Accordingly there is no need to test for incompatible input. 

# In[5]:


def gauss(M,i):
    n,m = np.shape(M) # Here we establish the dim of the matrix
    for x in range(n): # For each entry in the ith col
        if x == i: # assuming that M[i,i] =1, we don't alter this row
            pass
        else: # for all other rows
            if is_zero(M[x,i]) == False: # check if the xth entry is not 0
                coefficient = M[x,i]/M[i,i] # find out what you need to multiply the ith entry with to get the xth entry
                rowcopy = np.copy(M[i,:]) # store a copy of the ith row here
                M[x,:] = M[x,:] - coefficient*rowcopy # performing an ERO to make the xth, ith entry 0
            else: # if xth entry is 0 then do nothing to the row
                pass
    return -1
        
        


# In[6]:


# An initial test. You should do your own testing.
B1 = np.array([[ 8.  , 14.  , -7.  ,  0.  , -5.  ,  8.  ],
               [ 4.  ,  8.  , 25.  , -2.  ,  6.  , 27.  ],
               [-2.5 , -0.  ,  1.  , -1.5 , -0.5 ,  0.25],
               [13.  ,  3.  ,  6.  , 21.  ,  4.  , 23.  ],
               [21.  ,  0.  ,  3.  , -2.  , -7.  , 16.  ]])
B1_val = gauss(B1,2)
# B1 has now been modified (via elementary row operations)
# B1 should now be the same as B1_result below
B1_result = np.array([[ -9.5 ,  14.  ,   0.  , -10.5 ,  -8.5 ,   9.75],
                      [ 66.5 ,   8.  ,   0.  ,  35.5 ,  18.5 ,  20.75],
                      [ -2.5 ,  -0.  ,   1.  ,  -1.5 ,  -0.5 ,   0.25],
                      [ 28.  ,   3.  ,   0.  ,  30.  ,   7.  ,  21.5 ],
                      [ 28.5 ,   0.  ,   0.  ,   2.5 ,  -5.5 ,  15.25]])
assert B1_val == -1
assert np.allclose(B1,B1_result)


# **(c)** Now notice that, given a $n \times m$ `numpy` matrix `M`, with $m \ge n$,  if  we start by computing `pivot(M,0)` and this returns `-1` we can proceed by  computing `gauss(M,0)` and so reduce the first column of `M` to an array with first component `1` and all other components `0`. Then continuing,  if we next compute `pivot(M,1)` and this returns `-1` then we can proceed by  computing `gauss(M,1)` and so reduce the second column of `M` to an array with second component `1` and all other components `0`. Continuing in this way - provided `pivot(M,i)` does indeed return `-1` at every step `i` = `0`,$\dots$,`n-1` - after $n$ steps we will have reduced `M` so that its first $n$ columns are the $n \times n$ identity matrix `I`. 
# 
# Using the above  idea write a function `matrix_reduce` that takes as input a $n \times m$ numpy matrix `M`, and tries to row reduce `M` to a matrix whose first $n$ columns are the $n \times n$ identity matrix. Your function should use the functions `pivot` and `gauss`. If at some stage `i` the computation `pivot(M,i)` returns `i`  your function should immediately return `i` (so effectively aborting the computation). If, on the other hand this does not happen at any stage `i` = `0`,$\dots$,`n-1` (meaning that the reduction of `M` was successful) your function should return `-1`. 
# 
# **Note.** You can assume that $m \ge n$. I.e. there is no need to test for incompatible input. 

# In[7]:


def reduce_matrix(M): 
    nrows,_ = np.shape(M) # Here I am establishing the number of rows in the matrix M
    for i in range(nrows):
        if pivot(M,i) == -1:
            gauss(M,i)
        else:
            return i
    return -1
    


# In[8]:


# Initial tests. You should do your own testing.
B2 = np.array([[ 8., 14.,  -7.,  0., -5.,  8.,  3.],
               [ 4.,  8.,  25., -2.,  6., 27., -1.],
               [21.,  0.,   3., -2., -7., 16., 11.],
               [13.,  3.,   6., 21.,  4., 23., -5.],
               [20.,  0.,  -8., 12.,  4., -2., 12.]])
B2_val = reduce_matrix(B2)
# B2 has now been reduced and should be the same as B2_result below
B2_result = np.array([[ 1.,  0.,  0.,  0.,  0., 0.20081312,  0.7470541 ],
                      [ 0.,  1.,  0.,  0.,  0., 0.59194741, -0.12856537],
                      [ 0.,  0.,  1.,  0.,  0., 1.25116973, -0.35013236],
                      [ 0.,  0.,  0.,  1.,  0., 0.79038592, -0.72034135],
                      [ 0.,  0.,  0.,  0.,  1., -1.37288388, 0.72548882]])
assert B2_val == -1
assert np.allclose(B2,B2_result)
# Column C2[:,3]  is a scalar multiple of column C2[:,1] in C2 
C2 = np.array([[ 8., 14.,  -7.,  7., -5.,  8.,  3.],
               [ 4.,  8.,  25.,  4.,  6., 27., -1.],
               [21.,  0.,   3.,  0., -7., 16., 11.],
               [13.,  3.,   6., 1.5,  4., 23., -5.],
               [20.,  0.,  -8.,  0.,  4., -2., 12.]])
C2_val = reduce_matrix(C2)
# This time the computation is aborted as pivot(C2,3) fails
assert C2_val == 3 
# Note that reduce_matrix(C2) should print the message "Error: pivot is zero" below. 


# **(d)** Suppose now that we are given a system of $n$ linear equations in $n$ unknowns so that we are able to represent the system in the form  $A\mathbf{x} = \mathbf{b}$ where $A$ is a $n \times n$ matrix. Then, provided the system has a (unique) solution, we can use the function `matrix_reduce` to find the solution. Your task here is  to write just such a function `my_solve` which takes as input a $n \times n$ numpy matrix `A` and a column vector `b` in the form of a $n \times 1$ numpy array. Your function should form the augmented $n \times (n+1)$ matrix `M` with `A` comprising the first $n$ columns
# of `M` and `b` making up the last column. Your function should then pass `M` as input to the function `reduce_matrix`. If `reduce_matrix` succeeds in reducing `M` (i.e. in the manner  described in part (c)) then your function should return the last column of `M` as a $3 \times 1$ numpy array. If `reduce_matrix` does not succeed your function should simply pass on/return integer `i` returned by `reduce_matrix`.  
# 
# **Note.** The matrix `A` is singular if and only if the linear system has either no solution or infinitely many solutions. In this case, the function `pivot`, working as a subfunction of `reduce_matrix`, will encounter a zero at some stage `i` in the computation - i.e. when `pivot(M,i)` is being processed -  and so the error message `i` will be passed/returned from `pivot` to `reduce_matrix` to `my_solve` signalling that the computation has been aborted. If on the other hand `A` is non-singular then `my_solve` will return the solution of the system in the form of a $3 \times 1$ `numpy` array.  

# In[9]:


def my_solve(A,b):              # We are assuming that A is n x n (say) and b is n x 1. 
    M = np.concatenate((A,b),axis = 1) # This joins the two matrices, the axis ensures they join side by side
    rows,cols = np.shape(M) # establishing the dimensions nx(n+1)
    reduce_matrix(M) # performs the reduction
    if reduce_matrix(M) == -1: # the case when the reduction was successful
        ans = np.reshape(M[:,cols-1],(3,1)) # converts the ans into a 3x1 instead of 1x3
        return ans # returns the last column
    else:
        i = reduce_matrix(M)
        return i


# In[10]:


# Initial tests. You should carry out testing yourself. 
A3 = np.array([[1.0, 2, -3], [2, -1, 4], [1, -1, 1]])
b3 = np.array([[9.0],[1],[0]])
x3 = my_solve(A3,b3)
# x3 should now be the solution of this system. 
print("x3 = \n", x3, "\n")
assert np.allclose(x3,lag.solve(A3,b3))
# There is not a unique solution for the following system
B3 = np.array([[1.0, 2, -3], [2, -1, 4], [3.0, 6, -9]])
b3 = np.array([[9.0],[1],[0]])
y3 = my_solve(B3,b3)
# In fact pivot([B3|b3],2) failed during the computation
assert y3 == 2
# The message "Error: pivot is zero" should appear below 


# **(e)** Write a function `my_inverse` which takes as input a $n \times n$ matrix `M`, constructs the $n \times 2n$ augmented matrix `N`, whose first $n$ columns comprise `M` and whose last $n$ columns comprise  the $n \times n$ identity matrix `I`, and then passes `N` to `matrix_reduce`. If the reduction performed by `matrix_reduce` aborts/fails your function should return the integer `i` passed to it by `matrix_reduce`.  If the reduction is successful then your function should return the inverse of `M`. (You should read the description in the introduction to this question if you are not sure how to do this.)  
# 
# **Note.** By design, given input `M`, the computation of `my_inverse` will successfully output the inverse of `M` if and only if `M` is non-singular,  for reasons similar to those given in the note in part (d). Note that your function should not modify `M` itself otherwise the tests will fail. (See *Copying and swapping* in Appendix5_1.ipynb.) You can assume that `M` is a square matrix. I.e. there is no need to test for incompatible input.

# In[11]:


def my_inverse(M): 
    n_rows,n_cols = np.shape(M)
    N = np.concatenate((M,np.identity(n_rows)),axis=1)
    i = reduce_matrix(N)
    if i == -1:
        ans = N[:,n_rows:(2*(n_rows))]
        return ans
    else:
        return i
    


# In[12]:


# Some initial tests. You should do your own testing. 
#A4 = np.random.randint(-10,30,size=(4,4)) * 1.0
A4 = np.array([[12.,  5., 24., 29.],
               [ 1., 16.,  9., -2.],
               [ 8.,  9., -5., -4.],
               [ 0., -5.,  4.,  3.]])
A4_inv = my_inverse(A4)
assert np.allclose(np.dot(A4,A4_inv), np.eye(4))
# B4 is not invertible 
B4 = np.array([[12.,  5., 24., -6.],
               [ 1., 16.,  9., -.5],
               [ 8.,  9., -5., -4.],
               [ 0., -5.,  4.,  0.]])
B4_inv = my_inverse(B4)
# In fact pivot([B4|I],3) failed during the computation
assert B4_inv == 3
assert my_inverse(np.zeros((7,7))) == 0 
# The message  below "Error: pivot is zero" should appear twice below. 


# # Question 2: The Lotka-Volterra equations
# In the Canadian arctic, the main food of canadian lynx are snowshoe hares. While hares reproduce quickly (they are a type of rabbit, after all), if lynx predate too many hares, the population of hares crashes. This then leads to lynx having few to no kittens, and so the population of lynx reduces, allowing the hare population to recover. This in turns allows for more lynx kittens to survive and predation to rise again. Thise dynamics leads to a surprising phenomenon: **predator-prey population oscillations** in the lynx and hare numbers.
# 
# The simplest model of these predator-prey oscillations are the Lotka-Volterra equations:
# Let $H$ be the density of hares in a region, and $L$ the corresponding density of lynx, both in dimensionless units. Then the dynamics of $h$ and $l$ are given by
# \begin{align}
# \frac{dH}{dt} &= b H - p H L \\
# \frac{dL}{dt} &= - d L + r L H.
# \end{align}
# Here $b$ is the net birth rate of hares, $b>0$ always (they do breed like rabbits). The term $p H L$ expresses the rate of predation of hares by lynx, which reduces the hare population. The number of lynx decreases with their death rate $d$, but increases due to preying on hares with rate $r$.

# ## Part 1
# Numerically integrate the Lotka-Volterra equations using a 4th order Runge-Kutta integrator. To make the remainder of this question work, please wrap your whole solution into a function which takes as arguments the Lotka-Volterra parameters, the initial conditions, the time step and the time interval, and returns arrays for hare and lynx populations. PLEASE USE THE TEMPLATE PROVIDED.
# 
# Run your function on the sample parameters $b=p=d=r=1$ with initial conditions $H = L = 0.5$ , and plot the results to make sure that it works, i.e. that you see oscillations.

# In[13]:


import matplotlib.pyplot as plt
# Please COPY AND USE the following template for your answer (uncomment every line once)

# Put any extra functions above your main function
# Main function
# # Arguments: the 4 parameters of the model, the initial hare and lynx populations, 
# # the time step and the integration time
# def LotkaVolterra(b,p,d,r,H0,L0,dt,T):
#     # do your computation here. 
#     # Note of caution: We can't have functions inside of functions. 
#     # We can make extra functions for the RHS outside of this function if desired, however.
#     # return the time array and the arrays of hare and lynx populations
#     return t,H,L


# Main function: this computes the RHS of the differential eqns simultaneously
def func(t,b,p,d,r,H,L,dt):
    f1 = dt*(b*H - p*H*L)
    f2 = dt*(r*L*H - d*L)
    return f1,f2

# # Arguments: the 4 parameters of the model, the initial hare and lynx populations, 
# # the time step and the integration time


def LotkaVolterra(b,p,d,r,H0,L0,dt,T):
    N = int(T/dt)
    
    # time array 
    t = np.zeros((N,))
    # hare array
    h = np.zeros((N,))
    # lynx array
    l = np.zeros((N,))
    
    # initial conditions
    h[0] = H0
    l[0] = L0
    
    for n in range(0,N-1):
        # Compute both right hand side functions at t[n] for k1 and then so on
        # For technical reasons (these are sequences), can't multiply every line by dt
        # moved the dt into the function itself
        k1_f1,k1_f2 = func(t[n],b,p,d,r,h[n],l[n],dt)
        k2_f1,k2_f2 = func(t[n]+dt/2,b,p,d,r,h[n]+k1_f1/2,l[n]+k1_f2/2,dt)
        k3_f1,k3_f2 = func(t[n]+dt/2,b,p,d,r,h[n]+k2_f1/2,l[n]+k2_f2/2,dt)
        k4_f1,k4_f2 = func(t[n]+dt,b,p,d,r,h[n]+k3_f1,l[n]+k3_f2,dt)
        # Update
        t[n+1] = t[n] + dt
        h[n+1] = h[n] + k1_f1/6 + k2_f1/3 + k3_f1/3 + k4_f1/6
        l[n+1] = l[n] + k1_f2/6 + k2_f2/3 + k4_f2/3 + k4_f2/6
    return t, h, l


# In[14]:


# Basic test of your function. Make sure your syntax is right!
time,hare,lynx = LotkaVolterra(1,1,1,1,1,1,0.01,20)

assert (np.std(hare)<1e-14)
assert (np.std(lynx)<1e-14)


# In[15]:


# Some more tests here. This includes a hidden one of the same nature, but with different parameters. 
time,hare,lynx = LotkaVolterra(1,1,1,1,0.5,0.5,0.001,20)
assert (abs(np.std(hare)-0.612)<1e-3)
assert (abs(np.std(lynx)-0.612)<1e-3)

# Not a test, just a visualisation to help you see what is happening
plt.figure()
plt.plot(time,hare,'-b',label='hare')
plt.plot(time,lynx,'-r',label='lynx')
plt.xlabel('time')
plt.ylabel('population')
plt.legend()


# ## Part 2
# The Lotka-Volterra equations are integrable, and in particular, there is a conserved quantity associated to them, which can be expressed as
# \begin{equation} C = b \log L - p L -r H + d \log H, \end{equation}
# where $\log$ is the natural logarithm, aka `np.log(x)`.
# 
# Write a function that computes the conserved quantity, if given the 4 parameters and initial conditions for H and L. To do this, you need to call your LotkaVolterra function from inside your new function, with the correct 4 parameters, and a suitable integration range (not too long) and time step (small). Then make it compute solutions, and then use these solutions to compute the $C$ vector. 
# 
# Check that $C$ is approximately constant by creating a plot inside your function (do not worry if there is a slight drift).
# Finally, take the average of $C$. This is your result: make it the return argument of your function.

# In[16]:


import numpy as np
# Your answer here. Please use this TEMPLATE SOLUTION (uncomment every line once)
# def conserved(b,p,d,r,H0,L0):
#    # compute the Lotka-Volterra solution using the LotkaVolterra function
#    # compute the conserved quantity on the solution
#    # make a plot of the conserved quantity
#    # compute the mean value of the conserved quantity and return it
#    return cval

def conserved(b,p,d,r,H0,L0):
    # compute the Lotka-Volterra solution using the LotkaVolterra function
    time,hare,lynx = LotkaVolterra(b,p,d,r,H0,L0,0.1,10)
    # compute the conserved quantity on the solution
    C = np.zeros(len(time))
    for i in range(len(time)):
        C[i] = b*np.log(lynx[i]) - p*lynx[i] - r*hare[i] + d*np.log(hare[i])
    # make a plot of the conserved quantity
    plt.figure()
    plt.plot(time,C,'r')
    plt.xlabel('time')
    plt.ylabel('Conserved quantity')
    plt.show()
    # compute the mean value of the conserved quantity and return it
    cval = sum(C)/len(time)
    return cval


# In[17]:


# Checking a couple of values for C. There are two hidden checks here, of a similar nature.
cval = conserved(1,1,1,1,1.0,1.0)
assert abs(cval+2.0)<1e-14


# ## Part 3
# For a given set of $b,p,h,l$, the amplitude of the predator-prey oscillations strongly depends on the initial conditions for the population. We will again choose $b=p=d=r=1$ for simplicity.
# 
# You task is now to construct a phase space plot of $H$ vs. $L$ (on the $y$ and $x$ axis, respectively), that combines several initial conditions on the same plot. Additionally, compute the value of $C$ for each initial condition. This will allow you to answer a couple of questions.

# In[18]:


# A set of 5 initial conditions
Hini = [0.2,0.4,0.6,0.8,1.0]
Lini = [0.2,0.4,0.6,0.8,1.0]

t0,h0,l0 = LotkaVolterra(1,1,1,1,Hini[0],Lini[0],0.1,10) 
t1,h1,l1 = LotkaVolterra(1,1,1,1,Hini[1],Lini[1],0.1,10) 
t2,h2,l2 = LotkaVolterra(1,1,1,1,Hini[2],Lini[2],0.1,10) 
t3,h3,l3 = LotkaVolterra(1,1,1,1,Hini[3],Lini[3],0.1,10) 
t4,h4,l4 = LotkaVolterra(1,1,1,1,Hini[4],Lini[4],0.1,10)
#t5,h5,l5 = LotkaVolterra(1,1,1,1,0.1,0.1,0.1,10)

plt.figure(figsize=(8,6))
plt.plot(h0,l0,'r',label='H0 = L0 = 0.2')
plt.plot(h1,l1,'g',label='H0 = L0 = 0.4')
plt.plot(h2,l2,'k',label='H0 = L0 = 0.6')
plt.plot(h3,l3,'b',label='H0 = L0 = 0.8')
plt.plot(h4,l4,'y',label='H0 = L0 = 1.0')
plt.plot(h5,l5,'o',label='H0 = L0 = 0.1')
plt.legend()
plt.title('Phase space for varying intial H & L conditions')


# In[19]:


# Please fill in the Booleans True or False (not 'True' or 'False')

# 1. There is a stationary point of the equations at position (1,1).
stationary = True
# 2. The amplitude of oscillations decreases when the initial condition moves away from the stationary point.
amplitude = False
# 3. There is a minimum of C at the stationary point
minimum = False
# 4. The initial condition H0 = L0 = 0.1 would have larger amplitude oscillations 
# and a smaller C than all of the plotted curves.
initial = True

# YOUR CODE HERE
print('Cval of H0=L0=0.2 :'+str(conserved(1,1,1,1,Hini[0],Lini[0])))
print('Cval of H0=L0=0.4 :'+str(conserved(1,1,1,1,Hini[1],Lini[1])))
print('Cval of H0=L0=0.6 :'+str(conserved(1,1,1,1,Hini[2],Lini[2])))
print('Cval of H0=L0=0.8 :'+str(conserved(1,1,1,1,Hini[3],Lini[3])))
print('Cval of H0=L0=1.0 :'+str(conserved(1,1,1,1,Hini[4],Lini[4])))
print('Cval of H0=L0=0.1 :'+str(conserved(1,1,1,1,0.1,0.1)))


# In[20]:


# Checking your answers here (hidden, there is no other option :) ) 


# In[ ]:




