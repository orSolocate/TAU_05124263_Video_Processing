#Part1

#Q1
# To run a portion of the code we can copy it and run it on the "Python Console". (obviously we need to make sure this portion of
#code includes all the imports we use in this portion).

#Q2
#Create a breakpoint by cicking next the the line-of-code. and run our code with Run-> Debug..
# A debug window opens at the bottom and you can see the variables values (and continue running the program step-by-step)
# While the debug window is selected, by pressing 'Python Console' we can write and execute commands with existing variables

#Q3

import numpy as np
import random

a=[]
for i in range(30):
    a.append(random.randint(0,9))
matrix=np.array(a)
matrix=matrix.reshape((5,6))
print("original matrix:\n {0}".format(matrix))
three_smallest_values_fifth_row=np.argsort(matrix[-1])[:3]
matrix[-1,three_smallest_values_fifth_row[2]]=10
print("modified matrix:\n {0}".format(matrix))