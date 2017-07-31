#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:40:38 2017

@author: msdogan
"""
import numpy as np
import pandas as pd
import csv

#print('Hello python!')

# this is the first comment 
#spam = 1 # and this is the second comment
#print(spam)

## Using python as a calculator
#print(2 + 2)
#print(50 - 5*6)
#print((50 - 5*6) / 4)
#print(8 / 5) # division always returns a floating point number in v3+ not v2.7
#
#print(17 / 3)  # classic division returns a float in v3+
#print(17 // 3)  # floor division discards the fractional part
#print(17 % 3)  # the % operator returns the remainder of the division
#print(5 * 3 + 2)  # result * divisor + remainder
#
#print(5 ** 2)  # 5 squared
#print(2 ** 7)  # 2 to the power of 7
#
#width = 20
#height = 5 * 9
#print(width * height)

## Strings
#a = 'orange'
#b = "peach"
#c = 'mango'
#print(a + b + c) # concatenation
#print(a,b,c)

## Character position
#word = 'Python'
#print(len(word)) # number of characters in a string
#print(word[0])  # character in position 0
#print(word[5])  # character in position 5
#print(word[-1])  # last character
#print(word[-2])  # second-last character
#print(word[-6])
#print(word[0:2])  # characters from position 0 (included) to 2 (excluded)
#print(word[2:5])
#
#print(word[:2])   # character from the beginning to position 2 (excluded)
#print(word[:2] + word[2:])

## Lists
#squares = [1, 4, 9, 16, 25]
#print(squares)
#print(squares[0])  # indexing returns the item
#print(squares[-1])
#print(squares[-3:])  # slicing returns a new list
#
#cubes = [1, 8, 27, 65, 125]  # something's wrong here
#cubes[3] = 64  # replace the wrong value
#print(cubes)
#
#cubes.append(216)  # add the cube of 6
#cubes.append(7 ** 3)  # and the cube of 7
#print(cubes)
#
#letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
## replace some values
#letters[2:5] = ['C', 'D', 'E']
#print(letters)
#print(len(letters))
#
## It is possible to nest lists (create lists containing other lists)
#a = ['a', 'b', 'c']
#n = [1, 2, 3]
#x = [a, n]
#print(x)
#print(x[0])
#print(x[0][1])

## Sets
#basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
#print(basket) # show that duplicates have been removed
#print('orange' in basket) # fast membership testing
#print('crabgrass' in basket)
#
## set operations
#a = set('abracadabra')
#b = set('alacazam')
#print(a) # unique letters in a
#print(a - b) # letters in a but not in b
#print(a | b) # letters in a or b or both
#print(a & b) # letters in both a and b
#print(a ^ b) # letters in a or b but not both

## Dictionaries
#tel = {'jack': 4098, 'sape': 4139}
#tel['guido'] = 4127 # assing a new element
#print(tel)
#print(tel['jack'])
#del tel['sape'] # delete an element
#tel['irv'] = 4127
#print(tel)
#print(tel.keys()) # print dictionary keys
#print('guido' in tel)
#print('jack' not in tel)
#
## building a dictionary directly from sequences of key-value pairs:
#d1 = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
#print(d1)  
## building a dictionary from pairs using keyword arguments
#d2 = dict(sape=4139, guido=4127, jack=4098)
#print(d2)

## Numpy arrays
#a = np.arange(15).reshape(3, 5) # create a 3x5 array with elements from 0-14
#print(a.shape)
#print(a.ndim)
#print(a.dtype.name)
#print(a.size)
#print(type(a))
#
#a1 = np.array([6, 7, 8]) # create a numpy array from a list
#print(a1.dtype)
#b = np.array([1.2, 3.5, 5.1]) # float array
#print(b.dtype)
#b = np.array([(1.5,2,3), (4,5,6)]) # 2D array
#print(b)
#c = np.array( [ [1,2], [3,4] ], dtype=complex )
#print(c)
#
#a2 = np.arange( 10, 30, 5 ) # first item, last item, step size
#print(a2)
#
#x = np.zeros(15).reshape(3, 5) # create an array with zeros
#print(x)
#x = np.zeros((3,5)) # create an array with zeros
#print(x)
#y = np.ones((2,3,4), dtype=np.int16) # create an array with ones
#print(y)
#z = np.empty((2,3)) # empy array , output may vary
#print(z)

## Basic operations with Numpy arrays
#A = np.array( [[1,1],[0,1]] )
#print(A)
#B = np.array( [[2,0],[3,4]] )
#print(B)
#print(A*B) # elementwise product
#print(np.dot(A,B)) # matrix product, same as A.dot(B)
#
## sum, min, max
#np.random.seed(101) # if you don't set the seed, you get different results each time
#a = np.random.random((2,3)) # array with random elements from 0 to 1
#print(a)
#print(a.sum())
#print(a.mean())
#print(a.min(),a.max())
#
#b = np.arange(12).reshape(3,4)
#print(b)
#print(b.sum(axis=0)) # sum of each column
#print(b.min(axis=1)) # min of each row
#print(b.cumsum(axis=1)) # cumulative sum along each row
#print(np.exp(b)) # exponential of an array
#print(np.sqrt(b)) # square root of an array

# Pandas data frames
#df1 = pd.DataFrame({ 'A' : 1.,
#                     'B' : pd.Timestamp('20130102'),
#                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
#                     'D' : np.array([3] * 4,dtype='int32'),
#                     'E' : pd.Categorical(["test","train","test","train"]),
#                     'F' : 'foo' })
#print(df1)
#print(df1.dtypes)

#dates = pd.date_range('20130101', periods=6)
#df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
#print(df)
#print(df.keys()) # data frame keys (columns)
#print(df.index) # row index
#print(df.columns) # column name
#print(df['A']) # display column 'A'
#print(df[0:3]) # display first 3 column
#print(df.loc['20130102':'20130104',['A','B']]) # display defined rows and columns
#print(df.iloc[2]) # display 3rd row

## some basic operations with pandas data frames
#print(df.head(3)) # first 3 rows
#print(df.tail(2)) # last 2 rows
#print(df.describe()) # stats
#print(df.T) # transpose
#print(df.sort_values(by='B', ascending=True))
#print(df.mean()) # mean of each column (mean of each row df.mean(1))
#print(df['A'].sum()) # sum of column A

## If statements
#x = 1
#if x < 0:
#    x = 0
#    print('Negative changed to zero')
#elif x == 0:
#    print('Zero')
#elif x == 1:
#    print('Single')
#else:
#    print('More')
#
#if x != 5: # if x is not equal to 5
#    x += 5 # then set x = x + 5
#    print(x)

## Loops: for statements
#words = ['cat', 'window', 'demonstrate']
## iterate over items
#for w in words: 
#    print(w,len(w))
## iterate over indices
#for i in range(len(words)): 
#    print(i)
## iterate over both indices and items
#for i,w in enumerate(words):
#    print(i,w)
#
## double for loop
#for n in range(2, 10):
#    for x in range(2, n):
#        if n % x == 0:
#            print(n, 'equals', x, '*', n//x)
#            break
#        else:
#            # loop fell through without finding a factor
#            print(n,' is a prime number')
#
## while loop
## write a function that containts a while loop
#def fib(n):  # return Fibonacci series up to n
#    # Return a list containing the Fibonacci series up to n.
#    result = []
#    a, b = 0, 1
#    while a < n:
#        result.append(a)    # see below
#        a, b = b, a+b
#    return result
#print(fib(100))

# Load save data using numpy
x = np.loadtxt('input/text_x.txt', skiprows=1)
print(x)
sqr_x = x**2
print(sqr_x)
np.savetxt('output/sqr_x.txt',sqr_x,fmt='%.3f',header='x1_sqr x2_sqr')

# Load save data as it is
with open('input/text_x.txt') as f:
    w = f.read()
    print(w)
#
f = open('input/text_x.txt')
line = f.readlines()  
print(line)
#
with open('input/csv_x.csv') as f:
    w = f.read()
    print(w)

# use import csv with csv.reader and csv.writer
with open('input/csv_x.csv') as f:
    w = csv.reader(f)
    for row in w:
        print(row)
#
with open('output/cubes_x.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['x1_cube','x2_cube'])
    writer.writerows(x**3)
    
# Load save data using pandas - RECOMMENDED!!!!
# text file
df = pd.read_csv('input/text_x.txt')
print(df.keys())
# csv file
df = pd.read_csv('input/csv_x.csv')
print(df)
sqrt_x = df.applymap(np.sqrt)
sqrt_x.to_csv('output/sqrt_x.csv',index=False, header=['sqrt_x1','sqrt_x2'])


























