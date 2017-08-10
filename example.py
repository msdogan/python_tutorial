#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:31:19 2017

@author: msdogan
"""
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

#print(installed_solvers())

# # ********** Example 1 **********

# # Create two scalar optimization variables.
# x = Variable()
# y = Variable()

# # Create two constraints.
# constraints = [ x + y == 1,
#                 x - y >= 1 ]

# # Form objective.
# obj = Minimize(square(x - y))

# # Form and solve problem.
# prob = Problem(obj, constraints)
# prob.solve(verbose=True)

# # job status
# print(prob.status)
# # The optimal dual variable (Lagrange multiplier) for
# # a constraint is stored in constraint.dual_value.
# print("optimal (x + y == 1) dual variable", constraints[0].dual_value)
# print("optimal (x - y >= 1) dual variable", constraints[1].dual_value)
# print("x - y value:", (x - y).value)
# # objective value
# print(prob.value)
# # variable optimal value
# print(x.value)

# # ********** Example 2 **********

# # Problem data.
# m = 3
# n = 2
# numpy.random.seed(1)
# A = numpy.random.randn(m, n)
# b = numpy.random.randn(m)

# # Construct the problem.
# x = Variable(n)
# objective = Minimize(sum_squares(A*x - b))
# constraints = [0 <= x, x <= 1]
# prob = Problem(objective, constraints)

# # The optimal objective is returned by prob.solve().
# prob.solve(verbose=True)
# # The optimal value for x is stored in x.value.
# print(x.value)
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print(constraints[0].dual_value)

# ********** Example 3 **********
'''
ECI 153 Example

A company has 3 factories that together produce two products: cars and small trucks.
Each factory specializes in certain components required for both products.
Plant 1 makes truck bodies.
Plant 2 makes car bodies.
Plant 3 makes shared components and assembles trucks and cars.

Each plant has limited capacity:
Plant 1: less than 4 car per hour (car <= 4)
Plant 2: less than 6 trucks per hour (truck <= 6)
Plant 3: less than 6 cars or 9 trucks per hour (3*car + 2*truck <= 18)

Each product is sold for a fixed price with a fixed profit per unit:
car: $3, truck: $5

How many cars and truck should be produced to maximize profit per hour?
'''

# define variables
car = Variable()
truck = Variable()

# create constraints
constraints = [ car <= 4, # capacity constraints on production
                truck <= 6, # capacity constraints on production
                3*car + 2*truck <= 18, # capacity constraints on production
                car >= 0, # cannot produce negative products
                truck >= 0 ] # cannot produce negative products

# Form objective.
obj = Maximize(3*car + 5*truck)

# Form and solve problem.
prob = Problem(obj, constraints)
prob.solve(verbose=True)
# job status
print(prob.status)
print('optimal number or cars: ', int(car.value)) # optimal number or cars
print('optimal number or trucks: ', int(truck.value)) # optimal number or trucks
print('profit ($) per hour: ', round(obj.value,2)) # objective value
# dual values (Lagrange multipliers)
print(constraints[0].dual_value) # first constraint
print(constraints[1].dual_value) #  second constraint
print(constraints[2].dual_value) # third constraint

# plot results
plt.bar([0],car.value)
plt.bar([1],truck.value)
plt.xticks([0,1],['car','truck'])
plt.ylabel('Production (Unit/hour)')
plt.title('Optimal Factory Production')
plt.show()