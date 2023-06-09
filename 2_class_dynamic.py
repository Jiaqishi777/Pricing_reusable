# -*- coding: utf-8 -*-
"""2-class_dynamic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13atgMOF5r4rh-5Gg-mf-CHAZ2kVRY_gT
"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import math
# %matplotlib inline
import scipy.stats as spst     
from scipy.optimize import minimize
from scipy.optimize import fsolve

## compute constructed static policy given optimal dynamic policy: lm & corresponding steady-state probs
def static(lm,P,C):
  lmbda = np.zeros(2)
  term = 0
  for i in range(C+1):
    term += P[i,C-i]
  for j in range(2):
    for k in range(C):
      for i in range(k+1):
        lmbda[j] += lm[j,i,k-i]*P[i,k-i]/(1-term)
  return lmbda  

## compute the steady-state probs under the constructed static policy
def static_prob(lmbda,C,mu):
  term = lmbda[0]/mu[0] + lmbda[1]/mu[1]
  term1 = 0
  for k in range(C+1):
    term1 += term**k/math.factorial(k)
  P = np.zeros((C+1,C+1))
  for k in range(C+1):
    for i in range(k+1):
      P[i,k-i] = (lmbda[0]/mu[0])**(i)*(lmbda[1]/mu[1])**(k-i)/math.factorial(i)/math.factorial(k-i)/term1
  return P


## compute the profit of optimal dynamic policy when demand is linear, p(lm) = b -a*lm
def dynamic_profit(lm,P,b,a,C):
  profit = np.zeros(2)
  for j in range(2):
    for k in range(C):
      for i in range(k+1):
        profit[j] += lm[j,i,k-i]*(b[j]-a[j]*lm[j,i,k-i])*P[i,k-i]  
  return profit

## compute the profit of constructed static policy when demand is linear, p(lm) = b -a*lm
def static_profit(lmbda,P,b,a,C):
  profit = np.zeros(2)
  term = 0
  for i in range(C+1):
    term += P[i,C-i]  
  for j in range(2):
    profit[j] = lmbda[j]*(b[j]-a[j]*lmbda[j])*(1-term)
  return profit        

def print_solution(lm,lmbda,Prob,sta_prob,opt_rev,sta_rev):
  print('optimal policy:','\n',lm)
  print('static policy:',lmbda)
  print('Optimal_steady-state:','\n',Prob)
  print('static steady-state:','\n',sta_prob)
  print("opt_profit:",np.sum(opt_rev),"class 1",opt_rev[0],'class 2',opt_rev[1])
  print("sta_profit:",np.sum(sta_rev),"class 1",sta_rev[0],'class 2',sta_rev[1])
  print('ratio:',np.sum(sta_rev)/np.sum(opt_rev))

## We use relative value iteration to compute the optimal dynamic policy

def generate(C,b,a,mu):
  tot = 1e-8
  ## standard uniformization
  kai = 1/(C*np.max(mu)+ np.sum(b/a) + 1 )
  v = np.zeros((C+1,C+1))
  w = np.zeros((C+1,C+1))
  while True:
    v_old = v.copy()
    for i in range(C+1):
      w[i,C-i] = kai*i*mu[0]*v[i-1,C-i]*(i>0) + kai*(C-i)*mu[1]*v[i,C-i-1]*((C-i)>0) + (1- kai*(i*mu[0]*(i>0) + (C-i)*mu[1]*((C-i)>0) ))*v[i,C-i]
    for k in range(C):
      for i in range(k+1):
        constant = np.zeros(2)
        constant[0] = (b[0] + kai*v[i+1,k-i]-kai*v[i,k-i])/(2*a[0])
        constant[1] = (b[1] + kai*v[i,k-i+1]-kai*v[i,k-i])/(2*a[1])
        for j in range(2):
          constant[j] = min(max(constant[j],0),b[j]/a[j])
        w[i,k-i] = constant[0]*(b[0]-a[0]*constant[0]) + constant[1]*(b[1]-a[1]*constant[1]) + kai*constant[0]*(v[i+1,k-i]-v[i,k-i]) + kai*constant[1]*(v[i,k-i+1]-v[i,k-i]) + kai*mu[0]*i*(v[i-1,k-i]-v[i,k-i])*(i>0) + kai*mu[1]*(k-i)*(v[i,k-i-1]-v[i,k-i])*((k-i)>0) + v[i,k-i]
    for k in range(C+1):
      for i in range(k+1):
        v[i,k-i] = w[i,k-i] - w[C-1,1]  
    if all(abs(v_old[s,t]-v[s,t])<= tot  for s,t in zip(range(C+1),range(C+1))):
        break
  ## calcute the optimal dynmaic policy based on the last value of v
  lm = np.zeros((2,C,C))
  for k in range(C):
    for i in range(k+1):
      constant = np.zeros(2)
      constant[0] = (b[0] + kai*v[i+1,k-i]-kai*v[i,k-i])/(2*a[0])
      constant[1] = (b[1] + kai*v[i,k-i+1]-kai*v[i,k-i])/(2*a[1])
      for j in range(2):
        lm[j,i,k-i] = min(max(constant[j],0),b[j]/a[j])  
  ### compute the steady-state probs under the optimal dynamic policy lm
  P = cp.Variable((C+1,C+1),pos=True)
  obj = cp.Minimize(1)
  constraints = [ 
              lm[0,C-1,0]*P[C-1,0] == C*mu[0]*P[C,0],
              lm[1,0,C-1]*P[0,C-1] == C*mu[1]*P[0,C],
              (lm[0,0,0]+lm[1,0,0])*P[0,0] == mu[0]*P[1,0] + mu[1]*P[0,1] 
  ]
  for i in range(1,C):
    constraints += [ P[i,C-i]*(i*mu[0] + (C-i)*mu[1] ) == lm[0,i-1,C-i]*P[i-1,C-i] + lm[1,i,C-i-1]*P[i,C-i-1] ]
  for i in range(1,C):
    constraints += [ P[i,0]*(lm[0,i,0]+lm[1,i,0]+i*mu[0]) == P[i-1,0]*lm[0,i-1,0] + P[i+1,0]*(i+1)*mu[0] + P[i,1]*mu[1]  ]  
  for i in range(1,C):
    constraints += [ P[0,i]*(lm[1,0,i]+lm[0,0,i]+i*mu[1]) == P[0,i-1]*lm[1,0,i-1] + P[0,i+1]*(i+1)*mu[1] + P[1,i]*mu[0]  ]    
  if C>=3:
    for k in range(2,C-1):
      for i in range(1,C):
        constraints += [ P[i,k-i]*(lm[0,i,k-i]+lm[1,i,k-i]+i*mu[0]+(k-i)*mu[1]) == P[i-1,k-i]*lm[0,i-1,k-i] + P[i,k-i-1]*lm[1,i,k-i-1] + P[i+1,k-i]*(i+1)*mu[0] + P[i,k-i+1]*(k-i+1)*mu[1] ]
  sumation = 0
  for k in range(C+1):
    for i in range(k+1):
      sumation += P[i,k-i]
  constraints += [ sumation == 1]
  problem = cp.Problem(obj,constraints)
  result = problem.solve()
  if problem.status not in ["infeasible", "unbounded"]:
    Prob = P.value   
    static_lm = static(lm,Prob,C)
    static_probability = static_prob(static_lm,C,mu)
    profit_opt = dynamic_profit(lm,Prob,b,a,C)
    profit_sta = static_profit(static_lm,static_probability,b,a,C)
    ratio = np.sum(profit_sta)/np.sum(profit_opt)
    print_solution(lm,static_lm,Prob,static_probability,profit_opt,profit_sta)
    return 
  else:
    print("infesible policy:",'\n',lm )
    print(problem)

b = np.array([100,10])
a = np.array([0.05,50])     ##### p1(lm1) = 100-0.05*lm1, p2(lm2) = 10-50*lm2
mu = np.array([0.001,1000]) # mu1 = 0.001, mu2 = 1000
C = 3
generate(C,b,a,mu)

