#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxpy as cp
import numpy as np
import math
import scipy.stats as spst
from scipy.optimize import minimize
import gurobipy as gp
from gurobipy import GRB
from itertools import product
import matplotlib.pyplot as plt
import time
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


env = gp.Env(empty=True)
env.setParam("OutputFlag",0)
env.start()


# In[3]:


def coeff(C): # factorial
    w = [1]
    for k in range(1,C+1):
        w.append(w[k-1]*k)
    return w 

def in_stock(x,C): # service level
    w = coeff(C)
    term = 0
    for i in range(C):
        term += x**i/w[i]
    return term/(term + x**C/w[C])  

# compute the service level for static policy lm
def service(C,lm,mu):
    lm = np.array(lm)
    x = np.sum(lm/mu)
    return in_stock(x,C)

# revenue function for static policy lm; p(lm)= b - a*lm
def revenue(lm,mu,b,a,C):
    lm = np.array(lm)
    rev = np.sum(lm*(b-a*lm))
    SL = service(C,lm,mu)
    return -rev*SL

# revenue function of the fluid formulation
def revenue_workload(lm,mu,b,a):
    lm = np.array(lm)
    return -np.sum(lm*(b-a*lm)/mu)
    


# In[4]:


## our paper: using BFGS to solve the optimal static pricing problem
def method1(b,a,mu,C,M):
    up_bd = b/a/2
    bds = []
    for j in range(M):
        bds.append((0,up_bd[j]))
    bds = tuple(bds)
    start_pt = up_bd/2
    result = minimize(revenue,start_pt,args=(mu,b,a,C),method='L-BFGS-B',bounds=bds)
    if result.success:
        return result.x,-result.fun

 # compute the revenue from static policy derived from solving fluid problem   
def method3(b,a,mu,C,M,Del):
    up_bd = b/a
    m = gp.Model(env=env)
    x = m.addVars(M,vtype = GRB.CONTINUOUS, name="x")
    m.addConstr((gp.quicksum(x[j]/mu[j] for j in range(M)) <= Del))
    for i in range(M):  
        m.addConstr((x[i]<=up_bd[i])) 
        m.addConstr((x[i]>=0))
    obj = gp.quicksum(x[j]*(b[j]-a[j]*x[j]) for j in range(M))
    m.setObjective(obj,GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        lm = []
        for v in m.getVars():
            lm.append(v.x)
        rev = -revenue(lm,mu,b,a,C)    
        return np.array(lm),rev
    if m.Status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or ""unbounded")
        return 
    if m.Status != GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        return

# perform line search over [0,3C]    
def line_search(b,a,mu,C,M,T):
    max_value = min(np.sum(b/a/mu),3*C)
    thre = np.linspace(0,max_value,T)
    sample = []
    for i in range(T):
        result = method3(b,a,mu,C,M,thre[i])
        sample.append(result[1])
#     index = np.argmax(sample)
#     max_result = method3(b,a,mu,C,M,Delta[index])
    result_C = method3(b,a,mu,C,M,C)[1]
    return C,max(np.max(sample),result_C)


# In[8]:


# with line search
   
def generate2(C,M,N,T):
    start_time = time.time()
    test = []
    profit = np.empty((3,N))
    for j in range(N):
        b = 0.5 + 9.5*np.random.rand(M)
        a = 0.1 + 4.9*np.random.rand(M)
        mu = 0.02 + 19.98*np.random.rand(M)
        test.append((C,b,a,mu))
        outcome1 = method1(b,a,mu,C,M)
        profit[0,j] = outcome1[1]
        outcome2 = method3(b,a,mu,C,M,C)
        profit[1,j] = outcome2[1]
        outcome3 = line_search(b,a,mu,C,M,T)
        profit[2,j] = outcome3[1]
    index1 = np.argmin(profit[1]/profit[0])
    index2 = np.argmin(profit[2]/profit[0])
    print('No_opt:M',M,'C',C,'ratio',np.min(profit[1]/profit[0]),np.mean(profit[1]/profit[0]))   
    print('Wi_opt:M',M,'C',C,'ratio',np.min(profit[2]/profit[0]),np.mean(profit[2]/profit[0])) 
    print("--- %s seconds ---" % (time.time() - start_time))
    return np.min(profit[1]/profit[0]),np.mean(profit[1]/profit[0]),np.min(profit[2]/profit[0]),np.mean(profit[2]/profit[0])


# In[10]:


N = 1000
T = 100
M_set = [5,10,15,20]
C_set = [5,10,15,20]
for M in M_set:
    for C in C_set:
        result = generate2(C,M,N,T)


# # Exponential

# In[11]:


def coeff(C): # factorial
    w = [1]
    for k in range(1,C+1):
        w.append(w[k-1]*k)
    return w 

def in_stock(x,C): # service level
    w = coeff(C)
    term = 0
    for i in range(C):
        term += x**i/w[i]
    return term/(term + x**C/w[C])  

def service(C,lm,mu):
    lm = np.array(lm)
    x = np.sum(lm/mu)
    return in_stock(x,C)


### p = a*ln(b/a/lm)
def revenue(lm,mu,b,a,C):
    lm = np.array(lm)
    rev = np.sum(lm*(a*(np.log(b/a/lm))))
    SL = service(C,lm,mu)
    return -rev*SL

def revenue_workload(lm,mu,b,a):
    lm = np.array(lm)
    return -np.sum(lm*(a*(np.log(b/a/lm))))
    


# In[12]:


## our paper: using gradient desecent or Frank-wolfe
def method1(b,a,mu,C,M):
    up_bd = b/a
    bds = []
    for j in range(M):
        bds.append((1e-8,up_bd[j]))
    bds = tuple(bds)
    start_pt = up_bd
    result1 = minimize(revenue,start_pt,args=(mu,b,a,C),method='L-BFGS-B',bounds=bds)
    result2 = minimize(revenue,start_pt/8,args=(mu,b,a,C),method='L-BFGS-B',bounds=bds)
    if result1.fun<result2.fun:
        return result1.x,-result1.fun
    else:
        return result2.x,-result2.fun


def method3(b,a,mu,C,M,Del):
    x = cp.Variable(M,nonneg=True)
    d = cp.entr(x)
    obj = (a*np.log(b/a))@x + a@cp.entr(x)
    objective = cp.Maximize(obj)
    constraints = [x <= b/a, sum(x/mu)<= Del]
    prob = cp.Problem(objective,constraints)
    prob.solve(solver='ECOS')
    if prob.status =='optimal':
        lm = np.array(x.value)
        rev = -revenue(lm,mu,b,a,C)
        return lm,rev
    else:
        return C,0
#     else: 
#         print('Optimization was stopped with status ' + str(prob.status))
#     return        
    
    
def line_search(b,a,mu,C,M,T):
    max_value = min(np.sum(b/a/mu),3*C)
    thre = np.linspace(0,max_value,T)
    sample = []
    for i in range(T):
        try:
            result = method3(b,a,mu,C,M,thre[i])
            sample.append(result[1])
        except:
            pass
    sample.append(method3(b,a,mu,C,M,C)[1])
    return C,np.nanmax(sample)


# In[13]:


def generate2(C,M,N,T):
    start_time = time.time()
    profit = np.empty((3,N))
    for j in range(N):
        b = 0.5 + 9.5*np.random.rand(M)
        a = 0.1 + 4.9*np.random.rand(M)
        mu = 0.02 + 19.98*np.random.rand(M)
        try:
            outcome2 = method3(b,a,mu,C,M,C)
            profit[1,j] = outcome2[1]
            outcome3 = line_search(b,a,mu,C,M,T)
            profit[2,j] = outcome3[1]
        except: 
            pass
        outcome1 = method1(b,a,mu,C,M)
        profit[0,j] = outcome1[1]
    valid1 =  ((profit[1]/profit[0])<=1)*((profit[1]/profit[0])>0.6)
    valid2 =  ((profit[2]/profit[0])<=1)*((profit[2]/profit[0])>0.6)
    valen1 = np.nansum(valid1)
    valen2 = np.nansum(valid2)
    new1 = (profit[1]/profit[0])*valid1
    new2 = (profit[2]/profit[0])*valid2
    print('No_opt:M',M,'C',C,'ratio',np.nanmin(new1[np.nonzero(new1)]),np.nansum(new1)/valen1)
    print('Wi_opt:M',M,'C',C,'ratio',np.nanmin(new2[np.nonzero(new2)]),np.nansum(new2)/valen2)
    print("--- %s seconds ---" % (time.time() - start_time))
    return profit
# np.nanmin(profit[1]/profit[0]),np.nanmean(profit[1]/profit[0]),np.nanmin(profit[2]/profit[0]),np.nanmean(profit[2]/profit[0])


# In[14]:


N = 1000
T = 100
M_set = [5,10,15,20]
C_set = [5,10,15,20]
test = []
for M in M_set:
    for C in C_set:
        result = generate2(C,M,N,T)
        test.append(result)

