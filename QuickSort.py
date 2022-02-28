# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 13:09:44 2021

@author: karti
"""

file = open('test.txt','r')

intlist = file.readlines()

for i in range(len(intlist)):
    intlist[i] = int(intlist[i][:-1])

def quicksort(A, piv):
    
    comparisons = 0
    
    if (len(A)==1) or (len(A)==0) :
        return A, comparisons
    else:
        p = A[piv]
        i = piv+1
        for j in range(piv+1,len(A)):
            if A[j]<=p:
                a = A[j]
                b = A[i]
                A[j] = b
                A[i] = a
                i += 1
                
            else:
                continue
        
        c = A[piv]
        d = A[i-1]
        A[piv] = d
        A[i-1] = c
        
        comparisons += len(A)-1
        
        B = A[:i]
        C = A[i+1:]
        
 #       print('B = %s' %(B))
  #      print('C = %s' %(C))
   #     print('Next recursion.')
        
        D, r = quicksort(B,piv)
        E, t = quicksort(C,piv)
        
        A[:i] = D
        A[i+1:] = E
        comparisons += r
        comparisons += t
        
    return A, comparisons