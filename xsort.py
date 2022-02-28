# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:40:32 2021

@author: karti
"""
file = open('test.txt','r')

intlist = file.readlines()

for i in range(len(intlist)):
    intlist[i] = int(intlist[i][:-1])


def q1pivot(n):
    return 0


def partition(A,l,r):
    p = A[l]
    i = l+1
    for j in range(l+1,r):
        if A[j]<p:
            A[i],A[j] = A[j],A[i]
            i+=1
            
    A[l],A[i-1] = A[i-1],A[l]
    
#    print(A)
    return i


def xsort(A,n):
    if n<2:
        return A, 0
    
    else:
        
        A[0],A[-1]=A[-1],A[0]
        
        piv = 0
        
        p_ind = partition(A,piv,len(A))
        
        A[:p_ind-1], l = xsort(A[:p_ind-1],len(A[:p_ind-1]))
        
        A[p_ind:], r = xsort(A[p_ind:],len(A[p_ind:]))
        
        count = r + l + n-1
                
    return A, count