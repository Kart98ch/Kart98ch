import math

#Recursive - calls on itself, given a base case.

def karatsuba(x,y):
    if (x<10) or (y<10):
        return x*y
    else:
        n = max(len(str(x)),len(str(y)))
        
        n2 = math.floor(n/2)
        
        a = x // 10**(n2)
        b = x % 10**(n2)
        c = y // 10**(n2)
        d = y % 10**(n2)
        
        ac = karatsuba(a,c)
        bd = karatsuba(b,d)
        pp = karatsuba(a+b,c+d)
        part = pp - ac - bd
        
        prod = ac*(10**(2*n2)) + part*(10**(n2)) + bd
        
        return prod
