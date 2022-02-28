#Median implementation.

file = open('QuickSort.txt','r')

intlist = file.readlines()

for i in range(len(intlist)):
    intlist[i] = int(intlist[i][:-1])



def median(A):
    mid = round(len(A)/2 + 1/100)
    L = [A[0], A[len(A)-1], A[mid-1]]
    H = sorted(L)
    print(H)
    G = H[1]
    print('median = %s'%(G))
    
    if G == A[0]:
        return 0
    elif G == A[len(A)-1]:
        return len(A)-1
    elif G == A[mid-1]:
        return mid-1


def med(A):
    mid = round(len(A)/2)
    a = A[0]
    b = A[mid]
    c = A[len(A)-1]
    if a > b:
        if a < c:
            median = 0
        elif b > c:
            median = mid
        else:
            median = len(A)-1
    else:
        if a > c:
            median = 0
        elif b < c:
            median = mid
        else:
            median = len(A)-1
            
    return median



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


def xesort(A,n):
    if n<2:
        return A, 0
    
    else:
    
        m = median(A)
        
        A[0],A[m]=A[m],A[0]
        
        piv = 0
        
        p_ind = partition(A,piv,len(A))
        
        A[:p_ind-1], l = xesort(A[:p_ind-1],len(A[:p_ind-1]))
        
        A[p_ind:], r = xesort(A[p_ind:],len(A[p_ind:]))
        
        count = r + l + n-1
                
    return A, count