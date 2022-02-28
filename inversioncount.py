def Merge_Sort(array):
    if len(array)==1:
        return array, 0
    else:
        n = len(array)
        n2 = n//2
        
        obj1 = array[:n2]
        sortobj1, inv1 = Merge_Sort(obj1)
        
        obj2 = array[n2:]
        sortobj2, inv2 = Merge_Sort(obj2)
        
        i=0
        j=0
        inv = 0
        new = []
        
    for k in range(n):
        if sortobj1[i]>sortobj2[j]:
            new.append(sortobj2[j])
            
            if j == len(sortobj2)-1:
                new += sortobj1[i:]
                inv += len(sortobj1[i:])
                break
            else:
                j += 1
                inv += (n2-i)
            
        else:
            new.append(sortobj1[i])
            
            if i == len(sortobj1)-1:
                new += sortobj2[j:]
                
                break
            else:
                i += 1
            
    totinv = inv + inv1 + inv2
    return new, totinv


#def inversions(array):
 #   if len(array)==1:
  #      return array
   # else:
        