#import re
import random


ed = list(range(200))

def random_edge(edge_list):
    v2 = random.choice(edge_list)
    edge_list.remove(v2)
    v1 = random.choice(edge_list)
    
    return v1, v2


# Remember that all indexes i refer to the i+1th node.

def contract_edge(v1,v2,g):
    
    #edge = (v1,v2)
    #merge node v2 into v1. Add all adjacent nodes from v2 to v1, then make v2's adj list into [].
    
    g[v1] = g[v1] + g[v2]
    g[v2] = []
    
    #remove all occurences of v2 in adj lists with v1.
    
    for i in range(len(g)):
        g[i] = [v1+1 if j==(v2+1) else j for j in g[i]]
    
    #remove all self loops.
    while (v1+1 in g[v1]):
        g[v1].remove(v1+1)
    


minlist = []

for i in range(500):
    f = open('Adj_list.txt')
    lines = f.readlines()
    f.close()

    g = []

    for line in lines:
        k = []
        for i in range(len(line.split())-1):
            k.append(int(line.split()[i+1]))
        g.append(k)

    ed = list(range(len(g)))
    
    while (len(ed)>2):
        v1,v2 = random_edge(ed)
        contract_edge(v1,v2,g)
    minlist.append(len(g[ed[0]]))
    
print(min(minlist))













