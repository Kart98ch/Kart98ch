class Graph():
    def __init__(self, graph_dict=None):
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict
        
    def vertices(self):
        return list(self.__graph_dict.keys())
    
    def add_vertex(self,vertex):
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = {}
    
    def add_edge(self,edge,weight):
        vertex1,vertex2 = edge
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1][vertex2] = weight
        else:
            self.add_vertex(vertex1)
            self.__graph_dict[vertex1][vertex2] = weight
    
    def vertedge(self,vertex):
        edges = []
        for neighbour in self.__graph_dict[vertex]:
            edges.append(([vertex,neighbour],self.__graph_dict[vertex][neighbour]))
        return edges
    
    def test(self):
        print("testing")
    
    def dijkstra(self, start):
        visited = {}
        unvisited = {}
        for vertex in list(self.__graph_dict.keys()):
            if vertex==start:
                unvisited[vertex] = {'cost':0,'prev':None}
            else:
                unvisited[vertex] = {'cost':float('inf'),'prev':None}

        while len(unvisited)!=0:
            costs = {x:unvisited[x]['cost'] for x in list(unvisited.keys())}
            current_node = min(costs,key=costs.get)
            
            for neighbour in list(self.__graph_dict[current_node].keys()):
                if neighbour not in visited:
                    c = unvisited[current_node]['cost']
                    d = c + self.__graph_dict[current_node][neighbour]
                    if d < unvisited[neighbour]['cost']:
                        unvisited[neighbour]['cost'] = d
                        unvisited[neighbour]['prev'] = current_node
                        
            visited[current_node] = unvisited[current_node]
            del unvisited[current_node]
        return visited