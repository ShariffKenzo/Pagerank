import numpy as np

def pageranksimple(graph, iterations):
    
    numberOfpages = len(graph) #since it is n by n elements

    ranks = []
    for _ in range(numberOfpages):
        element = float(1/numberOfpages)
        ranks.append([element])
    #print(ranks)
    # Create the column vector using NumPy array and place the ranks in it
    column_vector = np.array(ranks)
    print(column_vector)

    for i in range(iterations):
        column_vector = np.dot(graph, column_vector)
        print("rank is :")
        print(column_vector)

graph = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

#print (graph[2])
#print(len(graph))
#pageranksimple(graph=graph,)

M= np.array([
    [1/2, 1/2, 0],
    [1/2, 0, 1],
    [0, 1/2, 0]
])

#pageranksimple(M,100)

def pagerankmodified(graph, dampingFactor, E,  iterations):

    c1 = dampingFactor
    c2 = 1 -c1

    numberOfpages = len(graph) #since it is n by n elements
    #initial ranks for the webpages
    ranks = []
    for _ in range(numberOfpages):
        element = float(1/numberOfpages)
        ranks.append([element])
    #print(ranks)
    # Create the column vector using NumPy array and place the ranks in it
    column_vector = np.array(ranks)
    print(column_vector)

    for i in range(iterations):
        column_vector = c1 * np.dot(graph, column_vector) + c2 * (np.ones(numberOfpages)/numberOfpages).reshape(-1, 1)
        print("rank is :")
        print(column_vector)


M= np.array([
    [1/2, 1/2, 0],
    [1/2, 0, 0],
    [0, 1/2, 1]
])

pagerankmodified(graph= M, dampingFactor= 0.8, E = 5, iterations=100)

#print(np.ones(5)/5)