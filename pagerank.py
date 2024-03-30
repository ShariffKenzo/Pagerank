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
    if(E == 'same'):
     for i in range(iterations):
        column_vector = c1 * np.dot(graph, column_vector) + c2 * (np.ones(numberOfpages)/numberOfpages).reshape(-1, 1)
        print("rank is :")
        print(column_vector)
    else:
     for i in range(iterations):
        column_vector = c1 * np.dot(graph, column_vector) + c2 * (np.array(E)).reshape(-1, 1)
        print("rank is :")
        print(column_vector)
    return column_vector
       


M= np.array([
    [1/2, 1/2, 0],
    [1/2, 0, 0],
    [0, 1/2, 1]
])

#pagerankmodified(graph= M, dampingFactor= 0.8, E = 'same', iterations=100)

#pagerankmodified(graph= M, dampingFactor= 0.8, E = [1/3, 1/3, 1/3], iterations=100)
#pagerankmodified(graph= M, dampingFactor= 0.8, E = "same", iterations=100)
#vect = [0,1,2]
#print(np.array(vect).reshape(-1, 1))
#print(np.ones(5)/5)

transition_matrix =  np.array([
    [0, 1/2, 0, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 0, 1, 1/2],
    [1/3, 1/2, 0, 0]
])

############# TESTING WITH DIFFERENT VALUES OF teleportation probability change value of 
############# dampingFactor, note teleportaton probability = 1-dampingFactor

pagerankmodified(graph = transition_matrix, dampingFactor=0.8, E = "same", iterations=100)


############# TESTING WITH DIFFERENT VALUES OF E COMMENT OUT TO SELECT THE E ###########
#distribution_probabilty = [1/4, 1/4, 1/4, 1/4]
#distribution_probabilty = [1/5, 1/5, 1/5, 2/5]
#distribution_probabilty = [1/6, 1/6, 1/6, 1/2]
#distribution_probabilty = [1/7, 1/7, 1/7, 4/7]
distribution_probabilty = [1/8, 1/8, 1/8, 5/8]

# COMMENT OUT TO RUN ####
#pagerankmodified(graph = transition_matrix, dampingFactor=0.8, E = distribution_probabilty, iterations=100)


############## TESTING WITH DIFFERENT web graph matrix M   #####################
transition_matrix =  np.array([
    [0, 1/2, 0, 0],
    [1/3, 0, 0, 1/2],
    [1/3, 0, 1, 1/2],
    [1/3, 1/2, 0, 0]
])

transition_matrix_1 =  np.array([
    [0, 1/3, 0, 0, 1/4],
    [1/4, 0, 0, 1/2, 1/4],
    [1/4, 0, 1, 1/2, 1/4],
    [1/4, 1/3, 0, 0, 1/4],
    [1/4, 1/3, 0, 0, 0]
])

transition_matrix_2 =  np.array([
    [0,   1/4, 0, 0,   0,   1/3],
    [1/5, 0,   0, 1/2, 0,   1/3],
    [1/5, 0,   1, 1/2, 0,   0],
    [1/5, 1/4, 0, 0,   1/3, 0],
    [1/5, 1/4, 0, 0,   1/3, 1/3],
    [1/5, 1/4, 0, 0,   1/3, 0]
])

####### COMMENT OUT TO RUN ##########
#pagerankmodified(graph = transition_matrix, dampingFactor=0.8, E = "same", iterations=100)
#pagerankmodified(graph = transition_matrix_1, dampingFactor=0.8, E = "same", iterations=100)
#pagerankmodified(graph = transition_matrix_2, dampingFactor=0.8, E = "same", iterations=100)


########## Map reduce portion for pagerank ################

def getChunks(vector):
   # Split the matrix horizontally into two sets
    split_length = int(len(vector)/2)
    top_half = vector[:split_length, :]  # Get the first 2 rows
    bottom_half = vector[split_length:, :]  # Get the last 2 rows
    print(f"top half : {top_half},\n bottom_half : {bottom_half}")


def mapTask(graph, dampingFactor, E,  iterations):
    #break into chunks
    numberOfpages = len(graph)
    halfPages = int(numberOfpages/2)
    final_column_vector = []
    # Extract chunk1
    chunk1 = graph[:halfPages, :halfPages]
    # Extract chunk2
    chunk2 = graph[halfPages:, :halfPages]
    #print(f"chunk 1: {chunk1}, \nchunk 2: {chunk2}")
    final_column_vector.append(pagerankmodified(graph=chunk1, dampingFactor=dampingFactor, E=E, iterations=iterations))
    final_column_vector.append(pagerankmodified(graph=chunk2, dampingFactor=dampingFactor, E=E, iterations=iterations))
    return final_column_vector


def reduceTask(final_column_vector):
   processed_column_vector = np.vstack(final_column_vector)
   return processed_column_vector


matrix_to_map = np.array([
    [1/2, 1/2, 0, 0],
    [1/2, 1/2, 0, 0],
    [0, 0, 1/2, 1],
    [0, 0, 1/2, 0]
])

################ UNCOMMENT TO RUN mapReduce for 4 by 4 matrix ############
#finaloutput = reduceTask(mapTask(graph=matrix_to_map, dampingFactor=0.8, E ="same", iterations=100))
#print(f"the final output for mapreduce is\n {finaloutput}")

