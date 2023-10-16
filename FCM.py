import networkx as nx
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

# Parse the Graphviz file
path = "./stock_fcm.dot" #TODO: Configurable DOT file
G = nx.Graph(nx.nx_pydot.read_dot(path))


# ------ Process graph file and initialize data from FCM ------
# Initialize the concepts and initial_values
concepts = []
initial_values = []

# Map nodeId and position of concepts
# this will be handy while creating the fuzzy weights
node_concept_map = {}

# Add the concepts and initial weights from the graphviz file
for node in list(G.nodes(data=True)):
    data = node[1]
    if(data):
        node_concept_map[node[0]] = len(concepts)
        concepts.append(data['label'].replace('"', ''))
        initial_values.append(float(data['value']))

# Create NxN fuzzy fuzzy weights metrix where n is the number of concepts
# prepopulate it with zeros
fuzzy_weights=np.zeros((len(concepts),len(concepts)), dtype=float)


# initialize fuzzy weights
for e in list(G.edges(data=True)):
    # e looks like ('node1', 'node2', {'label': '"+2"', 'weight': '2'})
    i1 = node_concept_map[e[0]]
    i2 = node_concept_map[e[1]]
    fuzzy_weights[i1][i2] = e[2]['weight']

  
# ------ GFCM: Generalized Fuzzy Cognitive Maps ------
# Transaction Function
def f(x):
    return 1 / (1 + np.exp(-x))

# Define FCM
class GFCM:
    def __init__(self, num_concepts, fuzzy_weights):
        self.num_concepts = num_concepts
        self.fuzzy_weights = fuzzy_weights

    def simulate(self, initial_values):
        new_values = np.zeros(self.num_concepts)
        for i in range(self.num_concepts):
            i = i-1
            new_values[i] = f(sum(self.fuzzy_weights[i,:] * initial_values[i]))
        return new_values

# ------ GFCM: Start the simulations over given time steps ------
values_overtime = []
values_overtime.append(concepts)

gfcm = GFCM(num_concepts=len(concepts), fuzzy_weights=fuzzy_weights)

for i in range(0,11): #TODO: Configurable time steps, currently 11
  new_values = gfcm.simulate(initial_values)
  values_overtime.append(new_values)
  initial_values = new_values

# Save the simulation results to CSV as per the graph name
# Timestamp
now =  datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
filename = "{name}_{time}".format(name =G.name, time = now )
df = pd.DataFrame(values_overtime)
df.columns = df.iloc[0]
df = df[1:]

df.to_csv(filename+".csv", index=False)



# ------ GFCM: Create Visualization from the time series ------

# plot each concept
for c in concepts:
    plt.plot(df[c],label=c)
    print(c)

# set x and y label
plt.ylabel('Values')
plt.xlabel('Time')
# set title and legend
plt.title('Interaction')
plt.legend(loc="upper left")

plt.savefig(filename+".png")