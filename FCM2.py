import networkx as nx
import numpy as np
import pandas as pd
import datetime

import os
from subprocess import Popen, PIPE


class FCM:
    def __init__(self, concepts, initial_values, w_mat_file):
        self.concepts = concepts
        self.initial_values = initial_values
        self.w_mat_file = w_mat_file

    def f(x):
        return 1 / (1 + np.exp(-x))

    def simulate(self, iterations):

        concepts = str(list(self.concepts)).replace("[", '').replace("]", '')
        initial_values = str(list(self.initial_values)).replace("[", '').replace("]", '')
        command = 'Rscript ./fcm.R "%s" "%s" "%s" %d' % (initial_values, concepts, self.w_mat_file, iterations)

        process = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = process.communicate()

        if stderr:
            raise Exception(stderr)
        else:
            return


class Simulation:
    graph = None
    weighted_matrix = None
    concepts = []
    activation_vector = []
    concept_pos_map = {}
    simulation_result = []

    def df_to_csv(self, df, fname, index=False):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = "{name}_{time}.csv".format(name=fname, time=now)
        filename = filename.replace(" ","")
        df.to_csv(filename, index=index)
        return filename

    def parse_graphviz(self, path):
        """Parses a Graphviz file into a NetworkX graph"""
        G = nx.DiGraph(nx.nx_pydot.read_dot(path))
        self.graph = G
        return self

    def init_fcm(self):
        """ Turn a NetworkX graph and obtain the concepts and initial weights."""

        # Map nodeId and position of concepts
        # this will be handy while creating the fuzzy weights

        for node in list(self.graph.nodes(data=True)):
            data = node[1]
            if data:
                self.concept_pos_map[node[0]] = len(self.concepts)
                if 'label' in data.keys():
                    self.concepts.append(data['label'].replace('"', ''))
                if 'xlabel' in data.keys():
                    self.concepts.append(data['xlabel'].replace('"', ''))
                self.activation_vector.append(float(data['value']))

        return self

    def create_weighted_matrix(self):
        """Creates a NxN fuzzy weights matrix where n is the number of concepts."""

        self.weighted_matrix = np.zeros((len(self.concepts), len(self.concepts)), dtype=float)

        for e in list(self.graph.edges(data=True)):
            i1 = self.concept_pos_map[e[0]]
            i2 = self.concept_pos_map[e[1]]
            self.weighted_matrix[i1][i2] = e[2]['weight']

        return self.weighted_matrix

    def simulate_fcm(self, num_time_steps, w_mat_file):
        """Simulates a FCM over a given number of time steps."""

        initial_values = self.activation_vector
        fcm = FCM(concepts=self.concept_pos_map.keys(), initial_values=initial_values, w_mat_file=w_mat_file)
        fcm.simulate(num_time_steps)

        simulation_results_file = "simulation_result_{fname}".format(fname=w_mat_file).replace(" ", "")

        return simulation_results_file

    def save_simulation_results_to_csv(self, filename):
        """Saves the simulation results to a CSV file."""
        df = pd.DataFrame(self.simulation_result)
        df.columns = self.concepts

        self.df_to_csv(df, self.graph.name + "results")

class GraphAnalysis:
    graph = None

    def __init__(self, graph):
        self.graph = graph

    def centrality(self):
        centrality = nx.degree_centrality(self.graph)
        return centrality

    def indegree(self):
        """Calculates the indegree of each node in an FCM."""
        indegree = nx.in_degree_centrality(self.graph)
        return indegree

    def outdegree(self):
        """Calculates the outdegree of each node in an FCM."""
        outdegree = nx.out_degree_centrality(self.graph)
        return outdegree

def test():
    """Main function."""

    # # Read the Graphviz file
    graphviz_path = "stockFCM.dot"
    #
    # #initiate simulation
    sim = Simulation()
    #
    # # Process graph file and initiate FCM
    FCMSim = sim.parse_graphviz(graphviz_path).init_fcm()
    #
    # # Create fuzzy weights matrix and save it
    wMat = FCMSim.create_weighted_matrix()
    wMatDf = pd.DataFrame(wMat)
    wMatDf.columns = list(sim.concept_pos_map.keys())
    wMat_filename = sim.df_to_csv(wMatDf, "wmat")

    simulation_results_file = FCMSim.simulate_fcm(10, wMat_filename)
    simulation_resuts = pd.read_csv(simulation_results_file)


    # Prepare the result dataset for plotting
    del simulation_resuts[simulation_resuts.columns[0]]
    column_names = list(simulation_resuts.columns)
    column_names[1:] = FCMSim.concepts
    simulation_resuts.columns = column_names


    analysis = GraphAnalysis(FCMSim.graph)

    # os.remove(simulation_results_file)
    # os.remove(wMat_filename)

    # print(analysis.indegree())
    # print(analysis.outdegree())
    # print(analysis.centrality())
    # print(simulation_resuts)

    return


if __name__ == "__main__":
    # main()
    test()
