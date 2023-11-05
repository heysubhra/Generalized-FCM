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