import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

import robust_neural_net

# CNN INITIALIZATION
loaded_model = robust_neural_net.RobustConvNet()
loaded_model.load_state_dict(torch.load('robust_conv_net.pth'))
loaded_model.eval()

# GENETIC ALGORITHM PARAMETERS
rows = 28
columns = 28
population_size = 40
mutation_rate = 0.005

# GENETIC ALGORITHM CLASS
class Member:
    def __init__(self, seed):
        self.seed = seed
        self.rows = seed.shape[0]
        self.columns = seed.shape[1]

    def convert_matrix_format(self):
        image_array = np.array(self.seed, dtype=np.uint8) * 255
        image = Image.fromarray(image_array)
        blurred_image = image.convert("L").filter(ImageFilter.GaussianBlur(0.4))
        return np.array(blurred_image)

    def score(self):
        seed_tensor = torch.FloatTensor(self.convert_matrix_format()).unsqueeze(0).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        logits = loaded_model(seed_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        output = probabilities[0, 6].item()
        loss = 1 - output
        return np.exp(-loss)

    def mutate(self, mutation_rate):
        mask = np.random.uniform(0, 1, size=(self.rows, self.columns)) < mutation_rate
        self.seed[mask] = 1 - self.seed[mask]

class Population:

    def __init__(self, population_size, mutation_rate, rows, columns):
        self.population_size = population_size
        self.rows = rows
        self.columns = columns
        self.mutation_rate = mutation_rate
        self.members = [Member(np.ones((rows, columns))) for _ in range(population_size)]
        self.sorted_population = sorted(self.members, key=lambda x: x.score(), reverse=True)
        total_fitness = sum(member.score() for member in self.members)
        self.fitness = [member.score()/total_fitness for member in self.members ]


    def print_best_seed(self):
        return self.sorted_population

    def choose_parents(self, num_of_parents):
        # based on our generated a list of members sorted by score
        # grabs the start of the sorted list - the best parents.

        parents = self.sorted_population[:num_of_parents]

        # return new list

        return parents

    def mutate(self, member_list):
        for member_inst in member_list:
            member_inst.mutate(mutation_rate=self.mutation_rate)

    def create_offsprings(self, member_1,member_2):
        new_seed = np.zeros(member_1.seed.shape)
        new_seed_2 = np.zeros(member_1.seed.shape)
        for i in range(member_1.seed.shape[0]):  # iterate over rows: # for every row
            j  = np.random.randint(0, member_1.seed.shape[1]-1) # create a random cutoff point
            new_seed[i] = np.append(member_1.seed[i][:j], member_2.seed[i][j:]) # create two offsprings like in the paper
            new_seed_2[i]  = np.append(member_2.seed[i][:j], member_1.seed[i][j:])
        return [Member(new_seed), Member(new_seed_2)]

    def choose_parents_in_random(self):
        parents = (np.random.choice(self.members, 2 ,p=  self.fitness ))
        return parents


    def reproduction(self):
        crossover_num = int(self.population_size * 0.75)
        crossovers = self.choose_parents(crossover_num )
        self.mutate(crossovers)
        reproduction_num = int(self.population_size * 0.25)
        for i in range(reproduction_num//2):
            parents = self.choose_parents_in_random()
            crossovers.extend(self.create_offsprings(*parents))
        self.members = crossovers




