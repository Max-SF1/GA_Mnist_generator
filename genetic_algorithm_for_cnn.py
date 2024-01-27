import torch
import torchvision.transforms as transforms
from torchsummary import summary
import robust_neural_net
import genetic_algorithm_for_cnn as ga
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

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
        blurred_image = image.convert("L").filter(ImageFilter.GaussianBlur(2))
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

    def print_best_seed(self):
        return sorted(self.members, key=lambda x: x.score(), reverse=True)

    def choose_parent(self):
        fitness = [genome.score() for genome in self.members]
        probabilities = np.array(fitness) / sum(fitness)
        return np.random.choice(self.members, p=probabilities)

    def reproduction(self):
        parents = [self.choose_parent() for _ in range(self.population_size)]
        children = [Member(np.concatenate((parents[i].seed[:parents[i].rows // 2, :], parents[self.population_size - i - 1].seed[parents[i].rows // 2:, :]))) for i in range(self.population_size // 2)]
        if self.population_size % 2 == 1:
            children.append(Member(self.choose_parent().seed))
        self.members = children

    def mutate_the_kids(self):
        for member_inst in self.members:
            member_inst.mutate(mutation_rate=self.mutation_rate)

# TRAINING LOOP
pop = Population(population_size=population_size, mutation_rate=mutation_rate, rows=rows, columns=columns)
for generation in range(3000):
    pop.reproduction()
    pop.mutate_the_kids()
    if generation % 500 == 0:
        best_member = pop.print_best_seed()[0]
        best_image = best_member.seed
        print(best_member.score())
        plt.imshow(best_image, cmap='gray')
        plt.title(f"Best Seed - Generation {generation}")
        plt.show()
