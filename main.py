import torch
import torchvision.transforms as transforms
from torchsummary import summary
import robust_neural_net
import genetic_algorithm_for_cnn as ga
import numpy as np
import random
import matplotlib.pyplot as plt


#### INITIALIZE GENETIC ALGORITHM PARAMETERS #####

rows = 28
columns = 28
population_size = 40
mutation_rate = 0.0005

######TRAINING LOOP######
pop = ga.population(population_size=population_size, mutation_rate=mutation_rate, rows=rows, columns=columns)
for generation in range(1400):
    pop.reproduction()
    pop.mutate_the_kids()
    print('generation done')
    if generation % 200 == 0:
        best_member = pop.print_best_seed()[0]
        best_image = best_member.seed
        print(best_member.score())
        # Plot the best image
        plt.imshow(best_image, cmap='gray')
        plt.title(f"Best Seed - Generation {generation}")
        plt.show()

