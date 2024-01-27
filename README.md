# GA_Mnist_generator
Trying to get a genetic algorithm to generate an MNIST like image of the number 7 using a fitness function that is based on feedback from an MNIST digit classifier. 

each member of my population is essentially a 28x28 binary matrix, which I then feed into a CNN MNIST digit classifier. 
what I'm essentially doing is defining the GA's fitness function based on how likely the genetic algorithm is to recognize that genome member as the number 7. 

