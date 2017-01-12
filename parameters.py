"""List of various parameters used in the algorithm. """

# Number of inputs
num_inputs = 1
# Number of outputs
num_outputs = 1
# St. dv. of initial weight distribution (mean 0)
init_weight_std = 1
# During crossover, probability of re-enabling gene if both parents are disabled.
p_enable_if_both_parents_disabled = 0.25

# Mutating weights: There is genome_weight_mutate probability of a
# genome having all of its weights mutated. In which case,
# each weight has p_perturb probability of being perturbed and
# (1 - p_perturb) probability of being assiend a new random value.
genome_weight_mutate = 0.8
p_perturb = 0.9
# Maximum amount of perturbation
max_perturb = 0.1


population_size = 150


#Speciation
c1 = 1.0
c2 = 1.0
c3 = 0.4
delta_threshold = 3


