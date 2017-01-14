"""List of various parameters used in the algorithm. """

num_generations = 100


# Number of inputs
num_inputs = 8
# Number of outputs
num_outputs = 4
# St. dv. of initial weight distribution (mean 0)
init_weight_std = 2
# During crossover, probability of re-enabling gene if both parents are disabled.
p_enable_if_both_parents_disabled = 0.25

# Mutating weights: There is genome_weight_mutate probability of a
# genome having all of its weights mutated. In which case,
# each weight has p_perturb probability of being perturbed and
# (1 - p_perturb) probability of being assigned a new random value.
p_weight_mutate = 0.8
p_perturb = 0.9
# Maximum amount of perturbation
max_perturb = 0.2

p_new_link = 0.3
p_new_node = 0.03




population_size = 250


#Speciation
c1 = 1.0
c2 = 1.0
c3 = 0.4
delta_threshold = 3

stagnate_threshold = 20 #float("inf")


#Keep the top fraction of the species for the next generation.
keep_best_amount = 0.1
