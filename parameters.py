"""List of various parameters used in the algorithm. """

num_generations = 300
population_size = 500
num_inputs = 8
num_outputs = 4

weight_init_std = 2
weight_mutate_std = 0.5
weight_mutate_rate= 0.8
weight_replace_rate= 0.1

bias_init_std = 1
bias_mutate_std = 0.5
bias_mutate_rate = 0.7
bias_replace_rate = 0.1

new_link_rate = 0.5
del_link_rate = 0.5

new_node_rate = 0.2
del_node_rate = 0.2


# During crossover, probability of re-enabling gene if both parents are disabled.
p_enable_if_both_parents_disabled = 0.25

# Mutating weights: There is genome_weight_mutate probability of a
# genome having all of its weights mutated. In which case,
# each weight has p_perturb probability of being perturbed and
# (1 - p_perturb) probability of being assigned a new random value.
# p_weight_mutate = 0.8
# p_perturb = 0.9
# Maximum amount of perturbation
# max_perturb = 0.2







#Speciation
c1 = 1.0
c2 = 1.0
c3 = 0.4
delta_threshold = 1.75

stagnate_threshold = 35 #float("inf") # 20


#Keep the top fraction of the species for the next generation.
keep_best_amount = 0.25
