import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt
import time

np.random.seed(0)

sa_list_fitness, rhc_list_fitness, ga_list_fitness, mimic_list_fitness = [],[],[],[]
sa_list_time, rhc_list_time, ga_list_time, mimic_list_time =[],[],[],[]

value=range(1,201,15)
classifier_accuracy=np.zeros(4)
time_to_train=np.zeros(4)
len=40
attempts=10
iterations=500

for v in value:
	problem=mlrose_hiive.DiscreteOpt(length=v, fitness_fn=mlrose_hiive.FlipFlop())
	problem.set_mimic_fast_mode(True)

	start_time=time.time()
	_, sa, _=mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(), max_attempts=10, max_iters=1000, init_state=np.random.randint(2, size=v), curve=True)
	end_time=time.time()
	sa_runtime=end_time-start_time
	sa_list_fitness.append(sa)
	sa_list_time.append(sa_runtime)



	start_time=time.time()
	_, rhc, _=mlrose_hiive.random_hill_climb(problem, max_attempts=10, max_iters=1000, init_state=np.random.randint(2, size=v), curve=True)
	end_time=time.time()
	rhc_runtime=end_time-start_time
	rhc_list_fitness.append(rhc)
	rhc_list_time.append(rhc_runtime)


	start_time=time.time()
	_, ga, _=mlrose_hiive.genetic_alg(problem, max_attempts=10, curve=True)
	end_time=time.time()
	ga_runtime=end_time-start_time
	ga_list_fitness.append(ga)
	ga_list_time.append(ga_runtime)

	start_time=time.time()
	_, mimic, _=mlrose_hiive.mimic(problem, pop_size=300, max_attempts=10, curve=True)
	end_time=time.time()
	mimic_runtime=end_time-start_time
	mimic_list_fitness.append(mimic)
	mimic_list_time.append(mimic_runtime)


sa_list_fitness = np.asarray(sa_list_fitness)
rhc_list_fitness = np.asarray(rhc_list_fitness)
ga_list_fitness = np.asarray(ga_list_fitness)
mimic_list_fitness = np.asarray(mimic_list_fitness)
sa_list_time=np.asarray(sa_list_time)
rhc_list_time=np.asarray(rhc_list_time)
ga_list_time=np.asarray(ga_list_time)
mimic_list_time=np.asarray(mimic_list_time)

plt.figure()
plt.plot(value, sa_list_fitness, label='Simulated Annealing')
plt.plot(value, rhc_list_fitness, label='Randomized Hill Climb')
plt.plot(value, ga_list_fitness, label='Genetic Algorithm')
plt.plot(value, mimic_list_fitness, label='MIMIC')
plt.title('Flip Flop: Problem Size Analysis')
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend()
plt.grid()
plt.savefig('flip_flop_fitness.png')

plt.figure()
plt.plot(value, sa_list_time, label='Simulated Annealing')
plt.plot(value, rhc_list_time, label='Randomized Hill Climb')
plt.plot(value, ga_list_time, label='Genetic Algorithm')
plt.plot(value, mimic_list_time, label='MIMIC')
plt.title('Flip Flop: Time Analysis ')
plt.legend()
plt.xlabel('Problem Size')
plt.ylabel('Time (seconds)')
plt.grid()
plt.savefig('flip_flop_computation.png')

problem=mlrose_hiive.DiscreteOpt(length=250, fitness_fn=mlrose_hiive.FlipFlop(), maximize=True, max_val=2)
problem.set_mimic_fast_mode(True)
_,_, fitness_curve_sa=mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(), max_iters=250, init_state=np.random.randint(2, size=250), curve=True)
_,_, fitness_curve_rhc=mlrose_hiive.random_hill_climb(problem, max_attempts=10, max_iters=250, init_state=np.random.randint(2, size=250), curve=True)
_,_, fitness_curve_ga=mlrose_hiive.genetic_alg(problem, curve=True)
_,_, fitness_curve_mimic=mlrose_hiive.mimic(problem, pop_size=300, curve=True)

plt.figure()
plt.plot(fitness_curve_sa[:,0], label='Simulated Annealing')
plt.plot(fitness_curve_rhc[:,0], label='Randomized Hill Climb')
plt.plot(fitness_curve_ga[:,0], label='Genetic Algorithm')
plt.plot(fitness_curve_mimic[:,0], label='MIMIC')
plt.title('Flip Flop: Fitness Analysis')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.savefig('flip_flop_iterations.png')

problem=mlrose_hiive.DiscreteOpt(length=len, fitness_fn=mlrose_hiive.FlipFlop(), maximize=True)
problem.set_mimic_fast_mode(True)
state=np.random.randint(2, size=len)

_,_, ExpDecay=mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ExpDecay(), max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)
_,_, GeomDecay=mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.GeomDecay(), max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)
_,_, ArithDecay=mlrose_hiive.simulated_annealing(problem, schedule=mlrose_hiive.ArithDecay(), max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)

plt.figure()
plt.plot(ExpDecay[:,0], label= 'Exponential Decay')
plt.plot(GeomDecay[:,0], label='Geometric Decay')
plt.plot(ArithDecay[:,0], label= 'Arithmetic Decay')
plt.title('Flip Flop: Simulated Annealing')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.savefig('flip_flop_sa.png')


_,_, rhc_0=mlrose_hiive.random_hill_climb(problem, restarts=0, max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)
_,_, rhc_5=mlrose_hiive.random_hill_climb(problem, restarts=5, max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)
_,_, rhc_10=mlrose_hiive.random_hill_climb(problem, restarts=10, max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)
_,_, rhc_20=mlrose_hiive.random_hill_climb(problem, restarts=20, max_iters=iterations, init_state=np.random.randint(2, size=250), curve=True)

plt.figure()
plt.plot(rhc_0[:,0], label='0 Restarts')
plt.plot(rhc_5[:,0], label='5 Restarts')
plt.plot(rhc_10[:,0], label='10 Restarts')
plt.plot(rhc_20[:,0], label='20 Restarts')
plt.title('Flip Flop: Randomized Hill Climbing')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.savefig('flip_flop_rhc.png')

_,_, mimic_1=mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=50, curve=True)
_,_, mimic_2=mlrose_hiive.mimic(problem, keep_pct=0.3, pop_size=50, curve=True)
_,_, mimic_3=mlrose_hiive.mimic(problem, keep_pct=0.5, pop_size=50, curve=True)

_,_, mimic_4=mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=250, curve=True)
_,_, mimic_5=mlrose_hiive.mimic(problem, keep_pct=0.3, pop_size=250, curve=True)
_,_, mimic_6=mlrose_hiive.mimic(problem, keep_pct=0.5, pop_size=250, curve=True)

_,_, mimic_7=mlrose_hiive.mimic(problem, keep_pct=0.1, pop_size=500, curve=True)
_,_, mimic_8=mlrose_hiive.mimic(problem, keep_pct=0.3, pop_size=500, curve=True)
_,_, mimic_9=mlrose_hiive.mimic(problem, keep_pct=0.5, pop_size=500, curve=True)

plt.figure()
plt.plot(mimic_1[:,0], label='Keep=0.1, Pop=50')
plt.plot(mimic_2[:,0], label='Keep=0.3, Pop=50')
plt.plot(mimic_3[:,0], label='Keep=0.5, Pop=50')
plt.plot(mimic_4[:,0], label='Keep=0.1, Pop=250')
plt.plot(mimic_5[:,0], label='Keep=0.3, Pop=250')
plt.plot(mimic_6[:,0], label='Keep=0.5, Pop=250')
plt.plot(mimic_7[:,0], label='Keep=0.1, Pop=500')
plt.plot(mimic_8[:,0], label='Keep=0.3, Pop=500')
plt.plot(mimic_9[:,0], label='Keep=0.5, Pop=500')
plt.title('Flip Flop: MIMIC')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.savefig('flip_flop_mimic.png')

_,_, ga_1=mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=50, curve=True)
_,_, ga_2=mlrose_hiive.genetic_alg(problem, mutation_prob=0.3, pop_size=50, curve=True)
_,_, ga_3=mlrose_hiive.genetic_alg(problem, mutation_prob=0.5, pop_size=50, curve=True)

_,_, ga_4=mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=250, curve=True)
_,_, ga_5=mlrose_hiive.genetic_alg(problem, mutation_prob=0.3, pop_size=250, curve=True)
_,_, ga_6=mlrose_hiive.genetic_alg(problem, mutation_prob=0.5, pop_size=250, curve=True)

_,_, ga_7=mlrose_hiive.genetic_alg(problem, mutation_prob=0.1, pop_size=500, curve=True)
_,_, ga_8=mlrose_hiive.genetic_alg(problem, mutation_prob=0.3, pop_size=500, curve=True)
_,_, ga_9=mlrose_hiive.genetic_alg(problem, mutation_prob=0.5, pop_size=500, curve=True)

plt.figure()
plt.plot(ga_1[:,0], label='Mutation rate=0.1, Pop=50')
plt.plot(ga_2[:,0], label='Mutation rate=0.3, Pop=50')
plt.plot(ga_3[:,0], label='Mutation rate=0.5, Pop=50')
plt.plot(ga_4[:,0], label='Mutation rate=0.1, Pop=250')
plt.plot(ga_5[:,0], label='Mutation rate=0.3, Pop=250')
plt.plot(ga_6[:,0], label='Mutation rate=0.5, Pop=250')
plt.plot(ga_7[:,0], label='Mutation rate=0.1, Pop=500')
plt.plot(ga_8[:,0], label='Mutation rate=0.3, Pop=500')
plt.plot(ga_9[:,0], label='Mutation rate=0.5, Pop=500')
plt.title('Flip Flop: Genetic Algorithm')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.grid()
plt.savefig('flip_flop_ga.png')