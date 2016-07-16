
import random

import array

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from Game import Game
from agents.agent_tools.utils import map_state_to_inputs


class LookAhead:

    def __init__(self, actions, lookahead=4, mutation_prob=0.25, crossover_prob=0.5, n_steps=20, pop_size=50,
                 discounted=0.70):
        self.mxprob = mutation_prob
        self.pop_size = pop_size
        self.cxprob = crossover_prob
        self.ngen = n_steps
        self.actions = actions
        self.env = None
        self.value_function = None
        self.discounted = discounted

        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        # Randomly generate chromosome
        self.toolbox.register("attr_int", random.randint, 0, len(actions)-1)

        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, lookahead)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.reward)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_action, indpb=mutation_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def find_best(self, board, value_function):
        self.value_function = value_function
        self.env = board
        pop, log, hof = self.find_solution()
        if int(hof[0].fitness.values[0]) != -2048:
            return hof[0]
        else:
            return None

    def mutate_action(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                others = tuple(x for x in self.actions if x is not individual[i])
                individual[i] = type(individual[i])(random.choice(others))
        return individual,

    def reward(self, individual):
        game = Game(game_board=self.env, spawning=False)
        intuitive_reward = 0
        cnt = 0
        memory_reward = 0
        for action in individual:
            if action in game.get_illegal_actions():
                return -2048, memory_reward
            action_values = self.value_function(map_state_to_inputs(game.get_state()[0]))
            predicted_reward = game.do_action(action) * (self.discounted ** cnt)
            if game.game_over():
                return -2048, -2048
            intuitive_reward += predicted_reward
            memory_reward += action_values[action] * (self.discounted ** cnt)
            cnt += 1

        return intuitive_reward, memory_reward

    def find_solution(self):
        random.seed()
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxprob, mutpb=self.mxprob, ngen=self.ngen,
                                       halloffame=hof, verbose=False, stats=stats)
        return pop, log, hof
