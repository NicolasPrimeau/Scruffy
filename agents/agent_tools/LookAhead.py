import array
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from Game import Game


class LookAhead:

    def __init__(self, actions, lookahead=5, mutation_prob=0.25, crossover_prob=0.5, n_steps=30, pop_size=40,
                 discounted=0.9):
        self.mxprob = mutation_prob
        self.pop_size = pop_size
        self.cxprob = crossover_prob
        self.ngen = n_steps
        self.actions = actions
        self.game_player = None
        self.discounted = discounted

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_int", random.randint, 0, len(actions)-1)

        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_int, lookahead)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.reward)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.mutate_action, indpb=mutation_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def find_best(self, game):
        self.game_player = game
        pop, log, hof = self.find_solution()
        if int(hof[0].fitness.values[0]) != -2048:
            return hof[0]
        else:
            return None

    def mutate_action(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = type(individual[i])(random.choice(self.actions))
        return individual,

    def reward(self, individual):
        game = Game(game_board=self.game_player.copy_gameboard(), spawning=False)
        reward = 0
        cnt = 0
        for action in individual:
            if action in game.get_illegal_actions():
                return -2048,
            this_reward = game.do_action(action) * (self.discounted ** cnt)
            if this_reward < 0:
                return -2048,
            reward += this_reward
        return reward,

    def find_solution(self):
        random.seed(64)
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", max)
        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxprob, mutpb=self.mxprob, ngen=self.ngen,
                                       halloffame=hof, verbose=False, stats=stats)
        return pop, log, hof
