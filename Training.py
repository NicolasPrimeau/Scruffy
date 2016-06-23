import multiprocessing as mp

import AutoTrain
from agents.DiscreteAgent import DiscreteAgent
from agents.NeuralNetAgent import NeuralNetAgent

numProcs = 1
procs = [mp.Process(target=AutoTrain.main, args=(NeuralNetAgent, False,)) for _ in range(numProcs)]
for p in procs:
    p.start()

[p.join() for p in procs]
