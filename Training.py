import multiprocessing as mp

import AutoTrain
from agents.DiscreteAgent import DiscreteAgent

numProcs = 5
procs = [mp.Process(target=AutoTrain.main, args=(DiscreteAgent, True,)) for _ in range(numProcs)]
for p in procs:
    p.start()

[p.join() for p in procs]
