import pickle
import gzip
from glob import glob
import numpy as np

from mpi4py import MPI

worldComm = MPI.COMM_WORLD
worldSize = worldComm.Get_size()
worldRank = worldComm.Get_rank()

import sys
import os

from itertools import combinations
from collections import Counter
import networkx as nx

nDegreeBins = 20
nAgentStrengthBins = 20
selected = str(sys.argv[1])

minimumDegree = 8
minimumStrength = 8
sampleFrac = .05

selected = str(sys.argv[1])
flattenEventsList = False

dataRoot = "/home/ubi/urns_serie"

if selected == "TWT":
    inputDir = os.path.join(dataRoot, "twitter/twitter/data-01-09/*")
    gzipped = False
    lineSplitter = lambda l: list(combinations([int(e) for e in l.strip().split()], 2))
    flattenEventsList = True
elif selected == "APS":
    inputDir = os.path.join(dataRoot, "APS/aff_data_ISI_original_divided_per_month_1960_2006/*")
    gzipped = False
    lineSplitter = lambda l: list(combinations([int(e) for e in l.strip().split()], 2))
    flattenEventsList = True
elif selected == "APS_samples":
    inputDir = os.path.join(dataRoot, "APS/subsamples/n_links_01/sample_00/*")
    minimumDegree = 4
    minimumStrength = 5
    sampleFrac = .95
    gzipped = False
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
    #flattenEventsList = True
elif selected == "URNS_TWT":
    inputDir = os.path.join(dataRoot,
            "data_old/Symm_SonsExchg0_StrctSmpl1_r05_n05_t000005000000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
elif selected == "URNS_APS":
    inputDir = os.path.join(dataRoot,
            "data_analyzed/Symm_SonsExchg1_StrctSmpl2_r06_n15_t000000500000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
elif selected == "MPC":
    inputDir = os.path.join(dataRoot, "tel/data/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[1:3]]
elif selected == "URNS_MPC":
    minimumDegree = 5
    minimumStrength = 5
    sampleFrac = .75

    inputDir = os.path.join(dataRoot,
        #"data_analyzed/Symm_SonsExchg1_StrctSmpl1_r21_n07_t000050000000_Run_00/*")
        "data_analyzed/Symm_SonsExchg1_StrctSmpl1_r13_n04_t000050000000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
elif selected == "URNS_PROVA":
    inputDir = os.path.join(dataRoot,
            "data_analyzed/Symm_SonsExchg1_StrctSmpl0_r10_n05_t000001000000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
else:
    raise(RuntimeError, "Case %s unknown!" % selected)



###############################################################
###############################################################

def readFilesInChuncks(fname, opencmd, linesplitter, flatten, chunkSize=10000000, filemode="rb"):
    with opencmd(fname, filemode) as fIn:
        tmp_chunk = fIn.readlines(chunkSize)
        while tmp_chunk:
            if flattenEventsList:
                yield [e for l in tmp_chunk for e in lineSplitter(l)]
            else:
                yield [lineSplitter(l) for l in tmp_chunk]
            tmp_chunk = fIn.readlines(chunkSize)


if worldRank == 0:
    # Group agents/edges by degree/strength
    agentStrength = Counter()
    linkStrength = Counter()

    # Load the sequence and then select nodes by their strength and compute the heaps
    # constant and exponent accordingly for each bin
    eveCounter = 0
    print(inputDir)
    for f in sorted(glob(inputDir)):
        apri = gzip.open if gzipped else open
        for tmp_events in readFilesInChuncks(f, apri, lineSplitter, flatten=flattenEventsList):
            tmp_events = [tuple(sorted(e)) for e in tmp_events if e[0] != e[1]]
            agentStrength.update([a for e in tmp_events for a in e])
            linkStrength.update(tmp_events)
            eveCounter += len(tmp_events)
        print(f, eveCounter)
    agentDegree = Counter([i for e in linkStrength.keys() for i in e])

    print("Sequence loaded")

    # Group agents by degree
    validNodes = set([n for n, s in agentStrength.items() if s >= minimumStrength
                                                              and agentDegree[n] >= minimumDegree
                                                              and np.random.rand() < sampleFrac])
    mainSubsequence = []
    for f in sorted(glob(inputDir)):
        apri = gzip.open if gzipped else open
        for tmp_events in readFilesInChuncks(f, apri, lineSplitter, flatten=flattenEventsList):
            tmp_events = [tuple(sorted(e)) for e in tmp_events if e[0] != e[1]]
            mainSubsequence.extend([e for e in tmp_events
                                        if e[0] in validNodes or e[1] in validNodes])
            eveCounter += len(tmp_events)
        print(f, eveCounter)
    workload = list(sorted(validNodes))
    # Bin the agents by their degree and the edges by their strength.
    # We also annotate once which nodes/links are in each bin.
    minDeg = minimumDegree
    degreeBins = np.logspace(np.log(minDeg), np.log10(max(agentDegree.values())+1), nDegreeBins)
    agentDegreeBin = {i: np.argmax(degreeBins >= k) for i, k in agentDegree.items()
                                                        if i in validNodes}

    minAgentStrength = minimumStrength
    agentStrengthBins = np.logspace(np.log(minAgentStrength), np.log10(max(agentStrength.values())+1),
                                        nAgentStrengthBins)
    agentStrengthBin = {i: np.argmax(agentStrengthBins >= k) for i, k in agentStrength.items()
                                                                if i in validNodes}
    workload = sorted(list(validNodes))
    for node in range(1, worldSize):
        print("Slicing for node %d..." % node,)
        nodesToDo = set(workload[node::worldSize])
        subSequence = [e for e in mainSubsequence if e[0] in nodesToDo or e[1] in nodesToDo]
        worldComm.send(nodesToDo, dest=node, tag=5)
        worldComm.send(subSequence, dest=node, tag=6)
        print("done!")
    nodesToDo = set(workload[::worldSize])
    subSequence = [e for e in mainSubsequence if e[0] in nodesToDo or e[1] in nodesToDo]
else:
    nodesToDo = worldComm.recv(source=0, tag=5)
    subSequence = worldComm.recv(source=0, tag=6)
    agentDegreeBin, agentStrengthBin = None, None

agentDegreeBin = worldComm.bcast(agentDegreeBin, root=0)
agentStrengthBin = worldComm.bcast(agentStrengthBin, root=0)


# The fraction of agents to sample from each bin...
print("Doing the heaps per node sequence...")
heapsParsPerDegreeBin = {k: [] for k in range(nDegreeBins)}
heapsParsPerStrengthBin = {k: [] for k in range(nAgentStrengthBins)}
heapsParsPerAgent = {i: [] for i in nodesToDo}
heapsSeqPerAgent = {i: [] for i in nodesToDo}

from scipy.optimize import curve_fit

def heaps_fit_foo(x, a, b):
    return a * x**b

def heaps_fit_foo(x, a, b):
    return (1 + a*x)**b

iii = 0
for agent in nodesToDo:
    # For each agent select the subsequence
    subSequenceAgent = [e for e in subSequence if agent in e]

    cumulativeNeighboors = set()
    originalBinarySequence = []
    for eve in subSequenceAgent:
        # Put the agent with focus as i
        i, j = eve[0], eve[1]
        if j == agent:
            j = i
            i = agent
        if j not in cumulativeNeighboors:
            cumulativeNeighboors.add(j)
        originalBinarySequence.append(len(cumulativeNeighboors))

    XXXs = np.arange(len(originalBinarySequence))
    YYYs = np.array(originalBinarySequence)

    heapsSeqPerAgent[agent] = YYYs

    res, cov = curve_fit(heaps_fit_foo, XXXs, YYYs,
                             p0=[1., .1],
                             bounds=((1e-4, 1e-2), (1e4, 1.5)))

    degBin = agentDegreeBin[agent]
    strBin = agentStrengthBin[agent]

    heapsParsPerDegreeBin[degBin].append(res)
    heapsParsPerStrengthBin[strBin].append(res)
    heapsParsPerAgent[agent].append(res)

    if worldRank == 0:
        iii += 1
        sys.stdout.write("\r%05d / %05d" % (iii, len(nodesToDo)))
        sys.stdout.flush()

if worldRank == 0:
    print("\nEverything done, collecting and saving...")
    for i in range(1, worldSize):
        tmp_perDeg = worldComm.recv(source=i, tag=10)
        for k in heapsParsPerDegreeBin.keys():
            heapsParsPerDegreeBin[k].extend(tmp_perDeg[k])

        tmp_perStr = worldComm.recv(source=i, tag=11)
        for k in heapsParsPerStrengthBin.keys():
            heapsParsPerStrengthBin[k].extend(tmp_perStr[k])

        tmp_perAgn = worldComm.recv(source=i, tag=12)
        heapsParsPerAgent.update(tmp_perAgn)

        tmp_perAgn = worldComm.recv(source=i, tag=13)
        heapsSeqPerAgent.update(tmp_perAgn)

    totalResultsEntropy = {
        "degreeBins": degreeBins, "agentStrengthBins": agentStrengthBins,
        "validNodes": validNodes, "agentDegree": agentDegree, "agentStrength": agentStrength,
        "heapsParsPerDegreeBin": heapsParsPerDegreeBin,
        "heapsParsPerStrengthBin": heapsParsPerStrengthBin,
        "heapsParsPerAgent": heapsParsPerAgent,
        "heapsSeqPerAgent": heapsSeqPerAgent,  "name": selected,
    }

    pickle.dump(totalResultsEntropy, open("heapsParsSequence_%s.pkl" % selected, "wb"))

else:
    worldComm.send(heapsParsPerDegreeBin, dest=0, tag=10)
    worldComm.send(heapsParsPerStrengthBin, dest=0, tag=11)
    worldComm.send(heapsParsPerAgent, dest=0, tag=12)
    worldComm.send(heapsSeqPerAgent, dest=0, tag=13)


