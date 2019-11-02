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
    inputDir = os.path.join(dataRoot,
            "data_analyzed/Symm_SonsExchg1_StrctSmpl1_r21_n07_t000050000000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
elif selected == "URNS_PROVA":
    inputDir = os.path.join(dataRoot,
            "data_analyzed/Symm_SonsExchg1_StrctSmpl0_r10_n05_t000001000000_Run_00/*")
    gzipped = True
    lineSplitter = lambda l: [int(e) for e in l.strip().split()[:2]]
else:
    raise RuntimeError, "Case %s unknown!" % selected


# Bin the agents by their degree and the edges by their strength.
# We also annotate once which nodes/links are in each bin.
nDegreeBins = 25
minDeg = 2
maxDeg = 10000
maxEventsPerNode = 100000
maxTimePerNode = 1000000

# The number of bins for nodes strength (total number of events in which the node
# is seen) and the number of bins for the edges
nStrengthBins = 25
minStr = 2

nLinkStrengthBins = 25
minLinkStrength = 2

fractionNodes = .01
fractionEdges = .001

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

    # Load the sequence
    eveCounter = 0
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

    degreeBins = np.logspace(np.log(minDeg), np.log10(max(agentDegree.values())+1), nDegreeBins)
    agentDegreeBin = {i: np.argmax(degreeBins >= k) for i, k in agentDegree.iteritems() if k>= minDeg}
    agentsInDegreeBin = {k: set(i for i, db in agentDegreeBin.iteritems() if db == k)
                                for k in range(nDegreeBins)}

    linkStrengthBins = np.logspace(np.log(minLinkStrength),
                                   np.log10(max(linkStrength.values())+1),
                                   nLinkStrengthBins)
    linkStrengthBin = {i: np.argmax(linkStrengthBins >= k)
                            for i, k in linkStrength.iteritems() if k>= minLinkStrength}
    linksInStrengthBin = {k: set(i for i, db in linkStrengthBin.iteritems() if db == k)
                            for k in range(nLinkStrengthBins)}


    strengthBins = np.logspace(np.log(minStr),
                               np.log10(max(agentStrength.values())+1),
                               nStrengthBins)
    agentStrengthBin = {i: np.argmax(strengthBins >= k)
                            for i, k in agentStrength.iteritems() if k>= minStr}
    agentsInStrengthBin = {k: set(i for i, db in agentStrengthBin.iteritems() if db == k)
                            for k in range(nStrengthBins)}

    # Now sample some agents and edges based on degree/strength
    sampledDegreeAgents = {}
    for db, candidates in agentsInDegreeBin.iteritems():
        sample = [i for i in candidates if np.random.rand() < fractionNodes]
        if len(sample) == 0 and len(candidates) > 0:
            # Maximum 50 candidates if we got none before
            tmp_indexes = np.arange(len(candidates))
            np.random.shuffle(tmp_indexes)
            candList = list(candidates)
            sample = [candList[i] for i in tmp_indexes[:min(len(tmp_indexes), 50)]]
        if len(sample) == 0:
            continue
        np.random.shuffle(sample)
        sampledDegreeAgents[db] = sample

    sampledStrengthAgents = {}
    for db, candidates in agentsInStrengthBin.iteritems():
        sample = [i for i in candidates if np.random.rand() < fractionNodes]
        if len(sample) == 0 and len(candidates) > 0:
            # Maximum 50 candidates if we got none before
            tmp_indexes = np.arange(len(candidates))
            np.random.shuffle(tmp_indexes)
            candList = list(candidates)
            sample = [candList[i] for i in tmp_indexes[:min(len(tmp_indexes), 50)]]
        if len(sample) == 0:
            continue
        np.random.shuffle(sample)
        sampledStrengthAgents[db] = sample

    sampledStrengthEdges = {}
    for db, candidates in linksInStrengthBin.iteritems():
        sample = [i for i in candidates if np.random.rand() < fractionEdges]
        if len(sample) == 0 and len(candidates) > 0:
            # Maximum 50 candidates if we got none before
            tmp_indexes = np.arange(len(candidates))
            np.random.shuffle(tmp_indexes)
            candList = list(candidates)
            sample = [tuple(sorted(candList[i]))
                         for i in tmp_indexes[:min(len(tmp_indexes), 50)]]
        if len(sample) == 0:
            continue
        np.random.shuffle(sample)
        sampledStrengthEdges[db] = sample

    # Now cast the samples to do to each node...
    for dest in range(1, worldSize):
        worldComm.send({k: v[dest::worldSize] for k, v in sampledDegreeAgents.iteritems()},
                        dest=dest, tag=1)
        worldComm.send({k: v[dest::worldSize] for k, v in sampledStrengthAgents.iteritems()},
                        dest=dest, tag=2)
        worldComm.send({k: v[dest::worldSize] for k, v in sampledStrengthEdges.iteritems()},
                        dest=dest, tag=3)
        worldComm.send(degreeBins, dest=dest, tag=4)
        worldComm.send(strengthBins, dest=dest, tag=5)
        worldComm.send(linkStrengthBins, dest=dest, tag=6)

    sampledDegreeAgentsBatch = {k: v[worldRank::worldSize] for k, v in sampledDegreeAgents.iteritems()}
    sampledStrengthAgentsBatch = {k: v[worldRank::worldSize] for k, v in sampledStrengthAgents.iteritems()}
    sampledStrengthEdgesBatch = {k: v[worldRank::worldSize] for k, v in sampledStrengthEdges.iteritems()}
else:
    sampledDegreeAgentsBatch = worldComm.recv(source=0, tag=1)
    sampledStrengthAgentsBatch = worldComm.recv(source=0, tag=2)
    sampledStrengthEdgesBatch = worldComm.recv(source=0, tag=3)
    degreeBins = worldComm.recv(source=0, tag=4)
    strengthBins = worldComm.recv(source=0, tag=5)
    linkStrengthBins = worldComm.recv(source=0, tag=6)



nodesToSample = set()
for db, nodes in sampledDegreeAgentsBatch.iteritems():
    nodesToSample.update(nodes)
for sb, nodes in sampledStrengthAgentsBatch.iteritems():
    nodesToSample.update(nodes)

edgesToSample = set()
for sb, edges in sampledStrengthEdgesBatch.iteritems():
    edgesToSample.update(edges)

# For each node:
# - `s` sequence of events new/old link
# - `t` sequence of node in events (total sequece)
# - `n` sequence per neighbors = {neighb: [t_0, t_1, ...]}
# - `c` counter of events of node...

if worldRank == 0:
    print("Nodes - edges to sample:", nodesToSample, edgesToSample)

resultNodes = {n: {"s": [], "t": [], "c": 0, "n": {}, } for n in nodesToSample}
resultEdges = {e: [] for e in edgesToSample}
eveCounter = 0
for f in sorted(glob(inputDir)):
    apri = gzip.open if gzipped else open
    for tmp_events in readFilesInChuncks(f, apri, lineSplitter, flatten=flattenEventsList):
        tmp_events = [tuple(sorted(e)) for e in tmp_events if e[0] != e[1]]
        for eve in tmp_events:
            for tmp_iii, tmp_jjj in zip(eve, eve[::-1]):
                # If iii is to sample we check the 
                try:
                    tmp_dict = resultNodes[tmp_iii]
                except KeyError:
                    continue

                tmp_count = tmp_dict["c"]
                # Node activation in total sequence...
                tmp_dict["t"].append(eveCounter)
                try:
                    # Edge activation in local sequence...
                    tmp_dict["n"][tmp_jjj].append(tmp_count)
                except KeyError:
                    tmp_dict["n"][tmp_jjj] = [tmp_count,]
                    # New edge activation in local sequence...
                    tmp_dict["s"].append(tmp_count)

                # Increment the node events counter
                tmp_dict["c"] += 1

            # The edge activation in the total sequence...
            try:
                resultEdges[eve].append(eveCounter)
            except KeyError:
                pass

            # Increment the total events counter
            eveCounter += 1
    if worldRank == 0:
        print(f, eveCounter)

# Now for each result that we have compute the entropy...
# We reconstruct the sequences by looking at the index when something happened.

entropyNewLink = {k: [] for k in range(nDegreeBins)}
entropyNewLinkShuf = {k: [] for k in range(nDegreeBins)}

interevent = [[] for i in range(nDegreeBins)]
intereventShuf = [[] for i in range(nDegreeBins)]

entropyPerLink = {k: [] for k in range(nLinkStrengthBins)}
entropyPerLinkShuf = {k: [] for k in range(nLinkStrengthBins)}

intereventPerLink = [[] for i in range(nLinkStrengthBins)]
intereventPerLinkShuf = [[] for i in range(nLinkStrengthBins)]

entropyNodeTot = {k: [] for k in range(nStrengthBins)}
entropyNodeTotShuf = {k: [] for k in range(nStrengthBins)}

intereventNodeTot = [[] for i in range(nStrengthBins)]
intereventNodeTotShuf = [[] for i in range(nStrengthBins)]

entropyPerLinkTot = {k: [] for k in range(nLinkStrengthBins)}
entropyPerLinkTotShuf = {k: [] for k in range(nLinkStrengthBins)}

intereventPerLinkTot = [[] for i in range(nLinkStrengthBins)]
intereventPerLinkTotShuf = [[] for i in range(nLinkStrengthBins)]


def writeOut(iii, tot):
    sys.stdout.write("\rNode %09d / %09d..." % (iii, tot))
    sys.stdout.flush()

def generateShuffledFromOnes(ones):
    '''
    Instead of sampling from the arange we directly sample random integer numbers
    until we have the selected amount of them...
    '''
    ones_arra = np.array(ones, dtype=int)
    t0, t1 = ones_arra.min(), ones_arra.max()
    numOfEve = t1 - t0
    numOfOnes = len(ones_arra)
    #timeEvents = ones_arra - t0
    #return np.sort(np.random.choice(np.linspace(0, numOfEve, numOfEve+1, dtype=int),
    #                        size=numOfOnes, replace=False))

    return np.sort(np.random.randint(0, numOfEve+1, numOfOnes, dtype=int))

def entropyFromOnesSmart(ones, shuffledSequence):
    ones_arra = np.array(ones)
    t0, t1 = ones_arra.min(), ones_arra.max()
    numOfEve = t1 - t0
    numOfOnes = float(len(ones_arra))
    timeEvents = ones_arra - t0

    splits = np.linspace(0, numOfEve+1, int(numOfOnes)+1, dtype=int)
    splits = np.unique(splits)
    splits.sort()
    #print splits
    S = Sshuffled = 0
    low_original_index = low_shuffled_index = 0
    for index in xrange(len(splits)-1):
        ini, fin = splits[index], splits[index+1]
        low_original_index += np.argmax(timeEvents[low_original_index:] >= ini)
        f = np.count_nonzero(np.logical_and(timeEvents[low_original_index:]>=ini,
                                            timeEvents[low_original_index:]<fin))
        
        low_shuffled_index += np.argmax(shuffledSequence[low_shuffled_index:] >= ini)
        fShuf = np.count_nonzero(np.logical_and(shuffledSequence[low_shuffled_index:]>=ini,
                                                shuffledSequence[low_shuffled_index:]<fin))
        if f > .0:
            dS = f/numOfOnes
            S -= dS*np.log(dS)
        if fShuf > 0:
            dS = fShuf/numOfOnes
            Sshuffled -= dS*np.log(dS)
    normS = np.log(numOfOnes)
    return S/normS, Sshuffled/normS


# Entropy of node sequence, new/old links...
if worldRank == 0:
    print("Doing the entropy per node with new/old links and neighbors...")
iii_count = 0
for node, nodeDict in resultNodes.iteritems():
    print("Proc %02d doing node %r 's'" % (worldRank, node))
    # The entropy and interevent on the sequence of old/new links...
    tmpDegNode = float(len(nodeDict["s"]))
    if tmpDegNode < minDeg or tmpDegNode > maxDeg:
        continue
    tmpDegBin = np.argmax(degreeBins >= tmpDegNode)

    tmp_ones = nodeDict["s"]
    if len(tmp_ones) > maxEventsPerNode or (max(tmp_ones) - min(tmp_ones)) > maxTimePerNode:
        continue

    shuffledSequence = generateShuffledFromOnes(tmp_ones)
    S, Sshuffled = entropyFromOnesSmart(ones=tmp_ones, shuffledSequence=shuffledSequence)
    entropyNewLink[tmpDegBin].append(S)
    entropyNewLinkShuf[tmpDegBin].append(Sshuffled)

    # The real interevents are simply the difference of the ones vectors...
    interevent[tmpDegBin].extend(list(np.diff(np.array(tmp_ones))))
    intereventShuf[tmpDegBin].extend(list(np.diff(np.array(shuffledSequence))))

    print("Proc %02d doing node %r 'n'" % (worldRank, node))
    # Now the entropy considering each neighbor sequence...
    # We save its entropy and stuff in the same deg bin as the node...
    for neigh, neighDict in nodeDict["n"].iteritems():
        tmp_edge = tuple(sorted([node, neigh]))
        if len(neighDict) < minLinkStrength:
            continue
        tmpEdgeStr = len(neighDict)
        tmpStrBin = np.argmax(linkStrengthBins >= tmpEdgeStr)

        tmp_ones = neighDict

        shuffledSequence = generateShuffledFromOnes(tmp_ones)
        S, Sshuffled = entropyFromOnesSmart(ones=tmp_ones, shuffledSequence=shuffledSequence)
        entropyPerLink[tmpStrBin].append(S)
        entropyPerLinkShuf[tmpStrBin].append(Sshuffled)

        intereventPerLink[tmpStrBin].extend(list(np.diff(np.array(tmp_ones))))
        intereventPerLinkShuf[tmpStrBin].extend(list(np.diff(np.array(shuffledSequence))))

    print("Proc %02d doing node %r 't'" % (worldRank, node))
    # The entropy and interevent on the node in event total sequence...
    tmpStrNode = float(len(nodeDict["t"]))
    if tmpStrNode < minStr:
        continue
    tmpStrBin = np.argmax(strengthBins >= tmpStrNode)

    # The entropy and interevent on the node in event total sequence...
    tmp_ones = nodeDict["t"]
    shuffledSequence = generateShuffledFromOnes(tmp_ones)
    S, Sshuffled = entropyFromOnesSmart(ones=tmp_ones, shuffledSequence=shuffledSequence)
    entropyNodeTot[tmpStrBin].append(S)
    entropyNodeTotShuf[tmpStrBin].append(Sshuffled)

    intereventNodeTot[tmpStrBin].extend(list(np.diff(np.array(tmp_ones))))
    intereventNodeTotShuf[tmpStrBin].extend(list(np.diff(np.array(shuffledSequence))))

    if worldRank == 0 and  iii_count % 100 == 0:
        writeOut(iii_count, len(resultNodes))
    iii_count += 1
if worldRank == 0:
    print("Done!")


# Entropy of node sequence, new/old links...
if worldRank == 0:
    print("Doing the entropy per edge in total sequence...")
iii_count = 0
for edge, edgeDict in resultEdges.iteritems():
    print("Proc %02d doing edge %r 't'" % (worldRank, edge))
    # The entropy and interevent on the sequence of old/new links...
    tmpStrEdge = float(len(edgeDict))
    if tmpStrEdge< minStr:
        continue
    tmpStrBin = np.argmax(linkStrengthBins>= tmpStrEdge)

    tmp_ones = edgeDict
    shuffledSequence = generateShuffledFromOnes(tmp_ones)
    S, Sshuffled = entropyFromOnesSmart(ones=tmp_ones, shuffledSequence=shuffledSequence)
    entropyPerLinkTot[tmpStrBin].append(S)
    entropyPerLinkTotShuf[tmpStrBin].append(Sshuffled)

    intereventPerLinkTot[tmpStrBin].extend(list(np.diff(np.array(tmp_ones))))
    intereventPerLinkTotShuf[tmpStrBin].extend(list(np.diff(np.array(shuffledSequence))))

    if worldRank == 0 and  iii_count % 100 == 0:
        writeOut(iii_count, len(resultEdges))
    iii_count += 1
if worldRank == 0:
    print("Done! Now collecting data...")
    for source in range(1, worldSize):
        for k, v in worldComm.recv(source=source, tag=0).iteritems():
            entropyNewLink[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=1).iteritems():
            entropyNewLinkShuf[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=2)):
            interevent[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=3)):
            intereventShuf[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=4).iteritems():
            entropyPerLink[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=5).iteritems():
            entropyPerLinkShuf[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=6)):
            intereventPerLink[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=7)):
            intereventPerLinkShuf[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=8).iteritems():
            entropyNodeTot[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=9).iteritems():
            entropyNodeTotShuf[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=10)):
            intereventNodeTot[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=11)):
            intereventNodeTotShuf[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=12).iteritems():
            entropyPerLinkTot[k].extend(v)
        for k, v in worldComm.recv(source=source, tag=13).iteritems():
            entropyPerLinkTotShuf[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=14)):
            intereventPerLinkTot[k].extend(v)
        for k, v in enumerate(worldComm.recv(source=source, tag=15)):
            intereventPerLinkTotShuf[k].extend(v)

    print("Done collecting, saving...")
    totalResultsEntropy = {
        "degreeBins": degreeBins, "linkStrengthBins": linkStrengthBins,
        "entropyNewLink": entropyNewLink, "entropyNewLinkShuf": entropyNewLinkShuf,
        "interevent": interevent, "intereventShuf": intereventShuf,
        "entropyPerLink": entropyPerLink, "entropyPerLinkShuf": entropyPerLinkShuf,
        "intereventPerLink": intereventPerLink, "intereventPerLinkShuf": intereventPerLinkShuf,
        "strengthBins": strengthBins, "linkStrengthBins": linkStrengthBins,
        "entropyNodeTot": entropyNodeTot, "entropyNodeTotShuf": entropyNodeTotShuf,
        "intereventNodeTot": intereventNodeTot, "intereventNodeTotShuf": intereventNodeTotShuf,
        "entropyPerLinkTot": entropyPerLinkTot, "entropyPerLinkTotShuf": entropyPerLinkTotShuf,
        "intereventPerLinkTot": intereventPerLinkTot, "intereventPerLinkTotShuf": intereventPerLinkTotShuf,
        "name": selected,
    }
    pickle.dump(totalResultsEntropy, open("entropySequence_MPI_%s.pkl" % selected, "wb"))
else:
    worldComm.send(entropyNewLink, dest=0, tag=0)
    worldComm.send(entropyNewLinkShuf, dest=0, tag=1)
    worldComm.send(interevent, dest=0, tag=2)
    worldComm.send(intereventShuf, dest=0, tag=3)
    worldComm.send(entropyPerLink, dest=0, tag=4)
    worldComm.send(entropyPerLinkShuf, dest=0, tag=5)
    worldComm.send(intereventPerLink, dest=0, tag=6)
    worldComm.send(intereventPerLinkShuf, dest=0, tag=7)
    worldComm.send(entropyNodeTot, dest=0, tag=8)
    worldComm.send(entropyNodeTotShuf, dest=0, tag=9)
    worldComm.send(intereventNodeTot, dest=0, tag=10)
    worldComm.send(intereventNodeTotShuf, dest=0, tag=11)
    worldComm.send(entropyPerLinkTot, dest=0, tag=12)
    worldComm.send(entropyPerLinkTotShuf, dest=0, tag=13)
    worldComm.send(intereventPerLinkTot, dest=0, tag=14)
    worldComm.send(intereventPerLinkTotShuf, dest=0, tag=15)




if False:

# Sample some agents for each degree bin and evaluate their sub-sequence
# Here we compute two kind of signals: the entropy of the sequence
# of all the events containing a node putting a one when the node
# contacts/is contacted by a new link and then a per-link entropy.
# In the latter we start from the same sequence as before but we
# put a one each time the selected link is active and a 0 otherwise.

# The fraction of agents to sample from each bin...
    frac = .01
    print("Doing the entropy per node sequence...")
    entropyNewLink = {k: [] for k in range(nDegreeBins)}
    entropyNewLinkShuf = {k: [] for k in range(nDegreeBins)}

    interevent = [[] for i in range(nDegreeBins)]
    intereventShuf = [[] for i in range(nDegreeBins)]

    entropyPerLink = {k: [] for k in range(nLinkStrengthBins)}
    entropyPerLinkShuf = {k: [] for k in range(nLinkStrengthBins)}

    intereventPerLink = [[] for i in range(nLinkStrengthBins)]
    intereventPerLinkShuf = [[] for i in range(nLinkStrengthBins)]

    for db, candidates in agentsInDegreeBin.iteritems():
        sample = [i for i in candidates if np.random.rand() < frac]
        if len(sample) == 0 and len(candidates) > 0:
            sample = list(candidates)
        if len(sample) == 0:
            continue
        sample = set(sample)

        mainSubsequence = [e for e in listone if e[0] in sample or e[1] in sample]
        # For each agent select the subsequence
        for agent in sample:
            subSequence = [e for e in mainSubsequence if agent in e]

            cumulativeNeighboors = set()
            originalBinarySequence = []
            for eve in subSequence:
                # Put the agent with focus as i
                i, j = eve[0], eve[1]
                if j == agent:
                    j = i
                    i = agent

                if j not in cumulativeNeighboors:
                    originalBinarySequence.append(1)
                    cumulativeNeighboors.add(j)
                else:
                    originalBinarySequence.append(0)
            # The degree of the agent...
            assert len(cumulativeNeighboors) == agentDegree[agent], "%d != %d" % (len(cumulativeNeighboors), agentDegree[agent])

            k, nEvents = agentDegree[agent], len(originalBinarySequence)

            originalBinarySequence = np.array(originalBinarySequence)
            shuffledLocalBinarySequence = np.array(originalBinarySequence)
            np.random.shuffle(shuffledLocalBinarySequence)

            splits = np.linspace(0, nEvents, k+1, dtype=int)
            splits = np.unique(splits)
            splits.sort()
            #print splits
            S = Sshuffled = 0
            for index in xrange(len(splits)-1):
                ini, fin = splits[index], splits[index+1]
                f = np.sum(originalBinarySequence[ini:fin])
                fShuf = np.sum(shuffledLocalBinarySequence[ini:fin])
                if f > .0:
                    dS = f/float(k)
                    S -= dS*np.log(dS)
                if fShuf > 0:
                    dS = fShuf/float(k)
                    Sshuffled -= dS*np.log(dS)

            for referenceSeq, targetAcc in zip(
                                    (originalBinarySequence, shuffledLocalBinarySequence),
                                    (interevent, intereventShuf)):
                interEve = 0
                first = True
                for eve in referenceSeq:
                    interEve += 1
                    if eve == 1:
                        if first:
                            first = False
                        else:
                            targetAcc[db].append(interEve)
                            interEve = 0
            
            entropyNewLink[db].append(S/np.log(k))
            entropyNewLinkShuf[db].append(Sshuffled/np.log(k))
            
            # Now the entropy considering each link per-se...
            for neighbor in cumulativeNeighboors:
                originalBinarySequence = []
                first = True
                for ev in subSequence:
                    if first and neighbor in ev:
                        first = False
                        originalBinarySequence.append(1)
                    else:
                        if neighbor in ev:
                            originalBinarySequence.append(1)
                        else:
                            originalBinarySequence.append(0)
                      
                k, nEvents = linkStrength[tuple(sorted([agent, neighbor]))], len(originalBinarySequence)
                assert k == sum(originalBinarySequence)
                
                if k < minLinkStrength:
                    continue
                tmp_linkStrengthBin = np.argmax(linkStrengthBins >= k)
                originalBinarySequence = np.array(originalBinarySequence)
                shuffledLocalBinarySequence = np.array(originalBinarySequence)
                np.random.shuffle(shuffledLocalBinarySequence)

                splits = np.linspace(0, nEvents, k+1, dtype=int)
                splits = np.unique(splits)
                splits.sort()
                #print splits
                S = Sshuffled = 0
                for index in xrange(len(splits)-1):
                    ini, fin = splits[index], splits[index+1]
                    f = np.sum(originalBinarySequence[ini:fin])
                    fShuf = np.sum(shuffledLocalBinarySequence[ini:fin])
                    if f > .0:
                        dS = f/float(k)
                        S -= dS*np.log(dS)
                    if fShuf > 0:
                        dS = fShuf/float(k)
                        Sshuffled -= dS*np.log(dS)

                for referenceSeq, targetAcc in zip(
                                        (originalBinarySequence, shuffledLocalBinarySequence),
                                        (intereventPerLink, intereventPerLinkShuf)):
                    interEve = 0
                    first = True
                    for eve in referenceSeq:
                        interEve += 1
                        if eve == 1:
                            if first:
                                first = False
                            else:
                                targetAcc[db].append(interEve)
                                interEve = 0

                entropyPerLink[tmp_linkStrengthBin].append(S/np.log(k))
                entropyPerLinkShuf[tmp_linkStrengthBin].append(Sshuffled/np.log(k))

        print db,

    print("Sequence per node done!")

# The number of bins for nodes strength (total number of events in which the node
# is seen) and the number of bins for the edges
    nStrengthBins = 25
    minStr = 2
    strengthBins = np.logspace(np.log(minStr), np.log10(max(agentStrength.values())+1), nStrengthBins)
    agentStrengthBin = {i: np.argmax(strengthBins >= k) for i, k in agentStrength.iteritems() if k>= minStr}
    agentsInStrengthBin = {k: set(i for i, db in agentStrengthBin.iteritems() if db == k) for k in range(nStrengthBins)}

    nLinkStrengthBins = 25
    minLinkStrength = 2
    linkStrengthBins = np.logspace(np.log(minLinkStrength), np.log10(max(linkStrength.values())+1), nLinkStrengthBins)
    linkStrengthBin = {i: np.argmax(linkStrengthBins >= k) for i, k in linkStrength.iteritems() if k>= minLinkStrength}
    linksInStrengthBin = {k: set(i for i, db in linkStrengthBin.iteritems() if db == k) for k in range(nLinkStrengthBins)}

    print("Doing node S on total seq...")

# Sample some agents for each strength bin and evaluate their sub-sequence
# Here we compute the entropy of the sequence of all the events starting
# with the first event containing the node putting a one when the node
# participate in the event and zero otherwise.

    frac = .01

    entropyNodeTot = {k: [] for k in range(nStrengthBins)}
    entropyNodeTotShuf = {k: [] for k in range(nStrengthBins)}

    intereventNodeTot = [[] for i in range(nStrengthBins)]
    intereventNodeTotShuf = [[] for i in range(nStrengthBins)]

    for db, candidates in agentsInStrengthBin.iteritems():
        sample = [i for i in candidates if np.random.rand() < frac]
        if len(sample) == 0 and len(candidates) > 0:
            sample = list(candidates)
        if len(sample) == 0:
            continue
        sample = set(sample)
        
        # For each agent select the subsequence
        for agent in sample:
            seqStart = 0
            for ev in listone:
                if agent in ev:
                    break
                else:
                    seqStart += 1
            subSequence = listone[seqStart:]
            
            originalBinarySequence = []
            for eve in subSequence:
                # Put the agent with focus as i
                if agent in eve:
                    originalBinarySequence.append(1)
                else:
                    originalBinarySequence.append(0)
                    
            # The degree of the agent...
            assert sum(originalBinarySequence) == agentStrength[agent], "%d != %d" % (sum(originalBinarySequence), agentStrength[agent])
            
            k, nEvents = agentStrength[agent], len(originalBinarySequence)
            
            originalBinarySequence = np.array(originalBinarySequence)
            shuffledLocalBinarySequence = np.array(originalBinarySequence)
            np.random.shuffle(shuffledLocalBinarySequence)
            
            splits = np.linspace(0, nEvents, k+1, dtype=int)
            splits = np.unique(splits)
            splits.sort()
            #print splits
            S = Sshuffled = 0
            for index in xrange(len(splits)-1):
                ini, fin = splits[index], splits[index+1]
                f = np.sum(originalBinarySequence[ini:fin])
                fShuf = np.sum(shuffledLocalBinarySequence[ini:fin])
                if f > .0:
                    dS = f/float(k)
                    S -= dS*np.log(dS)
                if fShuf > 0:
                    dS = fShuf/float(k)
                    Sshuffled -= dS*np.log(dS)
            
            for referenceSeq, targetAcc in zip(
                                    (originalBinarySequence, shuffledLocalBinarySequence),
                                    (intereventNodeTot, intereventNodeTotShuf)):
                interEve = 0
                first = True
                for eve in referenceSeq:
                    interEve += 1
                    if eve == 1:
                        if first:
                            first = False
                        else:
                            targetAcc[db].append(interEve)
                            interEve = 0
            
            entropyNodeTot[db].append(S/np.log(k))
            entropyNodeTotShuf[db].append(Sshuffled/np.log(k))
            
        print db,

    print("\nDone!")

    print("Doing entropy on the edges total seq...")

# Do the same with edges for each edge weight bin and evaluate their sub-sequence
# Here we compute the entropy of the sequence of all the events starting
# with the first event containing the edge then putting a one when the link is
# active and zero otherwise.

    frac = .001

    entropyPerLinkTot = {k: [] for k in range(nLinkStrengthBins)}
    entropyPerLinkTotShuf = {k: [] for k in range(nLinkStrengthBins)}

    intereventPerLinkTot = [[] for i in range(nLinkStrengthBins)]
    intereventPerLinkTotShuf = [[] for i in range(nLinkStrengthBins)]

    for db, candidates in linksInStrengthBin.iteritems():
        sample = [tuple(sorted(i)) for i in candidates if np.random.rand() < frac]
        if len(sample) == 0 and len(candidates) > 0:
            # Maximum 100 candidates
            #indexes = np.arange()
            tmp_indexes = np.arange(len(candidates))
            np.random.shuffle(tmp_indexes)
            candList = list(candidates)
            sample = [tuple(sorted(candList[i]))
                         for i in tmp_indexes[:min(len(tmp_indexes), 100)]]
        if len(sample) == 0:
            continue
        sample = set(sample)

        # For each link select the subsequence
        lll = 0
        for link in sample:
            if linkStrength[link] <= 1: continue

            seqStart = 0
            for ev in listone:
                if link == tuple(sorted(ev)):
                    break
                else:
                    seqStart += 1
            subSequence = listone[seqStart:]

            originalBinarySequence = [0] * (len(listone) - seqStart)
            for iiiIndex, eve in enumerate(subSequence):
                # Put the agent with focus as i
                if link == tuple(sorted(eve)):
                    originalBinarySequence[iiiIndex] = 1

            # The degree of the agent...
            assert sum(originalBinarySequence) == linkStrength[link], "%d != %d" % (sum(originalBinarySequence), linkStrength[link])
            
            k, nEvents = linkStrength[link], len(originalBinarySequence)
            
            originalBinarySequence = np.array(originalBinarySequence)
            shuffledLocalBinarySequence = np.array(originalBinarySequence)
            np.random.shuffle(shuffledLocalBinarySequence)
            
            splits = np.linspace(0, nEvents, k+1, dtype=int)
            splits = np.unique(splits)
            splits.sort()
            #print splits
            S = Sshuffled = 0
            for index in xrange(len(splits)-1):
                ini, fin = splits[index], splits[index+1]
                f = np.sum(originalBinarySequence[ini:fin])
                fShuf = np.sum(shuffledLocalBinarySequence[ini:fin])
                if f > .0:
                    dS = f/float(k)
                    S -= dS*np.log(dS)
                if fShuf > 0:
                    dS = fShuf/float(k)
                    Sshuffled -= dS*np.log(dS)
            
            for referenceSeq, targetAcc in zip(
                                    (originalBinarySequence, shuffledLocalBinarySequence),
                                    (intereventPerLinkTot, intereventPerLinkTotShuf)):
                interEve = 0
                first = True
                for eve in referenceSeq:
                    interEve += 1
                    if eve == 1:
                        if first:
                            first = False
                        else:
                            targetAcc[db].append(interEve)
                            interEve = 0

            entropyPerLinkTot[db].append(S/np.log(k))
            entropyPerLinkTotShuf[db].append(Sshuffled/np.log(k))

            lll += 1
            sys.stdout.write("\r%05d / %05d" % (lll, len(sample)))
            sys.stdout.flush()
        print "done bin: ", db

    print("Done, saving!")


    totalResultsEntropy = {
        "degreeBins": degreeBins, "linkStrengthBins": linkStrengthBins,
        "entropyNewLink": entropyNewLink, "entropyNewLinkShuf": entropyNewLinkShuf,
        "interevent": interevent, "intereventShuf": intereventShuf,
        "entropyPerLink": entropyPerLink, "entropyPerLinkShuf": entropyPerLinkShuf,
        "intereventPerLink": intereventPerLink, "intereventPerLinkShuf": intereventPerLinkShuf,
        "strengthBins": strengthBins, "linkStrengthBins": linkStrengthBins,
        "entropyNodeTot": entropyNodeTot, "entropyNodeTotShuf": entropyNodeTotShuf,
        "intereventNodeTot": intereventNodeTot, "intereventNodeTotShuf": intereventNodeTotShuf,
        "entropyPerLinkTot": entropyPerLinkTot, "entropyPerLinkTotShuf": entropyPerLinkTotShuf,
        "intereventPerLinkTot": intereventPerLinkTot, "intereventPerLinkTotShuf": 
    intereventPerLinkTotShuf,
        "name": selected,
    }

    pickle.dump(totalResultsEntropy, open("entropySequence_%s.pkl" % selected, "wb"))

