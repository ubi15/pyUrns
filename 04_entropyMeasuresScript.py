import pickle
import gzip
from glob import glob
import numpy as np

import sys

from itertools import combinations
from collections import Counter

selected = str(sys.argv[1])

if selected == "TWT":
    inputDir = "/home/ubi/owncloud/PhD/TVN/strong_ties/data/twitter/twitter/data-01-09/*"
    gzipped = False
    lineSplitter = lambda l: combinations([int(e) for e in l.strip().split()], 2)
elif selected == "APS":
    inputDir = "/home/ubi/owncloud/PhD/TVN/strong_ties/data/APS/aff_data_ISI_original_divided_per_month_1960_2006/*"
    gzipped = False
    lineSplitter = lambda l: combinations([int(e) for e in l.strip().split()], 2)
elif selected == "URNS_TWT":
    inputDir = "/home/ubi/urns/data_analyzed/Symm_SonsExchg0_StrctSmpl1_r05_n05_t000005000000_Run_00/*"
    gzipped = True
    lineSplitter = lambda l: [[int(e) for e in l.strip().split()[:2]]]
elif selected == "URNS_APS":
    inputDir = "/home/ubi/urns/data_analyzed/Symm_SonsExchg1_StrctSmpl2_r03_n09_t000000500000_Run_00/*"
    gzipped = True
    lineSplitter = lambda l: [[int(e) for e in l.strip().split()[:2]]]
elif selected == "MPC":
    inputDir = "/hpc/group/G_FISSTAT/eubaldi/data_MPC/data/*"
    gzipped = True
    lineSplitter = lambda l: [[int(e) for e in l.strip().split()[1:3]]]
elif selected == "URNS_MPC":
    inputDir = "/home/ubi/urns_serie/data_analyzed/Symm_SonsExchg1_StrctSmpl1_r21_n07_t000050000000_Run_00/*"
    gzipped = True
    lineSplitter = lambda l: [[int(e) for e in l.strip().split()[:2]]]
elif selected == "URNS_PROVA":
    inputDir = "/home/ubi/urns/data_analyzed/Symm_SonsExchg1_StrctSmpl0_r10_n05_t000001000000_Run_00//*"
    gzipped = True
    lineSplitter = lambda l: [[int(e) for e in l.strip().split()[:2]]]


# Load the sequence
listone = []
eveCounter = 0
for f in sorted(glob(inputDir)):
    apri = gzip.open if gzipped else open 
    with apri(f, "rb") as tmpF:
        for l in tmpF:
            tmp_events = list(lineSplitter(l))
            listone.extend(tmp_events)
            eveCounter += len(tmp_events)
    print(f, eveCounter)

print(len(listone))
listone = [e for e in listone if e[0] != e[1]]
print(len(listone))

print("Sequence loaded")

# Group agents by degree
agentStrength = Counter([a for e in listone for a in e])
linkStrength = Counter([tuple(sorted(l)) for l in listone])
agentDegree = Counter([i for e in linkStrength.keys() for i in e])


# Bin the agents by their degree and the edges by their strength.
# We also annotate once which nodes/links are in each bin.
nDegreeBins = 25
minDeg = 2
degreeBins = np.logspace(np.log(minDeg), np.log10(max(agentDegree.values())+1), nDegreeBins)
agentDegreeBin = {i: np.argmax(degreeBins >= k) for i, k in agentDegree.iteritems() if k>= minDeg}
agentsInDegreeBin = {k: set(i for i, db in agentDegreeBin.iteritems() if db == k) for k in range(nDegreeBins)}

nLinkStrengthBins = 25
minLinkStrength = 2
linkStrengthBins = np.logspace(np.log(minLinkStrength), np.log10(max(linkStrength.values())+1), nLinkStrengthBins)
linkStrengthBin = {i: np.argmax(linkStrengthBins >= k) for i, k in linkStrength.iteritems() if k>= minLinkStrength}
linksInStrengthBin = {k: set(i for i, db in linkStrengthBin.iteritems() if db == k) for k in range(nLinkStrengthBins)}


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

