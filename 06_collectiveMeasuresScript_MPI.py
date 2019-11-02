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
from time import sleep
#from itertools import izip
from itertools import combinations
from collections import Counter
#import networkx as nx
import graph_tool as gt
from graph_tool.clustering import global_clustering, local_clustering
from graph_tool.correlations import assortativity

selected = str(sys.argv[1])

if len(sys.argv) > 2:
    nLinksAPSsamples = int(sys.argv[2])
    if len(sys.argv) == 4:
        nRunAPSsamples = int(sys.argv[3])
    else:
        nRunAPSsamples = 0
else:
    nLinksAPSsamples = 1
    nRunAPSsamples = 0

# Params
nodeFraction = .05
edgeFraction = .01

networkDumpPresent = False
flattenEventsList = False
dataRoot = "/data/urns_serie"

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
    inputDir = os.path.join(dataRoot, "APS/subsamples/n_links_%02d/sample_%02d/*"
                            % (nLinksAPSsamples,nRunAPSsamples))
    minimumDegree = 4
    minimumStrength = 5
    sampleFrac = .95
    nodeFraction = .25
    edgeFraction = .1
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


# Here we write the graph to a graph-tool network: even though the initialization
# is slower than networkx the memory footprint is way smaller and we can broadcast
# the network to many computing nodes.
# Load the sequence and then select nodes by their strength and compute the heaps
# constant and exponent accordingly for each bin

if worldRank == 0 and not networkDumpPresent:
    # Load the sequence
    Graph = gt.Graph(directed=False)

    # the int to index and the Counter of edges strength...
    node2idx = {}
    edgesWeight = Counter()
    nodesDict = {}
    interevents = {}

    # Load the sequence
    eveCounter = 0
    for f in sorted(glob(inputDir)):
        apri = gzip.open if gzipped else open
        for tmp_events in readFilesInChuncks(f, apri, lineSplitter, flatten=flattenEventsList):
            tmp_events = [tuple(sorted(e)) for e in tmp_events if e[0] != e[1]]
            for eve in tmp_events:
                i, j = eve
                try:
                    node_i = node2idx[i]
                except KeyError:
                    node_i = len(node2idx)
                    node2idx[i] = node_i
                try:
                    node_j = node2idx[j]
                except KeyError:
                    node_j = len(node2idx)
                    node2idx[j] = node_j
                edgesWeight.update([tuple(sorted([node_i, node_j]))])

                # The interevent part...
                for who in eve:
                    try:
                        tmp_inter = eveCounter - nodesDict[who]
                        try:
                            interevents[tmp_inter] += 1
                        except KeyError:
                            interevents[tmp_inter] = 1
                    except KeyError:
                        pass
                    nodesDict[who] = eveCounter

                eveCounter += 1
        print(f, eveCounter)
    print("Sequence loaded")

    # Saving burstiness details...
    intereventsArray = np.array([[k, v] for k, v in interevents.items()])
    fBurst, bBurst = np.histogram(intereventsArray[:,0], weights=intereventsArray[:,1],
                                 bins=np.logspace(0, 12, 30), density=True)

    bBurst = (bBurst[1:] + bBurst[:-1])/2.
    bBurst = bBurst[fBurst>0]
    fBurst = fBurst[fBurst>0]
    del interevents, nodesDict, intereventsArray

    Graph = gt.Graph(directed=False)
    Graph.add_vertex(len(node2idx))
    # Build up the i, j, weight_ij
    edgesNweight = np.array([[k[0], k[1], w] for k, w in edgesWeight.items() if w>0], dtype=int)
    weightProp = Graph.new_edge_property("double")
    Graph.add_edge_list(edgesNweight, eprops=[weightProp])
    Graph.edge_properties["w"] = weightProp

    print("DONE loading graph, computing overlap and ki*kj...")
    kkk = Graph.degree_property_map("total").a
    overlapProp = Graph.new_edge_property("double", val=0.)
    kikjProp    = Graph.new_edge_property("double", val=1.)
    neighbContainer = {}
    iii = 0
    print(list(Graph.get_edges())[0])
    for source, target, edgeIndex in Graph.get_edges(eprops=[Graph.edge_index]):
        try:
            neighbors_i = neighbContainer[source]
        except KeyError:
            neighbors_i = set(Graph.get_out_neighbors(source))
            neighbContainer[source] = neighbors_i
        try:
            neighbors_j = neighbContainer[target]
        except KeyError:
            neighbors_j = set(Graph.get_out_neighbors(target))
            neighbContainer[target] = neighbors_j

        kkk_i = kkk[source]
        kkk_j = kkk[target]
        kikjProp.a[edgeIndex] = kkk_i*kkk_j
        tmp_overlap = .0
        if kkk_i > 1. and kkk_j > 1.:
            tmp_overlap = float(len(neighbors_i.intersection(neighbors_j)))
            #assert tmp_overlap <= min(kkk_i-1, kkk_j-1), "ki %d kj %d ov %d" % (kkk_i, kkk_j, tmp_overlap)
            tmp_overlap /= (kkk_i-1.) + (kkk_j-1.) - tmp_overlap
        overlapProp.a[edgeIndex] = tmp_overlap

        if iii % 1000 == 0:
            sys.stdout.write("\r%09d / %09d" % (iii, Graph.num_edges()))
            sys.stdout.flush()
        iii += 1
    Graph.edge_properties["o"] = overlapProp
    Graph.edge_properties["kij"] = kikjProp
    del neighbContainer

    print("DONE computing overlap and kikj, saving graph dump...")
    Graph.save("networkDump_%s.xml.gz" % selected)
    pickle.dump(bBurst, gzip.open("networkBburst_%s.pkl.gz" % selected, "wb"))
    pickle.dump(fBurst, gzip.open("networkFburst_%s.pkl.gz" % selected, "wb"))
    print("DONE saving graph!")
elif networkDumpPresent:
    Graph = gt.load_graph("networkDump_%s.xml.gz" % selected)
    weightProp = Graph.edge_properties["w"]
    overlapProp = Graph.edge_properties["o"]
    kikjProp = Graph.edge_properties["kij"]
    if worldRank == 0:
        bBurst = pickle.load(gzip.open("networkBburst_%s.pkl.gz" % selected, "rb"))
        fBurst = pickle.load(gzip.open("networkFburst_%s.pkl.gz" % selected, "rb"))


# If not present wait for one to finish then load... put a sleep for I/O delays
worldComm.Barrier()
if worldRank > 0:
    sleep(5)
    Graph = gt.load_graph("networkDump_%s.xml.gz" % selected)
    weightProp = Graph.edge_properties["w"]
    overlapProp = Graph.edge_properties["o"]
    kikjProp = Graph.edge_properties["kij"]


#FOR MPI over the nodes...
strengthProp = Graph.new_vertex_property("double")
degreeProp = Graph.new_vertex_property("double")
kkk = Graph.degree_property_map("total").a
sss = Graph.degree_property_map("total", weight=weightProp).a
degreeProp.a   = kkk
strengthProp.a = sss

ccc = local_clustering(Graph).a
ccc = np.array(ccc, dtype=np.float64)
ccw = np.ones(Graph.num_vertices(), dtype=np.float64)*-1.
cww = np.ones(Graph.num_vertices(), dtype=np.float64)*-1.
knn = np.ones(Graph.num_vertices(), dtype=np.float64)*-1.
knw = np.ones(Graph.num_vertices(), dtype=np.float64)*-1.

iii = worldRank
if worldRank == 0:
    print("Computing nodes properties...")
for nodeI in Graph.get_vertices()[worldRank::worldSize]:
    if np.random.rand() < nodeFraction:
        tmp_ccw = tmp_cww = .0
        tmp_knn = tmp_knw = tmp_wsum = .0

        tmp_strI = strengthProp[nodeI]
        tmp_degI  = degreeProp[nodeI]
        neighborsI = Graph.get_out_edges(nodeI, [Graph.edge_index])
        neighbor_index = 0
        for _, nodeJ, index_eIJ in neighborsI:
            tmp_weightIJ = weightProp.a[index_eIJ]
            tmp_degJ = degreeProp[nodeJ]
            tmp_knn += tmp_degJ
            tmp_knw += tmp_degJ*tmp_weightIJ
            tmp_wsum += tmp_weightIJ
            if tmp_degI > 1:
                for __, nodeK, index_eIK in neighborsI[neighbor_index+1:,:]:
                    tmp_edge = Graph.edge(nodeJ, nodeK)
                    if tmp_edge:
                        tmp_weightIK = weightProp.a[index_eIK]
                        tmp_weightJK = weightProp[tmp_edge]
                        tmp_cww += (tmp_weightIJ*tmp_weightIK*tmp_weightJK)**.33333
                        tmp_ccw += tmp_weightIJ + tmp_weightIK
                # We avoid the /2 as we counted only half of the matrix...
                tmp_cww /= tmp_degI*(tmp_degI - 1.)
                tmp_ccw /= tmp_strI*(tmp_degI - 1.)
            neighbor_index += 1

        tmp_knn /= tmp_degI
        tmp_knw /= tmp_wsum

        cww[iii] = tmp_cww
        ccw[iii] = tmp_ccw
        knn[iii] = tmp_knn
        knw[iii] = tmp_knw
    if worldRank == 0 and iii % 1000 == 0:
        sys.stdout.write("\r%08d / %08d" %
                         (iii, Graph.num_edges()))
        sys.stdout.flush()
    iii += worldSize

if worldRank == 0:
    print("\nknn, knw, cww and ccw done!")
    print("Local analysis done, collecting results...")
    tmp_buf = np.array(kkk, dtype=np.float64)
    for source in range(1, worldSize):
        worldComm.Recv(tmp_buf, source=source, tag=2)
        ccw[source::worldSize] = tmp_buf[source::worldSize]
        worldComm.Recv(tmp_buf, source=source, tag=3)
        cww[source::worldSize] = tmp_buf[source::worldSize]
        worldComm.Recv(tmp_buf, source=source, tag=4)
        knn[source::worldSize] = tmp_buf[source::worldSize]
        worldComm.Recv(tmp_buf, source=source, tag=5)
        knw[source::worldSize] = tmp_buf[source::worldSize]
    del tmp_buf
else:
    print("Process %03d finished local computations..." % worldRank)
    worldComm.Send(ccw, dest=0, tag=2)
    worldComm.Send(cww, dest=0, tag=3)
    worldComm.Send(knn, dest=0, tag=4)
    worldComm.Send(knw, dest=0, tag=5)

    del ccc, ccw, cww, knn, knw

if worldRank == 0:
    print("\nDone collection, now computing the degree correlations...")
    print("Node 0 doing the average overlap per percentile and per position...")
    weights = np.array(weightProp.a)
    overlaps = np.array(overlapProp.a)
    wmin, wmax = weights.min(), weights.max()
    weightsPercentiles = np.percentile(weights, np.linspace(0, 100, 20))
    avgOverlap = np.zeros(len(weightsPercentiles))
    iii = 0
    for weightsPercentile in weightsPercentiles:
        avgOverlap[iii] = np.mean(overlaps[weights <= weightsPercentile])
        iii += 1

    weight_indexes = np.argsort(weights)
    weightsPercentilesPosition = np.linspace(1, len(weights)-1, 20, dtype=int)
    avgOverlapPerPosition = np.zeros(len(weightsPercentilesPosition))
    iii = 0
    for weightsPercentile in weightsPercentilesPosition:
        avgOverlapPerPosition[iii] = np.mean(overlaps[weight_indexes[:weightsPercentile]])
        iii += 1
    del weights, overlaps

    print("Node 0 did the average overlap per percentile and per position...")
    print("Doing the cluster per slices...")

# Save the overlaps in a array for convenience...
overlaps = np.array(overlapProp.a)
omin, omax = overlaps.min(), overlaps.max()
overlapsPercentiles = np.percentile(overlaps, np.linspace(0, 100, 20))
overlapsPositions = np.linspace(0, len(overlaps)-1, 20, dtype=int)
overlapsArgsort = np.argsort(overlaps)
avgClustering = np.zeros(len(overlapsPercentiles), dtype=np.float64)
avgClusteringPerPosition = np.zeros(len(overlapsPositions), dtype=np.float64)

# Now per rank...
overlapFiltering = Graph.new_edge_property("bool")
for overlapIndex in range(worldRank, len(overlapsPercentiles), worldSize):
    overlapFiltering.a = overlaps >= overlapsPercentiles[overlapIndex]
    Graph.set_edge_filter(overlapFiltering)
    nodeFilter = Graph.degree_property_map("total").a
    tmp_avg_clust = local_clustering(Graph).a[nodeFilter > 0]
    avgClustering[overlapIndex] = np.mean(tmp_avg_clust)
    Graph.clear_filters()

# Now per position...
for overlapIndex in range(worldRank, len(overlapsPositions), worldSize):
    overlapFiltering.a = False
    overlapFiltering.a[overlapsArgsort[overlapsPositions[overlapIndex]:]] = True
    Graph.set_edge_filter(overlapFiltering)
    nodeFilter = Graph.degree_property_map("total").a
    tmp_avg_clust = local_clustering(Graph).a[nodeFilter > 0]
    avgClusteringPerPosition[overlapIndex] = np.mean(tmp_avg_clust)
    Graph.clear_filters()

del overlapsPositions, overlapsArgsort, overlaps, omin, omax

if worldRank == 0:
    print("Collecting overlap results...")

    tmp_buf = np.array(avgClustering, dtype=np.float64)
    for source in range(1, worldSize):
        worldComm.Recv(tmp_buf, source=source, tag=1)
        avgClustering[source::worldSize] = tmp_buf[source::worldSize]
    tmp_buf = np.array(avgClusteringPerPosition, dtype=np.float64)
    for source in range(1, worldSize):
        worldComm.Recv(tmp_buf, source=source, tag=2)
        avgClusteringPerPosition[source::worldSize] = tmp_buf[source::worldSize]
    del tmp_buf
else:
    print("Process %03d finished clustering computations..." % worldRank)
    worldComm.Send(avgClustering, dest=0, tag=1)
    worldComm.Send(avgClusteringPerPosition, dest=0, tag=2)

    del avgClustering, avgClusteringPerPosition

# The distribution of similarity between nodes...
if worldRank == 0:
    print("Computing similarity on edges...")

sampledEdges = Graph.get_edges(eprops=[Graph.edge_index])[worldRank::worldSize, :]
sampledEdges = sampledEdges[
                    np.random.choice(np.arange(sampledEdges.shape[0]),
                                     size=max(1,int(sampledEdges.shape[0]*edgeFraction)),
                                     replace=False), :]

nodesOrdered = sorted(set(list(sampledEdges[:,0]) + list(sampledEdges[:,1])))
weightsPerNode = {}
for node_i in nodesOrdered:
    tmp_vals = {neighb: weightProp.a[index] for n, neighb, index in Graph.get_out_edges(node_i, [Graph.edge_index])}
    weightsPerNode[node_i] = {
        "v": tmp_vals,
        "n": np.sqrt(np.sum(np.array(list(tmp_vals.items()))**2.)),
    }
if worldRank == 0:
    print("Norm done...")

edgesStats = {}
iii = 0
for source, target, index in sampledEdges:
    edge = (source, target)
    weights1 = weightsPerNode[source]["v"]
    norm1 = weightsPerNode[source]["n"]
    neighb_i = set(list(weights1.keys()))

    weights2 = weightsPerNode[target]["v"]
    norm2 = weightsPerNode[target]["n"]
    neighb_j = set(list(weights2.keys()))

    common_neighbs = neighb_i.intersection(neighb_j)

    tmp_sim = .0
    if common_neighbs:
        for k in common_neighbs:
            tmp_sim += weights1[k]*weights2[k]
        tmp_sim /= norm1*norm2

    edgesStats[edge] = {
            "ki_kj": kikjProp.a[index],
            "e_str": weightProp.a[index],
            "sim": tmp_sim,
            "vrlp": overlapProp.a[index],
        }

    iii += 1
    if iii % 100 == 0 and worldRank == 0:
        sys.stdout.write("\r%08d / %08d" %
                         (iii, sampledEdges.shape[0]))
        sys.stdout.flush()


if worldRank == 0:
    print("\nEdges measures done, collecting and saving...")
    for i in range(1, worldSize):
        tmp_kikj = worldComm.recv(source=i, tag=11)
        edgesStats.update(tmp_kikj)
    print("Everything done, saving...")
else:
    print("Process %03d finished edges computations..." % worldRank)
    worldComm.send(edgesStats, dest=0, tag=11)
    del edgesStats



if worldRank == 0:
    # Cleaning out all the nodes that I did not computed...
    idxs = np.where(knn >= .0)
    kkk = np.array(kkk[idxs])
    sss = np.array(sss[idxs])
    knn = np.array(knn[idxs])
    knw = np.array(knw[idxs])
    ccc = np.array(ccc[idxs])
    ccw = np.array(ccw[idxs])
    cww = np.array(cww[idxs])

    totalResults = {"name": selected,
        "fBurst": fBurst, "bBurst": bBurst,
        "nodeDegree": kkk, "nodeStrenght": sss, "nodeClust": ccc,
        "nodeWClust": ccw, "nodeWWWClust": cww, "nodeNNdeg": knn,
        "nodeNWdeg": knw, "edgesStats": edgesStats,
        "edgesOverlap": overlapProp.a, "edgeKiKj": kikjProp.a,
        "edgesWeight": weightProp.a,
        "averageOverlapPerPerc": avgOverlap,
        "averageOverlapPerPosi": avgOverlapPerPosition,
        "averageClusterPerPerc": avgClustering,
        "averageClusterPerPosi": avgClusteringPerPosition,
    }
    if selected == "APS_samples":
        fout = "collectiveResults_%s_link%02d_sample%02d.pkl.gz" % (selected, nLinksAPSsamples, nRunAPSsamples)
    else:
        fout = "collectiveResults_%s.pkl.gz" % selected
        
    with gzip.open(fout, "wb") as f:
        pickle.dump(totalResults, f)

