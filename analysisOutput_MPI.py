# import matplotlib
# matplotlib.use("Agg")

import numpy as np
# import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
# import seaborn as sns
import cPickle, sys, os, gzip
from glob import glob

from scipy.optimize import curve_fit

from my_foos import Lin_Log_Bins, Smooth_Curve, p_n_pow_const, Power_Growth
from analysisOutputFoos import computeExponentKt, computeBetaOpt, computeFactorsAndCostants


#################
# Configuration #
#################
IDIr_rawData = "/home/ubi/urns_serie/data_analyzed/"
IDIr_outData = "/home/ubi/urns_serie/out/"

mnemonicName = "resultsTOT_ALL_MPI_FINAL_"

minimumTime = lambda ratio_val, rho_val: 1000 if ratio_val < 1. else 1000
sonExchSchemes = range(2)
smplStrategies = range(3)

#################
#################

from mpi4py import MPI
worldComm = MPI.COMM_WORLD
worldSize = worldComm.Get_size()
worldRank = worldComm.Get_rank()

if worldRank == 0:
    # Computes the list of files and folders to do and passes it to the others...
    # We pass a tuple that is like ({"conn", "outF"}, IDIR, fname, SON, EXCH)
    # We assume that for each folder in `IDIr_outData` there is a corrsponding file ending
    # with "_connections.dat" in teh IDIr_rawData folder. The run idx will be assumed to
    # be identical in the two.
    workLoad = []

    for sonExch in sonExchSchemes:
        for sampleStrat in smplStrategies:
            file_prefix = "Symm_SonsExchg%d_StrctSmpl%d_r" % (sonExch, sampleStrat)
            file_suffix = "_connections.dat"

            iii = 0
            dirsToConsider = sorted(os.listdir(IDIr_outData))
            for fname in dirsToConsider:
                iii += 1
                if fname.startswith(file_prefix)\
                   and os.path.isdir(os.path.join(IDIr_outData, fname)):
                    runIDX = int(fname.split("_Run_")[-1][:2])
                    fnameGrace = fname + file_suffix
                    workLoad.append((IDIr_outData, fname, fnameGrace,
                                    sonExch, sampleStrat, runIDX))
    print(len(workLoad))
    print(workLoad)
else:
    workLoad = None

workLoad = worldComm.bcast(workLoad, root=0)

ResultsTot = {}
iii = 0
for tmp_todo in workLoad[worldRank::worldSize]:
    IDIr, fname, fnameGrace, sonExch, sampleStrat, runIDX = tmp_todo

    ResultsTot.setdefault(sonExch, {})
    ResultsTot[sonExch].setdefault(sampleStrat, {})
    Results = ResultsTot[sonExch][sampleStrat]

    # First compute the exponents with the grace command
    computeFactorsAndCostants(IDIr_rawData, fnameGrace, ResultsTot, sonExch, sampleStrat, worldRank)

    # Then compute beta, kat etc. and append the runIDX and the Steps_Eve

    Rho = float(fname.split("_r")[1][:2])
    Nu = float(fname.split("_n")[1][:2])
    Ratio = Rho/Nu
    Evolution_Steps = int(fname.split("_t")[-1].split("_")[0])

    if Evolution_Steps < minimumTime(Ratio, Rho):
        continue

    fname = os.path.join(IDIr, fname)
    fname = os.path.join(fname, "data")
    fnameL = glob(os.path.join(fname, "*_bSchek.dat.gz"))
    if not fnameL:
        print "Warning, nothing found for %s" % fname
        continue
    fname = fnameL[0]

    DATA = cPickle.load(gzip.open(fname, "rb"))

    timeEvents = np.array(DATA["TimeVecs"]["EventsT"])
    selectedTime = timeEvents

    def asymptoticLevel(x, y0):
        return np.ones(len(x))*y0

    Results.setdefault(Ratio, {})
    Results[Ratio].setdefault(Rho, {})
    for kkkk in ('clust_t', "Fa_nu", "Fk_mu", 'old-open', 'old-close',
                 'new-open', 'new-close', 'Ev_Steps', 'beta_opt',
                 'beta_res', 'k_a_t', 'run_idx'):
        Results[Ratio][Rho].setdefault(kkkk, [])

    Results[Ratio][Rho]['run_idx'].append(runIDX)
    Results[Ratio][Rho]['Ev_Steps'].append(Evolution_Steps)

    XXXs, YYYs = timeEvents, DATA["TimeVecs"]["Clust_t"]
    SSSs = 1./(XXXs + 1.)
    p_clust, c_clust = curve_fit(asymptoticLevel, XXXs, YYYs,
                                sigma=SSSs, p0=(.5), bounds=(1e-5, 1.))
    Results[Ratio][Rho]['clust_t'].append(p_clust[0])

    totEve_t = np.array(DATA["TimeVecs"]["newCloseTriang"]) +\
               np.array(DATA["TimeVecs"]["newOpenTriang"])+\
               np.array(DATA["TimeVecs"]["oldOpenTriang"]) +\
               np.array(DATA["TimeVecs"]["oldCloseTriang"])
    for what, label in zip(
                ["oldOpenTriang", "oldCloseTriang", "newOpenTriang", "newCloseTriang"],
                ["old-open", "old-close", "new-open", "new-close"]
            ):
        XXXs, YYYs = selectedTime, np.array(DATA["TimeVecs"][what],dtype=float)/totEve_t
        SSSs = 1./(XXXs + 1.)
        p_res, c_res = curve_fit(asymptoticLevel, XXXs, YYYs, sigma=SSSs, p0=(.5), bounds=(1e-5, 1.))
        Results[Ratio][Rho][label].append(p_res[0])

    def powLaw(x, const, esponent):
        return const * x**(-esponent)

    f, b = np.histogram(DATA["Arrays"]["Act"], bins =Lin_Log_Bins(1, 20000, factor=1.5), density=True)
    b = (b[:-1] + b[1:])/2.
    b = b[f>0]
    f = f[f>0]

    try:
        res_nu, cov_nu = curve_fit(powLaw, b, f, sigma=1./b**2., maxfev=100000,
                               p0=[1., -1.], bounds=((1e-8, -3.1), (1e8, 3.1)))
    except RuntimeError:
        print "Nu Fit failed for", fname
        res_nu = [1., 1.]
    Results[Ratio][Rho]['Fa_nu'].append(res_nu[1])

    f, b = np.histogram(DATA["Arrays"]["Deg"], bins =Lin_Log_Bins(1, 20000, factor=1.3), density=True)
    b = (b[:-1] + b[1:])/2.
    b = b[f>0]
    f = f[f>0]
    try:
        res_mu, cov_mu = curve_fit(powLaw, b, f, sigma=1./b**2.,
                               p0=[1., -1.], bounds=((1e-8, -3.1), (1e8, 3.1)))
    except RuntimeError:
        print "Mu Fit failed for", fname
        res_mu = [1., 1.]
    Results[Ratio][Rho]['Fk_mu'].append(res_mu[1])

    # Fit the beta...
    tmp_beta_res = computeBetaOpt(DATA)
    # res = {"pn": Pn_Curves, "Bopt": Beta_Opt, "tot_chi2sums": Tot_Chi_Sums, "opt_pars": opt_pars}
    #Results[Ratio][Rho]['beta_res'].append(tmp_beta_res)
    Results[Ratio][Rho]['beta_opt'].append(tmp_beta_res["Bopt"])

    # Fit the <k(a,t)>
    res_kat = computeExponentKt(DATA)
    Results[Ratio][Rho]['k_a_t'].append(res_kat)

    del DATA
    if worldRank == 0:
        sys.stdout.write("\rProcess 0 did %03d / %03d" % (iii, len(workLoad)))
        sys.stdout.write(" ratio %d rho %d beta_opt %f" % (Ratio, Rho, tmp_beta_res["Bopt"]))
        sys.stdout.write(" Son %d Exch %d..." % (sonExch, sampleStrat))
        sys.stdout.flush()
    iii += worldSize
    #print "\nSon Exchange", sonExch, "done by rank %d" % worldRank

def aggregateDicts(ds):
    agg = {}
    for d in ds:
        for k0, v0 in d.iteritems():
            agg.setdefault(k0, {})
            a0 = agg[k0]
            for k1, v1 in v0.iteritems():
                a0.setdefault(k1, {})
                a1 = a0[k1]
                for k2, v2 in v1.iteritems():
                    a1.setdefault(k2, {})
                    a2 = a1[k2]
                    for k3, v3 in v2.iteritems():
                        a2.setdefault(k3, {})
                        a3 = a2[k3]
                        for k4, v4 in v3.iteritems():
                            a3.setdefault(k4, [])
                            a3[k4].extend(v4)
    return agg

# Now collecting everything...
if worldRank == 0:
    outName = mnemonicName + IDIr_outData.replace("/", "-") + ".pkl"
    print("\nEverything done, collecting and saving in %s..." % outName)
    resultsArray = [ResultsTot]
    for i in range(1, worldSize):
        resultsArray.append(worldComm.recv(source=i, tag=12))
    agg = aggregateDicts(resultsArray)
    cPickle.dump(agg, open(outName, "wb"))
    print(" DONE!!!\n")
else:
    worldComm.send(ResultsTot, dest=0, tag=12)

print("\nProcess %03d did all, bye!" % worldRank)


