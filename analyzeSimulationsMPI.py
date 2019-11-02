import sys
from mpi4py import MPI
import datetime
import pickle
import glob
import os

from tvnImporter.Network_Importer_Dev import Network_Importer as NII

worldComm = MPI.COMM_WORLD
worldSize = worldComm.Get_size()
worldRank = worldComm.Get_rank()

doneFile = "done_analysis_%02d.dat" % worldRank
try:
    doneList = pickle.load(open(doneFile, "rb"))
except:
    doneList = []


datFolder = "../data_done"
outFolder = "../out"

skipExisting = True

#####################

todo = [os.path.join(datFolder, p) for p in sorted(os.listdir(datFolder))]
todo = [p for p in todo if os.path.isdir(p)]

def writeOut(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def saveWorkDone(tmp, done):
    done.append(tmp)
    pickle.dump(done, open(doneFile, "wb"))

if worldRank == 0:
    writeOut("Got %d entries:\n%r" % (len(todo), todo))

for d in todo[worldRank::worldSize]:
    if d in doneList and skipExisting: continue

    tmp_dirname = os.path.basename(d)
    tmp_outdirn = os.path.join(outFolder, tmp_dirname)
    # # #if skipExisting and os.path.exists(tmp_outdirn) and os.path.isidir(tmp_outdirn):
    # # #    continue

    for scheme in ["eak", "ek", "ak"]:
        writeOut("Process %03d doing folder %r with scheme %s at %r..." %
                   (worldRank, d, scheme, datetime.datetime.now()))

        dat_file = NII(caller_idx=0, called_idx=1, clr_company_idx=2, cld_company_idx=3,
               starting_time=.0012, n_t_smpl=50, binning_scheme=scheme,
               act_bins_factor=1.25, deg_bins_factor=1.25, entr_bins_factor=1.5,
               zipped_f=True, IDir=d, ODir=tmp_outdirn);

    saveWorkDone(d, doneList)
