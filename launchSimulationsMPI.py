import sys
from mpi4py import MPI
from random import shuffle
import datetime
import pickle

worldComm = MPI.COMM_WORLD
worldSize = worldComm.Get_size()
worldRank = worldComm.Get_rank()

fileList = "todoList_missing.dat"
todoList = pickle.load(open(fileList, "rb"))

doneFile = "done_list_%02d.dat" % worldRank
try:
    doneList = pickle.load(open(doneFile, "rb"))
except:
    doneList = []

def writeOut(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

def saveWorkDone(tmp, done):
    done.append(tmp)
    pickle.dump(done, open(doneFile, "wb"))

if worldRank == 0:
    writeOut("Got %d entries from %s:\n%r" % (len(todoList), fileList, todoList))

for tmp_pars in todoList[worldRank::worldSize]:
    if tmp_pars in doneList: continue
    son_scheme, sample_scheme, rho, nu, nf, neve, runIDX = tmp_pars
    if nu > rho: continue
    if son_scheme == 0:
        if sample_scheme == 0:
            from codeSon0Sample0.Urns import Urnes_Evolution as UE
        elif sample_scheme == 1:
            from codeSon0Sample1.Urns import Urnes_Evolution as UE
        elif sample_scheme == 2:
            from codeSon0Sample2.Urns import Urnes_Evolution as UE
    elif son_scheme == 1:
        if sample_scheme == 0:
            from codeSon1Sample0.Urns import Urnes_Evolution as UE
        elif sample_scheme == 1:
            from codeSon1Sample1.Urns import Urnes_Evolution as UE
        elif sample_scheme == 2:
            from codeSon1Sample2.Urns import Urnes_Evolution as UE
    elif son_scheme == 2:
        if sample_scheme == 0:
            from codeSon2Sample0.Urns import Urnes_Evolution as UE
        elif sample_scheme == 1:
            from codeSon2Sample1.Urns import Urnes_Evolution as UE
        elif sample_scheme == 2:
            from codeSon2Sample2.Urns import Urnes_Evolution as UE
    writeOut("Process %03d doing %r at %r..." % (worldRank, tmp_pars, datetime.datetime.now()))
    UE(rho, nu, nf, neve, runIDX)
    del UE
    saveWorkDone(tmp_pars, doneList)

