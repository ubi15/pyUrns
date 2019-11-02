import glob
import pickle

listone = []
for i in sorted(glob.glob("../data/Symm_SonsExchg*_Run_00")):
    # Pattern Symm_SonsExchg0_StrctSmpl2_r09_n06_t000050000000_Run_00
    tmp_val = i.split("_")
    son_sch = int(tmp_val[1][-1])
    exc_sch = int(tmp_val[2][-1])
    rho = int(tmp_val[3][1:])
    nu = int(tmp_val[4][1:])
    tTot = int(tmp_val[5][1:])

    nFile = 10
    nPerFile = tTot / nFile

    tmp_vals = (son_sch, exc_sch, rho, nu, nFile, nPerFile)
    print i, tmp_vals
    listone.append(tmp_vals)

pickle.dump(listone, open("todoList_missing.dat", "wb"))


