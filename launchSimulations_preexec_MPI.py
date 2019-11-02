import pickle
import sys
from random import shuffle

# If you already did simulations and don't want to overwrite
# delete previous results indrease this counter...
startingRunIDX = 0

parsToDo = {\
        1./1.: {"r": range(1, 16, 1), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        2./3.: {"r": range(2, 16, 2), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        1./2.: {"r": range(1, 11, 1), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        3./4.: {"r": range(3, 16, 3), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        4./5.: {"r": range(4, 16, 4), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        4./3.: {"r": range(4, 28, 4), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        3./2.: {"r": range(3, 28, 3), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        5./3.: {"r": range(5, 28, 5), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        5./4.: {"r": range(5, 28, 5), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        7./4.: {"r": range(7, 28, 7), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        2./1.: {"r": range(2, 28, 2), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        5./2.: {"r": range(5, 28, 5), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        9./4.: {"r": range(9, 28, 9), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        11./4.:{"r": range(11,28,11), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        7./3.: {"r": range(7, 28, 7), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        8./3.: {"r": range(8, 28, 8), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        3./1.: {"r": range(3, 28, 3), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        13./4.:{"r": range(13,28,13), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        7./2.: {"r": range(7, 28, 7), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        15./4.:{"r": range(15,31,15), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        4./1.: {"r": range(4, 28, 4), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        9./2.: {"r": range(9, 28, 9), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
        5./1.: {"r": range(5, 28, 5), "nf": 10, "nRuns": 10, "nevef": int(5e5)},
    }

parList = []
for sonExch in range(2):
    for schExch in range(3):
        for ratio, ratiopar in sorted(parsToDo.iteritems()):
            for rho in ratiopar["r"]:
                nu = int(rho/ratio)
                NF = ratiopar["nf"]
                NevF = ratiopar["nevef"]
                nRuns = ratiopar["nRuns"]
                for runIDX in range(nRuns):
                    parList.append((sonExch, schExch, rho, nu,
                                    NF, NevF, runIDX+startingRunIDX))

outFile = str(sys.argv[1])
pickle.dump(parList, open(outFile, "wb"))
