import matplotlib as matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import os, string, sys, gzip


if True: # gzipped
    ApriFile = gzip.open
else: # normali
    ApriFile = open

DATDir = sys.argv[1]
ODir = sys.argv[2]

Callers = set()
Calleds = set()
Both = set()
EVCount = 0

fnames = sorted(os.listdir(DATDir))

NofFiles = len(fnames)
T_Eve = np.zeros(NofFiles)

Neigh = {}

Clrs_Eve = np.zeros(NofFiles)
Clds_Eve = np.zeros(NofFiles)
Both_Eve = np.zeros(NofFiles)

A_t_Eve = np.zeros(NofFiles)


for ind, fn in enumerate(fnames):
    listone = []
    f = ApriFile(DATDir + fn, 'rb')
    for l in f:
        v = l.strip().split()
        listone.append([int(v[0]), int(v[1])])
    f.close()

    for v in listone:
        EVCount += 1
        Callers.add(v[0])
        Calleds.add(v[1])
        Both.add(v[0])
        Both.add(v[1])

        Neigh.setdefault(v[0], set())
        Neigh[v[0]].add(v[1])
        Neigh.setdefault(v[1], set())
        Neigh[v[1]].add(v[0])

    T_Eve[ind] = float(EVCount)
    Clrs_Eve[ind] = float(len(Callers))
    Clds_Eve[ind] = float(len(Calleds))
    Both_Eve[ind] = float(len(Both))
    A_t_Eve[ind] = float(sum([len(v) for v in Neigh.values()]))

    sys.stdout.write("File %s - %03d of %03d done...\r"\
            %(fn, ind+1, len(fnames)))
    sys.stdout.flush()

sys.stdout.write("\n\n Everything Done, plotting!\n\n")

plt.loglog(T_Eve, Clrs_Eve, 'o-', label=r'$Callers(t)$')
plt.loglog(T_Eve, Clds_Eve, 's-', label=r'$Calleds(t)$')
plt.loglog(T_Eve, Both_Eve, 'x--', label=r'$N(t)$')
plt.loglog(T_Eve, A_t_Eve, '.-', label=r'$A(t)$')
plt.loglog(T_Eve, A_t_Eve/Both_Eve, '^--', label=r'$\langle k(t)\rangle$')


fout = open('%s00/rhos/D_t.dat' % ODir, 'w')
for i in range(len(T_Eve)):
    fout.write('%.03e\t%.03e\n' % (T_Eve[i], Both_Eve[i]))
fout.close()

fout = open('%s00/rhos/CLRs_t.dat' % ODir, 'w')
for i in range(len(T_Eve)):
    fout.write('%.03e\t%.03e\n' % (T_Eve[i], Clrs_Eve[i]))
fout.close()

fout = open('%s00/rhos/CLDs_t.dat' % ODir, 'w')
for i in range(len(T_Eve)):
    fout.write('%.03e\t%.03e\n' % (T_Eve[i], Clds_Eve[i]))
fout.close()

fout = open('%s00/rhos/A_t.dat' % ODir, 'w')
for i in range(len(T_Eve)):
    fout.write('%.03e\t%.03e\n' % (T_Eve[i], A_t_Eve[i]))
fout.close()

#Exponent = 4./5.
#plt.loglog(T_Eve, Clrs_Eve[-1]*((T_Eve/T_Eve[-1])**Exponent), '--r',\
#        label=r'$t^{%.02f}$' % Exponent)

plt.xlabel(r'$t \,-\, [events]$', size=22)
plt.ylabel(r'$N(t) $', size=22)
plt.xticks(size=20)
plt.yticks(size=20)
plt.legend(loc=2, prop={'size': 22})
plt.grid(False)
plt.tight_layout()
plt.gcf().set_size_inches(12,9)
plt.savefig('%s00/rhos/zzz_D_t.pdf' % ODir, bbox_inches='tight')
plt.close()

