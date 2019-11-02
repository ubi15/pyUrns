import os, sys,gzip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IDir = sys.argv[1]
ODir = sys.argv[2]

if True: #gzipped files...
    Apri = gzip.open
else:
    Apri = open

Files = sorted(os.listdir(IDir))

nFiles = len(Files)


Dat = {}
eve_time = np.zeros(nFiles)

NEve = 0
for Find, fn in enumerate(Files):
    with Apri(os.path.join(IDir, fn), 'rb') as f:
        for l in f:
            v = l.strip().split()
            clr = v[0]
            cld = v[1]

            Dat.setdefault(clr, {"a": .0, "t0m": (Find+1.)-.5,\
                    "t0e": NEve})

            Dat[clr]["a"] += 1.

            Dat.setdefault(cld, {"a": .0, "t0m": (Find+1.)-.5,\
                    "t0e": NEve})

            NEve += 1

    eve_time[Find] = NEve

    sys.stdout.write("File %s - %03d of %03d done...\r"\
            %(fn, Find+1, len(Files)))
    sys.stdout.flush()

print ""
print "Done!"
print ""

Acts = [.0]*len(Dat)
T0s_mo = [.0]*len(Dat)
T0s_ev = [.0]*len(Dat)
for ID, Vals in enumerate(Dat.values()):
    Acts[ID] = Vals["a"]
    T0s_mo[ID] = Vals["t0m"]
    T0s_ev[ID] = Vals["t0e"]

Acts = np.array(Acts, dtype=np.double)
T0s_mo = np.array(T0s_mo, dtype=np.double)
T0s_ev = np.array(T0s_ev, dtype=np.double)


if not os.path.exists(ODir):
    os.mkdir(ODir)
ODir = os.path.join(ODir, "00")
if not os.path.exists(ODir):
    os.mkdir(ODir)
ODir = os.path.join(ODir, "rhos")
if not os.path.exists(ODir):
    os.mkdir(ODir)


with open(os.path.join(ODir, "zzz_Act_vs_t0.dat"), "wb") as of:
    for A, tm, te in zip(Acts, T0s_mo, T0s_ev):
        of.write("%d\t%.01f\t%.03e\n" % (A, tm, te))

plt.hexbin(T0s_mo, Acts, bins='log', cmap=plt.cm.YlOrRd_r)
plt.axis([T0s_mo.min(), T0s_mo.max(), Acts.min(), Acts.max()])
plt.xlabel(r"$t_0$[months]")
plt.ylabel(r"activity")
plt.savefig(os.path.join(ODir, "zzz_Act_vs_t0_file.pdf"))

plt.close()
plt.hexbin(T0s_ev, Acts, bins='log', cmap=plt.cm.YlOrRd_r)
plt.axis([T0s_ev.min(), T0s_ev.max(), Acts.min(), Acts.max()])
plt.xlabel(r"$t_0$[events]")
plt.ylabel(r"activity")
plt.savefig(os.path.join(ODir, "zzz_Act_vs_t0_events.pdf"))



