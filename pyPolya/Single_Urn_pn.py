import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import sys
import cPickle
import gzip
import os

def SingleUrnPn(N_Run=int(1e+3), RunLen=int(1e+3), N0=10, Rho=3, Nu=1, Perc_required=.9, ODir="./data_single_urn"):
    '''
    Function that simulates a single urn starting with *N0* equals ids inside and evolving accordingly to
    *Rho* and *Nu*. We simulate *N_Run* runs with *RunLen* steps each. Each point of the $p(k)$ attachment function
    is evaluated only on the points (degree values) where at least *Perc_required* percent of the run has put a value.
    We also evaluate the $<k(t)>$ and $P(k,t)$ functions.
    All the data and results are saved within the *ODir* folder.
    '''


    # The numerator and denominator of the p(n) function evaluated as the number of events toward a new id
    # done at a certain neighbors size Ssiz (Num) and the total number of events at that degree (Den).
    Num = [.0]
    Den = [.0]

    # Number of times where to track the $<k(t)>$ and $P(a,k,t)$
    NTimes = 20
    TV = np.ceil(np.logspace(2., np.log10(RunLen), NTimes))
    KV = np.zeros(NTimes)
    PKT = [[] for i in range(NTimes)]


    NP1 = Nu + 1
    Nup1 = [i for i in range(NP1)]

    for run in range(N_Run):
        Urn_ID = [i for i in range(N0)]
        Urn_SZ = [1]*N0
        Urn_Len = sum(Urn_SZ)

        #S = []
        Sset = set()
        Ssiz = 0

        for t in range(RunLen):
            if t in TV:
                KV[np.where(TV==t)] += Ssiz
                PKT[np.where(TV==t)[0]].append(Ssiz)
            sel = np.random.randint(Urn_Len)

            ID = 0
            count = Urn_SZ[ID]
            while count < sel:
                ID += 1
                count += Urn_SZ[ID]

            #S.append(ID)
            if ID not in Sset:
                Urn_ID.extend([Urn_Len+i for i in Nup1])
                Urn_SZ.extend([1]*NP1)
                Urn_Len += NP1

                # now the p(n)...
                Num[Ssiz] += 1.
                Den[Ssiz] += 1.

                Sset.add(ID)
                Ssiz += 1
                Num.append(.0)
                Den.append(.0)
            else:
                Den[Ssiz] += 1.

            Urn_SZ[ID] += Rho
            Urn_Len += Rho
        sys.stdout.write('Rep %04d of %04d done...\r' % (run, N_Run))
        sys.stdout.flush()
    print 'Done!'

    if Nu < Rho:
        handle = lambda x: ((Rho-Nu)**(float(Nu)/Rho)) * (x**(float(Nu)/Rho - 1.))
        my_handle = lambda x: x**(1. - Rho/Nu)
    else:
        my_handle = lambda x: 1.
        my_handle = lambda x: 1.

    if not os.path.exists(ODir):
        os.mkdir(ODir)
    ODir = os.path.join(ODir, "r%02d_n%02d_Nini%02d_Nreps%05d_Nsteps%06d" % (Rho, Nu, N0, N_Run, RunLen))
    if not os.path.exists(ODir):
        os.mkdir(ODir)

    Num = np.array(Num)
    Den = np.array(Den)

    K = np.arange(0, len(Num), 1.)
    K = K[Num>=N_Run*Perc_required]
    Den = Den[Num>=N_Run*Perc_required]
    Num = Num[Num>=N_Run*Perc_required]

    plt.loglog(K, Num/Den, 's-r')
    plt.loglog(K[K>.0], handle(K[K>.0]), '--k')
    plt.loglog(K[K>.0], my_handle(K[K>.0])*Num[-1]/Den[-1]/my_handle(K[-1]), '--b')
    plt.savefig(os.path.join(ODir, 'pn.pdf'))
    plt.close()

    plt.loglog(TV, KV/N_Run, 'o-b')
    plt.loglog(TV, TV**(float(Nu)/Rho), '--r')
    plt.savefig(os.path.join(ODir, 'kt.pdf'))
    plt.close()

    for t in range(0, NTimes, 3):
        f,b = np.histogram(PKT[t], density=True)
        b = (b[1:]+b[:-1])/2.
        avgk = (f*b).sum()/f.sum()#*(b[1]-b[0])

        plt.semilogy( (b-avgk)/avgk**.5, avgk**.5 * f, '.-', label=r'$t=%d$'%TV[t])
    plt.legend(loc=1)
    plt.savefig(os.path.join(ODir, 'Pkt.pdf'))
    plt.close()

    with open(os.path.join(ODir, 'pn.dat'), 'w') as f:
        for k, p in zip(K, Num/Den):
            f.write('%06f\t%.04e\n'% (k,p))

    DATA = {"t": TV, "avg_k": KV, "NRun": N_Run, "PKT": PKT, "ks": K, "Num": Num, "Den": Den}
    of = gzip.open(os.path.join(ODir, "data.dat"), 'wb')
    cPickle.dump(DATA, of)
    of.close()






