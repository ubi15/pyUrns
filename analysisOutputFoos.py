import os
import numpy as np
from subprocess import Popen, PIPE
from scipy import optimize
from scipy.optimize import curve_fit

from my_foos import Lin_Log_Bins, Smooth_Curve, p_n_pow_const, Power_Growth

def minDeg(DATA, nodeClass):
    tmpBin = DATA["Bins"]["Bins"]
    binningScheme = DATA["Params"]["binningScheme"]
    for i in range(binningScheme.index("k")):
        tmpBin = tmpBin["v"][nodeClass[i]]
    return tmpBin["b"][nodeClass[i+1]]


def computeBetaOpt(DATA, Threshold=5, Nmin=5):
    ################################################
    # Threshold: a point must have at least thres-events, thres new events and
    # a difference between the two of at least thres to be accounted for...
    #  Nmin : Minimum number of valid points for a curve to be fitted...
    # Function that returns the mother class(es) to be used for a single pna plot
    currentSuperClass = lambda classID: classID[:1]

    ################################################
    # Beta interval to sweep...
    Beta_int = np.arange(.0, 3.75, .01)

    # Fixed beta overall or use the optimal beta for each bin...
    Fixed_Beta = True

    # setting bounds for c...
    const_bounds = (1e-4, 1e+4)

    # Factor smooting the curves...
    smooth_factor = lambda v: (float(max(v))/max(1., min(v)))*.099

    binningScheme = DATA["Params"]["binningScheme"]
    pna = DATA['pna']

    # Here we save all the curves for the later fitting procedure and heat-map...
    Pn_Curves = {}

    # We plot the p(n,a)
    lastClass = None
    for nodeClass, pni in sorted(pna.items()):
        valid_degs = [k for k, evs in sorted(pni.items())\
                     if evs['s_new'] >= Threshold and evs['s_eve'] >= Threshold\
                     and (evs['s_eve'] - evs['s_new']) >= Threshold\
                     and float(k) <= minDeg(DATA, nodeClass)]

        X = np.array([float(k) for k in valid_degs])
        Y = np.array([float(pni[k]['s_new'])/float(pni[k]['s_eve']) for k in valid_degs])

        # Cleaning the zeroes...
        X = X[Y>.0]
        Y = Y[Y>.0]

        # Computing the uncertainity on the measured $p(k)$...
        STD_err = np.array([((Y[i]*(1.-Y[i]))/float(pni[k]['s_eve']))**.5\
                            for i,k in enumerate(valid_degs)])

        if len(X) > Nmin:
            X_plot, Y_plot = Smooth_Curve(X, Y, factor=smooth_factor(X))
            Pn_Curves[nodeClass] = {'x': X, 'y': Y, 'w': STD_err, 'x_smooth': X_plot, 'y_smooth': Y_plot}

    # Now we compute the optimal $\beta$, then plotting the fitted $p(k\to k+1)$ with the total rescaled one...

    Tot_Chi_Sums = np.zeros(len(Beta_int), dtype=float)
    opt_pars = {b: {nc: [] for nc, _ in Pn_Curves.items()} for b in Beta_int}
    Chi_Sums = {b: {nc: .0 for nc, _ in Pn_Curves.items()} for b in Beta_int}

    for ii_ind, bb_val in enumerate(Beta_int):
        opt_params = np.array([bb_val, 2.])
        opt_bounds = [(bb_val*.9999, bb_val*1.0001), const_bounds]

        sum_chi_temp, sum_pop_temp, pts_pop_temp = .0, .0, .0
        for nodeClass, classDict in sorted(Pn_Curves.items()):
            nn = classDict['x']
            pn = classDict['y']
            wn = classDict['w']

            par_out, chi2_tmp, dic_out = optimize.fmin_l_bfgs_b(p_n_pow_const, x0 = opt_params,\
                    args=(nn, pn, wn), bounds=opt_bounds, approx_grad=True,\
                    maxfun=1000000, maxiter=1000000)

            # Saving the summed chi square...
            sum_chi_temp += chi2_tmp
            sum_pop_temp += DATA['Bins']['N_A_K'][nodeClass]
            pts_pop_temp += float(len(nn) - 1)

            # Saving the constant and the chi_squared...
            opt_pars[bb_val][nodeClass] = [v for v in par_out]
            Chi_Sums[bb_val][nodeClass] = chi2_tmp

        if sum_pop_temp != .0:
            # We normalize by the number of fitted points, indeed this will be
            # the same number for every beta (we always fit the same curves!).
            Tot_Chi_Sums[ii_ind] = sum_chi_temp/pts_pop_temp

    Ind_Bopt = np.argmin(Tot_Chi_Sums)
    Beta_Opt = Beta_int[Ind_Bopt]

    return {"pn": Pn_Curves, "Bopt": Beta_Opt, "tot_chi2sums": Tot_Chi_Sums, "opt_pars": opt_pars}

def avgAct(DATA, nodeClass, activityStr="a"):
    tmpBin = DATA["Bins"]["Bins"]
    binningScheme = DATA["Params"]["binningScheme"]
    i = -1
    for i in range(binningScheme.index(activityStr)):
        tmpBin = tmpBin["v"][nodeClass[i]]
    return sum(tmpBin["b"][nodeClass[i+1]:nodeClass[i+1]+1])/2.

def computeExponentKt(DATA, fit_from=5):
    # Fit from the X that has t/t_e >= fit_from

    # Select the aggregator of classes...
    currentSuperClass = lambda classID: classID[:1]
    tmpBins = DATA["Bins"]["Bins"]
    binningScheme = DATA["Params"]["binningScheme"]
    selectedSuperClasses = [(i,) for i in range(2, len(tmpBins["b"])-1, 1)]

    activityString = "a" if binningScheme in ("ak", "eak") else "e"

    # Fitting classes to account...
    classFrom, classTo = 0, len(selectedSuperClasses)

    Times = range(0, DATA["Params"]["timeSampled"])     # The indexes of the TVec to be considered...

    kat = DATA['pkt']
    Pakt = DATA['pkt']

    acts = np.array(DATA['Arrays']['Act'])
    max_act = acts.max()
    acts /= max_act
    avg_act = acts.sum()/len(acts)
    Fit_X, Fit_Y = [], []
    for superClass in sorted(selectedSuperClasses):
        if superClass not in selectedSuperClasses[classFrom:classTo]:
            continue
            
        degsFreqs = {}
        validKeys = [nodeClass for nodeClass in sorted(Pakt.keys()) if currentSuperClass(nodeClass) == superClass]
        for nodeClass in validKeys:
            Res = Pakt[nodeClass]
            for time, Values in Res.iteritems():
                if time not in Times:
                    continue
                for tmpDeg, tmpFreq in Values.iteritems():
                    degsFreqs.setdefault(time, {"k": .0, "n": .0})
                    degsFreqs[time]["k"] += tmpDeg*tmpFreq
                    degsFreqs[time]["n"] += tmpFreq        
        if len(degsFreqs) < 2: continue

        # Define something similar to mindeg to compute the avg act...
        act_tmp = avgAct(DATA, nodeClass, activityStr=activityString)
        X = np.array([float(DATA["TimeVecs"]["EventsT"][k]) for k, v in sorted(degsFreqs.items())])
        X *= max_act/X[-1]########

        #Y = np.array([v["k"]/v["n"] for k, v in sorted(degsFreqs.items())])
        Y = np.array([v["k"]/degsFreqs[max(degsFreqs.keys())]["n"] for k, v in sorted(degsFreqs.items())])

        if activityString == "a":
            ####### X *= act_tmp + avg_act
            X *= act_tmp/max_act + avg_act
        elif activityString == "e":
            X /= act_tmp

        Fit_X.append(X[X>=fit_from])
        Fit_Y.append(Y[X>=fit_from])

    pars_out = []
    chi2_out = []
    for curveX, curveY in zip(Fit_X, Fit_Y):
        if len(curveX) < 5: continue
        par_out, chi2_tmp, dic_out = optimize.fmin_l_bfgs_b(Power_Growth, x0=[1., .5, 1e-6],
            args=(curveX, curveY), bounds=[(1e-8, 1e+8), (0.025, 1.025), (.0, 1e-4)],
            approx_grad=True, maxfun=1000000, maxiter=1000000)
        pars_out.append(par_out)
        chi2_out.append(chi2_tmp)

    if len(pars_out) > 0:
        best_pars = np.array([p[1] for p in pars_out])
        bestExp = np.average(best_pars,
                              weights=[1./max(1e-3, c) for c in chi2_out])
        bestExpSTD = np.std(best_pars)
    else:
        best_pars = np.array([])
        bestExp = .0
        bestExpSTD = .0
    return (best_pars, np.array(chi2_out), bestExp, bestExpSTD)


def computeFactorsAndCostants(IDIr, fname, ResultsTot, sonExch, sampleStrat, procRank, Factor_Analysis='./Factors_Analysis.sh'):

    Results = ResultsTot[sonExch][sampleStrat]
    file_prefix = "Symm_SonsExchg%d_StrctSmpl%d_r" % (sonExch, sampleStrat)
    file_suffix = "_connections.dat"
    if not ( fname.startswith(file_prefix) and fname.endswith(file_suffix) ):
        raise RuntimeError, "Invalid filename passed to computeFactorsAndCostants: %s" % fname

    Rho = float(fname.split("_r")[1][:2])
    Nu = float(fname.split("_n")[1][:2])
    Evolution_Steps = int(fname.split("_t")[-1].split("_")[0])
    Ratio = Rho/Nu

    fname = os.path.join(IDIr, fname)
    args = [Factor_Analysis, fname, "%d" % procRank]
    P  = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = P.communicate()

    if err:
        print err
    else:
        #print procRank, out
        pass

    str2float = lambda s: float(s.replace(",", "."))
    out = out.splitlines()
    for line_num, line in enumerate(out):
        if "DTexp" in line:
            DT_exp = str2float(out[line_num+1])
        elif "ATexp" in line:
            AT_exp = str2float(out[line_num+1])
        elif "FTexp" in line:
            FT_exp = str2float(out[line_num+1])
        elif "PToFATH" in line:
            PtoF_const = str2float(out[line_num+1])
        elif "LAST TIME VALS" in line:
            vals = out[line_num+1].split()
            [DT_const, AT_const, FT_const] = [str2float(vals[i]) for i in [2,8,9]]

    if False:
        print "\n###################################\n"
        print "File: ", fname
        print "Ratio: ", Ratio, " rho ", Rho, " nu ", Nu, " Steps ", Evolution_Steps
        print "D(t): ", DT_const, DT_exp
        print "A(t): ", AT_const, AT_exp
        print "F(t): ", FT_const, FT_exp
        print "p(t): ", PtoF_const
        print "\n###################################\n"


    Results.setdefault(Ratio, {})
    Results[Ratio].setdefault(Rho, {})
    for kkk in ('D_c', "D_e", "A_c", "A_e", "F_c", "F_e", "p_to_f"):
        Results[Ratio][Rho].setdefault(kkk, [])

    Results[Ratio][Rho]['D_c'].append(DT_const)
    Results[Ratio][Rho]['D_e'].append(DT_exp)
    Results[Ratio][Rho]['A_c'].append(AT_const)
    Results[Ratio][Rho]['A_e'].append(AT_exp)
    Results[Ratio][Rho]['F_c'].append(FT_const)
    Results[Ratio][Rho]['F_e'].append(DT_exp)
    Results[Ratio][Rho]['p_to_f'].append(PtoF_const)


