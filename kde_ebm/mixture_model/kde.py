# Authors: Nicholas C. Firth <ncfirth87@gmail.com>; Neil Oxtoby <https://github.com/noxtoby>
import numpy as np
from sklearn import neighbors
from awkde import GaussianKDE 
# from scipy import stats

class KDEMM(object):
    """docstring for KDEMM"""
    def __init__(self, kernel='gaussian', bandwidth=None, n_iters=1500):
        super(KDEMM, self).__init__()
        self.n_iters = n_iters
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.controls_kde = None
        self.patholog_kde = None
        self.mixture = None
        self.alpha = 0.3 # this is the bandwidth parameter
        self.beta = self.alpha # alpha for controls # sensitivity parameter: 0...1

    def fit(
            self, 
            X, 
            y, 
            implement_fixed_controls=False, 
            patholog_dirn=None, 
            outlier_controls_quantile = 0.90,
            verbose = False):
        
        if(verbose):
            print("starting `KDEMM.fit()`")
            print("patholog_dirn = ", patholog_dirn)
            
        
        #* Requires direction of disease progression as input
        if patholog_dirn is None:
            patholog_dirn = disease_direction(X,y)
        
        # ####### Diagnostic
        # if patholog_dirn < 0:
        #     print('kde.py DIAGNOSTIC: fit(), Disease progresses with decreasing biomarker values - ')
        # elif patholog_dirn > 0:
        #     print('kde.py DIAGNOSTIC: fit(), Disease progresses with increasing biomarker values + ')
        # else:
        #     print('kde.py DIAGNOSTIC. fit(), ERROR: Disease direction in fit(...,patholog_dirn) must be either positive or negative. \n patholog_dirn = {0]}'.format(patholog_dirn))
        # #######
        
        sorted_idx = X.argsort(axis=0).flatten()
        kde_values = X.copy()[sorted_idx].reshape(-1,1)
        kde_labels0 = y.copy()[sorted_idx]
        kde_labels = kde_labels0
        # if(verbose):
        #     print("reached here")
        #print('Original labels')
        #print(kde_labels.astype(int))
        
        bin_counts = np.bincount(y).astype(float)
        mixture0 = sum(kde_labels==0)/len(kde_labels) # Prior of being a control
        mixture = mixture0
        old_ratios = np.zeros(kde_labels.shape)
        iter_count = 0
        
        if(self.bandwidth is None):
            #* 1. Rule of thumb
            self.bandwidth = hscott(X)
            # #* 2. Estimate full density to inform variable bandwidth: wide in tails, narrow in peaks
            # all_kde = neighbors.KernelDensity(kernel=self.kernel,
            #                                   bandwidth=self.bandwidth)
            # all_kde.fit(kde_values)
            # f = np.exp(all_kde.score_samples(kde_values))
            # #* 3. Local, a.k.a. variable, bandwidth given by eq. 3 of https://ieeexplore.ieee.org/abstract/document/7761150
            # g = stats.mstats.gmean(f)
            # alpha = 0.5 # sensitivity parameter: 0...1
            # lamb = np.power(f/g,-alpha)

        for i in range(self.n_iters):
            if(verbose):
                print('Iteration {0}. kde_labels = {1}'.format(i,[int(k) for k in kde_labels]))
            
            #* Automatic variable/local bandwidth for each component: awkde package from github
            controls_kde = GaussianKDE(glob_bw="scott", alpha=self.beta, diag_cov=False)
            patholog_kde = GaussianKDE(glob_bw="scott", alpha=self.alpha, diag_cov=False)
            # controls_kde = GaussianKDE(glob_bw="scott", alpha=0.1, diag_cov=False)
            # patholog_kde = GaussianKDE(glob_bw="scott", alpha=0.1, diag_cov=False)
            controls_kde.fit(kde_values[kde_labels == 0], verbose = verbose, debug_info = "controls")
            patholog_kde.fit(kde_values[kde_labels == 1], verbose = verbose, debug_info = "cases")

            controls_score = controls_kde.predict(kde_values, verbose = verbose)
            patholog_score = patholog_kde.predict(kde_values, verbose = verbose)
            
            controls_score = controls_score*mixture # DM: `mixture` factor gets canceled out of `cdf_controls`
            patholog_score = patholog_score*(1-mixture)

            ratio = controls_score / (controls_score + patholog_score) # DM: doesn't seem to get used
            
            # print('Iteration {0}. ratio (percent) = {1}'.format(i,[int(r*100) for r in ratio]))
            
            #* Empirical cumulative distribution: used to swap labels for patients with super-normal values (greater/less than CDF=0.5)
            cdf_controls = np.cumsum(controls_score)/max(np.cumsum(controls_score)) # DM: this is why we needed to sort; note controls_score is joint likelihood p(x, ~E)
            cdf_patholog = np.cumsum(patholog_score)/max(np.cumsum(patholog_score))
            cdf_diff = (cdf_patholog - cdf_controls)/(cdf_patholog + cdf_controls)
            disease_dirn = -np.sign(np.nansum(cdf_diff)) # disease_dirn = -np.sign(np.mean(cdf_diff))
            if disease_dirn > 0:
                cdf_direction = 1 + cdf_diff
            else:
                cdf_direction = -cdf_diff
            
            #* Identify "normal" biomarkers as being on the healthy side of the controls median => flip patient labels
            if patholog_dirn < 0:
                #* More normal (greater) than half the controls: CDF_controls > 0.5
                labels_forced_normal_cdf = cdf_controls > 0.5
                labels_forced_normal_alt = kde_values > np.nanmedian(kde_values[kde_labels0 == 0]) # DM: apparently not used
            elif patholog_dirn>0:
                #* More normal (less)    than half the controls: CDF_controls < 0.5
                labels_forced_normal_cdf = cdf_controls < 0.5
                labels_forced_normal_alt = kde_values < np.nanmedian(kde_values[kde_labels0 == 0])
            labels_forced_normal = labels_forced_normal_cdf
            
            if(verbose):
                print("labels_forced_normal = ")
                print(labels_forced_normal)
            #* FIXME: Make this a prior and change the mixture modelling to be Bayesian
            #* First iteration only: implement "prior" that flips healthy-looking patients (before median for controls) to pre-event label
            #* Refit the KDEs at this point
            if i==0:
                if(verbose):
                    print("in first iteration special step")
                #* Disease direction: force pre-event/healthy-looking patients to flip
                kde_labels[np.where(labels_forced_normal)[0]] = 0
                
                bin_counts = np.bincount(kde_labels).astype(float)
                mixture = bin_counts[0] / bin_counts.sum()
                #* Refit the KDE components. FIXME: this is copy-and-paste from above. Reimplement in a smarter way.
                controls_kde.fit(kde_values[kde_labels == 0], verbose=verbose, debug_info = "controls")
                patholog_kde.fit(kde_values[kde_labels == 1], verbose=verbose, debug_info = "cases")
                controls_score = controls_kde.predict(kde_values, verbose=verbose)
                patholog_score = patholog_kde.predict(kde_values, verbose=verbose)
                controls_score = controls_score*mixture
                patholog_score = patholog_score*(1-mixture)
                ratio = controls_score / (controls_score + patholog_score)
                #* Empirical cumulative distribution: used to swap labels for patients with super-normal values (greater/less than CDF=0.5)
                cdf_controls = np.cumsum(controls_score)/max(np.cumsum(controls_score))
                cdf_patholog = np.cumsum(patholog_score)/max(np.cumsum(patholog_score))
                cdf_diff = (cdf_patholog - cdf_controls)/(cdf_patholog + cdf_controls)
                disease_dirn = -np.sign(np.nansum(cdf_diff)) # disease_dirn = -np.sign(np.mean(cdf_diff))
                if disease_dirn > 0:
                    cdf_direction = 1 + cdf_diff
                    # print('Disease direction is estimated to be POSTIIVE')
                else:
                    cdf_direction = -cdf_diff
                    # print('Disease direction is estimated to be NEGATIVE')
                #* Identify "normal" biomarkers as being on the healthy side of the controls median => flip patient labels
                if patholog_dirn<0:
                    #* More normal (greater) than half the controls: CDF_controls > 0.5
                    labels_forced_normal_cdf = cdf_controls > 0.5
                    labels_forced_normal_alt = kde_values > np.nanmedian(kde_values[kde_labels0 == 0])
                elif patholog_dirn>0:
                    #* More normal (less)    than half the controls: CDF_controls < 0.5
                    labels_forced_normal_cdf = cdf_controls < 0.5
                    labels_forced_normal_alt = kde_values < np.nanmedian(kde_values[kde_labels0 == 0])
                labels_forced_normal = labels_forced_normal_cdf
            
            if(verbose):
                print('after first iteration special step')

            if(np.all(ratio == old_ratios)):
                # print('MM finished in {0} iterations'.format(iter_count))
                break
            iter_count += 1
            old_ratios = ratio
            kde_labels = ratio < 0.5
            
            #* Labels to swap: 
            diff_y = np.hstack(([0], np.diff(kde_labels))) # !=0 where adjacent labels differ
            
            if ( (np.sum(diff_y != 0) >= 2) & (np.unique(kde_labels).shape[0] == 2) ): 
                split_y = int(np.all(np.diff(np.where(kde_labels == 0)) == 1)) # kde_label upon which to split: 1 if all 0s are adjacent, 0 otherwise
                sizes = [x.shape[0] for x in
                         np.split(diff_y, np.where(diff_y != 0)[0])] # lengths of each contiguous set of labels
                
                #* Identify which labels to swap using direction of abnormality: avg(controls) vs avg(patients)
                #* N ote that this is now like k-medians clustering, rather than k-means
                split_prior_smaller = (np.median(kde_values[kde_labels ==
                                                            split_y])
                                       < np.median(kde_values[kde_labels ==
                                                              (split_y+1) % 2]))
                if split_prior_smaller:
                    replace_idxs = np.arange(kde_values.shape[0])[-sizes[2]:] # greater values are swapped
                else:
                    replace_idxs = np.arange(kde_values.shape[0])[:sizes[0]] # lesser values are swapped
                kde_labels[replace_idxs] = (split_y+1) % 2 # swaps labels
            
            #* Disease direction: force pre-event/healthy-looking patients to flip
            kde_labels[np.where(labels_forced_normal)[0]] = 0
            
            #*** Prevent label swapping for "strong controls"
            fixed_controls_criteria_0 = (kde_labels0==0) # Controls 
            # #*** CDF criteria - do not delete: potentially also used for disease direction
            # en = 10
            # cdf_threshold = (en-1)/(en+1) # cdf(p) = en*(1-cdf(c)), i.e., en-times more patients than remaining controls
            # controls_tail = cdf_direction > (cdf_threshold * max(cdf_direction))
            # #fixed_controls_criteria_0 = fixed_controls_criteria_0 & (~controls_tail)
            # #*** PDF ratio criteria
            # ratio_threshold_strong_controls = 0.33 # P(control) / [P(control) + P(patient)]
            # fixed_controls_criteria = fixed_controls_criteria & (ratio > ratio_threshold_strong_controls) # "Strong controls" 
            #*** Outlier criteria for weak (e.g., low-performing on test; or potentially prodromal in sporadic disease) controls: quantiles
            if disease_dirn>0:
                q = outlier_controls_quantile # upper
                f = np.greater
                g = np.less
                # print('Disease direction: positive')
            else:
                q = 1 - outlier_controls_quantile # lower
                f = np.less
                g = np.greater
                # print('Disease direction: negative')
            extreme_cases = f(kde_values,np.quantile(kde_values,q)).reshape(-1,1) #& (kde_labels0==0)
            fixed_controls_criteria = fixed_controls_criteria_0.reshape(-1,1) & ~(extreme_cases)
            if implement_fixed_controls:
                kde_labels[np.where(fixed_controls_criteria)[0]] = 0
                #kde_labels[np.where(controls_outliers)[0]] = 1 # Flip outlier controls
            
            #* One final fit, following relabelling
            controls_kde.fit(kde_values[kde_labels == 0])
            patholog_kde.fit(kde_values[kde_labels == 1])
            
            bin_counts = np.bincount(kde_labels).astype(float)
            mixture = bin_counts[0] / bin_counts.sum()
            if(mixture < 0.10 or mixture > 0.90): # if(mixture < (0.90*mixture0) or mixture > 0.90):
                # print('MM finished (mixture weight too low/high) in {0} iterations'.format(iter_count))
                break
        self.controls_kde = controls_kde
        self.patholog_kde = patholog_kde
        self.mixture = mixture
        self.iter_ = iter_count
        return self

    def likelihood(self, X):
        controls_score, patholog_score = self.pdf(X)
        data_likelihood = controls_score+patholog_score
        data_likelihood = np.log(data_likelihood)
        return -1*np.sum(data_likelihood)


    # actually gives the joint probability of X and group membership
    def pdf(self, X, **kwargs):
        #* Old version: sklearn fixed-bw KDE
        # controls_score = self.controls_kde.score_samples(X)
        # controls_score = np.exp(controls_score)*self.mixture
        # patholog_score = self.patholog_kde.score_samples(X)
        # patholog_score = np.exp(patholog_score)*(1-self.mixture)
        #* Auto-Variable-bw KDE: awkde
        controls_score = self.controls_kde.predict(X)*self.mixture
        patholog_score = self.patholog_kde.predict(X)*(1-self.mixture)
        return controls_score, patholog_score

    def pdfs_mixture_components(self, X, **kwargs):
        #* Old version: sklearn fixed-bw KDE
        # controls_score = self.controls_kde.score_samples(X)
        # patholog_score = self.patholog_kde.score_samples(X)
        #* Auto-Variable-bw KDE: awkde
        controls_score = self.controls_kde.predict(X)
        patholog_score = self.patholog_kde.predict(X)
        return controls_score, patholog_score

    def impute_missing(self,X,num=1000):
        # FIXME: move to mixture_model/utils.py (because it's also in gmm.py)
        if np.isnan(X).any():
            # High-resolution fake data vector
            x = np.linspace(np.nanmin(X),np.nanmax(X),num=num).reshape(-1, 1)
            # Unweighted likelihoods from the MM: p(x|E) and p(x|~E)
            p_x_given_notE, p_x_given_E = self.pdfs_mixture_components(x)
            # Find x_missing where p(x_missing|E) == p(x_missing|~E)
            likelihood_abs_diff = np.abs(p_x_given_notE - p_x_given_E) # BEWARE: minimum diff might be at edges => use d^2 (diff) / dx^2 > 0
            likelihood_abs_diff_d2 = np.gradient(np.gradient(likelihood_abs_diff))
            central_minimum = np.where(likelihood_abs_diff_d2==np.max(likelihood_abs_diff_d2))[0]
            x_missing = x[ np.nanmedian(central_minimum).astype(int) ] # handles multiple x (takes median)
            # Impute
            missing_entries = np.isnan(X)
            X_imputed = np.copy(X)
            np.putmask(X_imputed,missing_entries,x_missing)
            return X_imputed
        else:
            return X

    def probability(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        #* Handle missing data
        # controls_score[np.isnan(controls_score)] = 0.5
        # patholog_score[np.isnan(patholog_score)] = 0.5
        # controls_score[controls_score==0] = 0.5
        # patholog_score[patholog_score==0] = 0.5
        c = controls_score / (controls_score+patholog_score)
        c[np.isnan(c)] = 0.5
        return c

    def BIC(self, X):
        controls_score, patholog_score = self.pdf(X.reshape(-1, 1))
        likelihood = controls_score + patholog_score
        likelihood = -1*np.log(likelihood).sum()
        return 2*likelihood+2*np.log(X.shape[0])


def hscott(x, weights=None):

    IQR = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    A = min(np.nanstd(x, ddof=1), IQR / 1.349)

    if weights is None:
        weights = np.ones(len(x))
    n = float(np.nansum(weights))
    #n = n/sum(~np.isnan(x))

    return 1.059 * A * n ** (-0.2)

def disease_direction(X,y,label='Biomarker',plot_bool=False):
    """
    disease_direction(X,y, [label])
    
    Estimates disease progression direction by comparing empirical CDFs
    in controls and patients
    
    Author: Neil Oxtoby, November 2019
    """
    not_a_number = np.isnan(X) | np.isnan(y)
    not_a_number = not_a_number | (~np.isin(y,[0,1]))
    y_ = y[~not_a_number]
    X_ = X[~not_a_number]
    sorted_idx = X_.argsort(axis=0)
    kde_values = X_.copy()[sorted_idx].reshape(-1,1)
    kde_labels = y_.copy()[sorted_idx]
    bin_counts = np.bincount(y_).astype(float)
    mixture = sum(kde_labels==0)/len(kde_labels) # Prior of being a control
    controls_kde = GaussianKDE(glob_bw="scott", alpha=0.3, diag_cov=False)
    patholog_kde = GaussianKDE(glob_bw="scott", alpha=0.1, diag_cov=False)
    controls_kde.fit(kde_values[kde_labels == 0])
    patholog_kde.fit(kde_values[kde_labels == 1])
    controls_score0 = controls_kde.predict(kde_values)
    patholog_score0 = patholog_kde.predict(kde_values)
    controls_score = controls_score0*mixture
    patholog_score = patholog_score0*(1-mixture)
    ratio = controls_score / (controls_score + patholog_score)
    #* Empirical cumulative distribution: (CDF_patients-CDF_controls) < 0 => disease progression is positive
    cdf_controls = np.cumsum(controls_score)/max(np.cumsum(controls_score))
    cdf_patholog = np.cumsum(patholog_score)/max(np.cumsum(patholog_score))
    cdf_diff = (cdf_patholog - cdf_controls)/(cdf_patholog + cdf_controls)
    disease_dirn = -np.sign(np.nansum(cdf_diff)) #-np.sign(np.mean(cdf_diff))

    if plot_bool:
        f,a=plt.subplots()
        a.plot(kde_values,cdf_controls,label='Controls')
        a.plot(kde_values,cdf_patholog,label='Patients')
        a.legend()
        a.set_title('Disease direction: {0}'.format(disease_dirn))
        a.set_ylabel('Empirical Distribution (CDF)')
        a.set_xlabel(label)
        f.show()

    return disease_dirn
