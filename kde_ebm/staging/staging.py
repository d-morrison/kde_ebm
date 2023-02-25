from ..mixture_model import get_prob_mat
import numpy as np

def ebm_staging(x,mixtures,samples, verbose = False):
    """
    Given a trained EBM (mixture_models,mcmc_samples), and correctly-formatted data, stage the data
    NOTE: To use CV-EBMs, you'll need to call this for each fold, then combine.
    Author: Neil P Oxtoby, UCL, September 2018
    """
    if type(mixtures[0]) is list:
        #* List of mixture models from cross-validation / bootstrapping
        n_cv = len(mixtures)
        prob_mat = []
        stages = []
        stage_likelihoods = []
        stages_expected = []
        for k in range(n_cv):
            #* Stage the data
            prob_mat.append(get_prob_mat(x, mixtures[k]))
            stages_k, stage_likelihoods_k = samples[k][0].stage_data(prob_mat[k])
            stages.append(stages_k)
            stage_likelihoods.append(stage_likelihoods_k)
            #* Average (expectation value) stage
            stages_expected_k = np.ndarray(shape=stages_k.shape)
            for kk in range(stages_expected_k.shape[0]):
                stages_expected_k[kk] = np.sum(stage_likelihoods_k[kk,:]*np.arange(1,stage_likelihoods_k.shape[1]+1,1))/np.sum(stage_likelihoods_k[kk,:]) - 1
            stages_expected.append(stages_expected_k)
    else:
        if verbose:
            print("type(mixtures[0]) is not list")
        #* Stage the data
        prob_mat = get_prob_mat(x, mixtures)
        if type(samples[0]) is list:
            if verbose:
                print("type(samples[0]) is list")
            n_bs = len(samples)
            stages = []
            stage_likelihoods = []
            stages_expected = []
            for k in range(n_bs):
                #* Stage the data
                stages_k, stage_likelihoods_k = samples[k][0].stage_data(prob_mat)
                stages.append(stages_k)
                stage_likelihoods.append(stage_likelihoods_k)
                #* Average (expectation value) stage
                stages_expected_k = np.ndarray(shape=stages_k.shape)
                for kk in range(stages_expected_k.shape[0]):
                    stages_expected_k[kk] = np.sum(stage_likelihoods_k[kk,:]*np.arange(1,stage_likelihoods_k.shape[1]+1,1))/np.sum(stage_likelihoods_k[kk,:]) - 1
                stages_expected.append(stages_expected_k)
        else:
            if verbose:
                print("type(samples[0]) is not list")
            stages, stage_likelihoods = samples[0].stage_data(prob_mat)
            #* Average (expectation value) stage
            stages_expected = np.ndarray(shape=stages.shape)
            for k in range(stages_expected.shape[0]):
                stages_expected[k] = np.sum(stage_likelihoods[k,:]*np.arange(1,stage_likelihoods.shape[1]+1,1))/np.sum(stage_likelihoods[k,:]) - 1
    # #* Average (expectation value) stage
    # stages_expected_n = np.sum(stage_likelihoods,axis=1)
    # stages_expected_ = np.average(stage_likelihoods_long_ml,axis=1,weights=np.arange(1,stage_likelihoods_long_ml.shape[1]+1,1))
    # stages_expected_ = stages_expected_/stages_expected_n
    
    return prob_mat, stages, stage_likelihoods, stages_expected