import abc

import numpy as np
import pandas as pd
from loguru import logger

from .data_loader import DataLoader
from . import advanced_composition
from typing import Dict, Tuple


class Synthesizer(object):
    """the class include functions to synthesize noisy marginals
    note that some functions just own a draft which yet to be used in practice
    
    
    """
    # every class can inherit the base class object;
    # abc means Abstract Base Class
    __metaclass__ = abc.ABCMeta
    Marginals = Dict[Tuple[str], np.array]

    def __init__(self, data: DataLoader, update_iterations: int, eps: float, delta: float, sensitivity: int, ratio = None):
        self.data = data
        self.eps = eps
        self.delta = delta
        self.sensitivity = sensitivity
        self.update_iterations = update_iterations
        self.ratio = ratio

    @abc.abstractmethod
    def synthesize(self, fixed_n: int) -> pd.DataFrame:
        pass

    # make sure the synthetic data size does not exceed the max allowed size
    # currently not used
    def synthesize_cutoff(self, submit_data: pd.DataFrame) -> pd.DataFrame:
        if submit_data.shape > 0:
            submit_data.sample()
        return submit_data

    def anonymize(self, priv_marginal_sets: Dict, epss: Dict, priv_split_method: Dict) -> Marginals:
        """the function serves for adding noises
        priv_marginal_sets: Dict[set_key,marginals] where set_key is an key for eps and noise_type
        priv_split_method serves for mapping 'set_key' to 'noise_type' which can be hard coded but currently unused

        """
        noisy_marginals = {}
        # as for now, set_key only havs one value, i.e. "priv_all_two_way"
        for set_key, marginals in priv_marginal_sets.items():
            # for debug about num
            tmp_num = np.mean([np.sum(marginal.values) for marginal_att, marginal in marginals.items()])
            print("**************** help debug ************** num of records from marginal count before adding noise:", tmp_num)

            eps = epss[set_key]
            print("------------------------> now we decide the noise type: ")
            print("considering eps:", eps, ", delta:", self.delta, ", sensitivity:", self.sensitivity,
            ", len of marginals:", len(marginals))
            
            noise_type, noise_param = advanced_composition.get_noise(eps, self.delta, self.sensitivity, len(marginals))
            print("------------------------> noise type:", noise_type)
            print("------------------------> noise parameter:", noise_param)

            # noise_type = priv_split_method[set_key]
            # tip: you can hard code the noise type or let program decide it 

            # the advanced_composition is a python module which provides related noise parameters
            # for instance, as to laplace noises, it computes the reciprocal of laplace scale
            
            if noise_type == 'lap':
                noise_param = 1 / advanced_composition.lap_comp(eps, self.delta, self.sensitivity, len(marginals))
                for marginal_att, marginal in marginals.items():
                    marginal += np.random.laplace(scale=noise_param, size=marginal.shape)
                    noisy_marginals[marginal_att] = marginal
            else:   
                noise_param = advanced_composition.gauss_zcdp(eps, self.delta, self.sensitivity, len(marginals))
                for marginal_att, marginal in marginals.items():
                    noise = np.random.normal(scale=noise_param, size=marginal.shape) 
                    marginal += noise
                    noisy_marginals[marginal_att] = marginal 
            logger.info(f"marginal {set_key} use eps={eps}, noise type:{noise_type}, noise parameter={noise_param}, sensitivity:{self.sensitivity}")
        return noisy_marginals


    def get_noisy_marginals(self, priv_marginal_config, priv_split_method) -> Marginals:
        """instructed by priv_marginal_config, it generate noisy marginals
        generally, priv_marginal_config only includes one/two way and eps,
        e.g.
        priv_all_two_way: 
          total_eps: 990
        
        btw, currently we don't set priv_split method in hard code
      
        """
        # generate_marginal_by_config return Tuple[Dict,Dict]     
        # epss[marginal_key] = marginal_dict['total_eps']
        # marginal_sets[marginal_key] = marginals
        # return marginal_sets, epss
        # we firstly generate punctual marginals
        priv_marginal_sets, epss = self.data.generate_marginal_by_config(self.data.private_data, priv_marginal_config)
        # todo: consider fine-tuned noise-adding methods for one-way and two-way respectively
        # and now we add noises to get noisy marginals
        noisy_marginals = self.anonymize(priv_marginal_sets, epss, priv_split_method)
        
        # ge the difference between marginal and noisy_marginal
        diff_score = []
        for select_marginal in noisy_marginals.keys():
            try:
                diff = noisy_marginals[select_marginal] - priv_marginal_sets["priv_all_two_way"][select_marginal]
            except:
                diff = noisy_marginals[select_marginal] - priv_marginal_sets["priv_all_one_way"][select_marginal]
            score = diff.sum().sum()
            diff_score.append(score)
        print("************* diff score **************")
        print(np.mean(diff_score))
        
        # we delete the original marginals 
        del priv_marginal_sets
        return noisy_marginals
