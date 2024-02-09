import copy
import multiprocessing as mp
from typing import List, Tuple, Dict, KeysView

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from numpy import linalg as LA

from .consistent import Consistenter
from .record_synthesizer import RecordSynthesizer
from .view import View
from .synthesizer import Synthesizer


class DPSyn(Synthesizer):
    """Note that it inherits the class Synthesizer,
    which already has the following attributes :
    (data: DataLoader, eps, delta, sensitivity) initialized

    """

    synthesized_df = None

    # the magic value is set empirically and users may change in command lines
    update_iterations = 30

    attrs_view_dict = {}
    onehot_view_dict = {}

    attr_list = []
    domain_list = []
    attr_index_map = {}

    # despite phthon variables can be used without claiming its type, we import typing to ensure robustness
    Attrs = List[str]
    Domains = np.ndarray
    Marginals = Dict[Tuple[str], np.array]
    Clusters = Dict[Tuple[str], List[Tuple[str]]]
    d = None

    def obtain_consistent_marginals(
        self, priv_marginal_config, priv_split_method
    ) -> Marginals:
        """marginals are specified by a dict from attribute tuples to frequency (pandas) tables
        first obtain noisy marginals and make sure they are consistent

        """

        # get_noisy_marginals() is in synthesizer.py
        # which first calls generate_..._by_config(), and computes on priv_data to return marginal_sets, epss
        # (note that 'marginal_key' could be 'priv_all_one_way' or 'priv_all_two_way')
        # later it calls anonymize() which add noises to marginals
        # (what decides noises is 'priv_split_method')
        # priv_split_method[set_key]='lap' or....
        # Step 1: generate noisy marginals
        noisy_marginals = self.get_noisy_marginals(
            priv_marginal_config, priv_split_method
        )

        # since calculated on noisy marginals
        # we use mean function to estimate the number of synthesized records
        num_synthesize_records = (
            np.mean([np.sum(x.values) for _, x in noisy_marginals.items()])
            .round()
            .astype(int)
        )
        print(
            "------------------------> now we get the estimate of records' num by averaging from nosiy marginals:",
            num_synthesize_records,
        )

        # the list of all attributes' name(str)  except the identifier attribute
        self.attr_list = self.data.obtain_attrs()
        # domain_list is an array recording the count of each attribute's candidate values
        self.domain_list = np.array(
            [len(self.data.encode_schema[att]) for att in self.attr_list]
        )

        # map the attribute str to its index in attr_list, for possible use
        # use enumerate to return Tuple(index, element)
        self.attr_index_map = {att: att_i for att_i, att in enumerate(self.attr_list)}

        # views are wrappers of marginals with additional functions for consistency
        # Step 2: create some data structures
        noisy_onehot_view_dict, noisy_attr_view_dict = self.construct_views(
            noisy_marginals
        )

        # all_views is one-hot to view dict, views_dict is attribute to view dict
        # they have different format to satisfy the needs of consistenter and synthesiser
        # to fit in code when we do not have public things to utilize
        pub_onehot_view_dict = noisy_onehot_view_dict
        pub_attr_view_dict = noisy_attr_view_dict

        self.onehot_view_dict, self.attrs_view_dict = self.normalize_views(
            pub_onehot_view_dict,
            pub_attr_view_dict,
            noisy_attr_view_dict,
            self.attr_index_map,
            num_synthesize_records,
        )

        # consist the noisy marginals to submit to some rules
        consistenter = Consistenter(self.onehot_view_dict, self.domain_list)
        consistenter.consist_views()

        # consistenter uses unnormalized counts;
        # after consistency, synthesizer uses normalized counts
        for _, view in self.onehot_view_dict.items():
            view.count /= sum(view.count)

        return noisy_marginals, num_synthesize_records

    # in experiment.py, tmp = synthesizer.synthesize(fixed_n=n)
    # in below function, we call synthesize_records()
    # it further utilize the lib function in record_synthesizer.py

    def train(self):
        """
        privsyn is a non-parametric differentially private synthesizer
        the training process is to obtain noisy marginals and make sure they are consistent
        """
        if self.ratio != None:
            # devide the eps into two parts
            one_way_eps = self.eps * self.ratio
            two_way_eps = self.eps * (1 - self.ratio)
            priv_marginal_config = {"priv_all_two_way": {"total_eps": two_way_eps},"priv_all_one_way": {"total_eps": one_way_eps}}
        else:
            priv_marginal_config = {"priv_all_two_way": {"total_eps": self.eps},"priv_all_one_way": {"total_eps": self.eps}}
        
        priv_split_method = {}

        # step1: get noisy marginals and make sure they are consistent
        noisy_marginals, num_records = self.obtain_consistent_marginals(
            priv_marginal_config, priv_split_method
        )

        self.num_records = num_records

    def synthesize(self, num_records=0) -> pd.DataFrame:
        """synthesize a DataFrame in fixed_n size if denoted n!=0"""
        if num_records != 0:
            self.num_records = num_records
        # TODO: just based on the marginals to synthesize records
        # if in need, we can find clusters for synthesize; a cluster is a set of marginals closely connected
        # here we do not cluster and use all marginals as a single cluster
        clusters = self.cluster(self.attrs_view_dict)
        attrs = self.attr_list
        domains = self.domain_list
        print("------------------------> attributes: ")
        print(attrs)
        print("------------------------> domains: ")
        print(domains)
        print("------------------------> cluseters: ")
        print(clusters)
        print("********************* START SYNTHESIZING RECORDS ********************")

        self.synthesize_records(attrs, domains, clusters, self.num_records)
        print("------------------------> synthetic dataframe before postprocessing: ")
        print(self.synthesized_df)
        return self.synthesized_df

    #  we have a graph where nodes represent attributes and edges represent marginals,
    #  it helps in terms of running time and accuracy if we do it cluster by cluster
    def synthesize_records(
        self,
        attrs: Attrs,
        domains: Domains,
        clusters: Clusters,
        num_synthesize_records: int,
    ):
        print("------------------------> num of synthesized records: ")
        print(num_synthesize_records)
        for cluster_attrs, list_marginal_attrs in clusters.items():
            logger.info("synthesizing for %s" % (cluster_attrs,))

            # singleton_views = {attr: self.attr_view_dict[frozenset([attr])] for attr in attrs}
            singleton_views = {}
            for cur_attrs, view in self.attrs_view_dict.items():
                if len(cur_attrs) == 1:
                    # get the element from frozen set
                    cur_attrs = list(cur_attrs)[0]
                    singleton_views[cur_attrs] = view

            synthesizer = RecordSynthesizer(attrs, domains, num_synthesize_records)
            synthesizer.initialize_records(
                list_marginal_attrs, method = "singleton", singleton_views=singleton_views
            )
            attrs_index_map = {
                attrs: index for index, attrs in enumerate(list_marginal_attrs)
            }

            for update_iteration in range(self.update_iterations):
                logger.info("update round: %d" % (update_iteration,))

                synthesizer.update_alpha(update_iteration)
                sorted_error_attrs = synthesizer.update_order(
                    update_iteration, self.attrs_view_dict, list_marginal_attrs
                )

                for attrs in sorted_error_attrs:
                    attrs_i = attrs_index_map[attrs]
                    synthesizer.update_records(self.attrs_view_dict[attrs], update_iteration,attrs)
            if self.synthesized_df is None:
                self.synthesized_df = synthesizer.df
            else:
                self.synthesized_df.loc[:, cluster_attrs] = synthesizer.df.loc[
                    :, cluster_attrs
                ]

    @staticmethod
    def calculate_l1_errors(records, target_marginals, attrs_view_dict):
        l1_T_Ms = []
        l1_T_Ss = []
        l1_M_Ss = []

        for cur_attrs, target_marginal_pd in target_marginals.items():
            view = attrs_view_dict[cur_attrs]
            syn_marginal = view.count_records_general(records)
            target_marginal = target_marginal_pd.values.flatten()

            T = target_marginal / np.sum(target_marginal)
            M = view.count
            S = syn_marginal / np.sum(syn_marginal)

            l1_T_Ms.append(LA.norm(T - M, 1))
            l1_T_Ss.append(LA.norm(T - S, 1))
            l1_M_Ss.append(LA.norm(M - S, 1))

        return np.mean(l1_T_Ms), np.mean(l1_T_Ss), np.mean(l1_M_Ss)

    @staticmethod
    def normalize_views(
        pub_onehot_view_dict: Dict,
        pub_attr_view_dict,
        noisy_view_dict,
        attr_index_map,
        num_synthesize_records,
    ) -> Tuple[Dict, Dict]:
        pub_weight = 0.00
        noisy_weight = 1 - pub_weight

        views_dict = pub_attr_view_dict
        onehot_view_dict = pub_onehot_view_dict
        for view_att, view in noisy_view_dict.items():
            if view_att in views_dict:
                views_dict[view_att].count = (
                    pub_weight * pub_attr_view_dict[view_att].count
                    + noisy_weight * view.count
                )
                views_dict[view_att].weight_coeff = (
                    pub_weight * pub_attr_view_dict[view_att].weight_coeff
                    + noisy_weight * view.weight_coeff
                )
            else:
                views_dict[view_att] = view
                view_onehot = DPSyn.one_hot(view_att, attr_index_map)
                onehot_view_dict[tuple(view_onehot)] = view
        return onehot_view_dict, views_dict

    @staticmethod
    def obtain_singleton_views(attrs_view_dict):
        singleton_views = {}
        for cur_attrs, view in attrs_view_dict.items():
            # other use
            # puma and year won't be there because they only appear together (size=2)
            if len(cur_attrs) == 1:
                singleton_views[cur_attrs] = view
        return singleton_views

    def construct_views(self, marginals: Marginals) -> Tuple[Dict, Dict]:
        """construct views for each marginal item,
        return 2 dictionaries, onehot2view and attr2view

        """
        onehot_view_dict = {}
        attr_view_dict = {}

        for marginal_att, marginal_value in marginals.items():
            # since one_hot is @staticmethod, we can call it by DPSyn.one_hot
            # return value is an array marked
            view_onehot = DPSyn.one_hot(marginal_att, self.attr_index_map)

            # domain_list is an array recording the count of each attribute's candidate values
            view = View(view_onehot, self.domain_list)

            # we use flatten to create a one-dimension array which serves for when the marginal is two-way
            view.count = marginal_value.values.flatten()

            # we create two dictionaries to map ... to view
            onehot_view_dict[tuple(view_onehot)] = view
            attr_view_dict[marginal_att] = view

            # obviously if things go well, it should match
            if not len(view.count) == view.domain_size:
                raise Exception("no match")

        return onehot_view_dict, attr_view_dict

    def log_result(self, result):
        self.d.append(result)

    @staticmethod
    def build_attr_set(attrs: KeysView[Tuple[str]]) -> Tuple[str]:
        attrs_set = set()

        for attr in attrs:
            attrs_set.update(attr)

        return tuple(attrs_set)

    # simple clustering: just build the data structure; not doing any clustering
    def cluster(self, marginals: Marginals) -> Clusters:
        clusters = {}
        keys = []
        for marginal_attrs, _ in marginals.items():
            keys.append(marginal_attrs)

        clusters[DPSyn.build_attr_set(marginals.keys())] = keys
        return clusters

    @staticmethod
    def one_hot(cur_att, attr_index_map):
        # it marks the attributes included in cur_attr by one-hot way in a len=attr_index_map array
        # return value is an array marked
        cur_view_key = [0] * len(attr_index_map)
        for attr in cur_att:
            cur_view_key[attr_index_map[attr]] = 1
        return cur_view_key
