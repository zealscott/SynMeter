# pylint: disable = W0614, W0401, C0411
# the above errcodes correspond to unused wildcard import, wildcard import, wrong-import-order
# In fact, we can run pylint in cmd and set options like: pylint --disable=Cxxxx,Wxxxx yyyy.py zzzz.py
import argparse
import copy

from pathlib import Path
from loguru import logger
import numpy as np


parser = argparse.ArgumentParser()

# original dataset file 
parser.add_argument("--priv_data", type=str, default="./data/accidential_drug_deaths.csv",
                    help="specify the path of original data file in csv format")

# priv_data_name for use of naming mile-stone files
parser.add_argument("--priv_data_name", type=str, 
help="users must specify it to help mid-way naming and avoid possible mistakings")

# config file which include identifier and binning settings 
parser.add_argument("--config", type=str, default="./config/data.yaml",
                    help="specify the path of config file in yaml format")

# the default number of records is set as 100
parser.add_argument("--n", type=int, default=0, 
                    help="specify the number of records to generate")

# params file which include schema of the original dataset
parser.add_argument("--params", type=str, default="./data/parameters.json",
                    help="specify the path of parameters file in json format")

# datatype file which include the data types of the columns
parser.add_argument("--datatype", type=str, default="./data/column_datatypes.json",
                    help="specify the path of datatype file in json format")

# marginal_config which specify marginal usage method
parser.add_argument("--marginal_config", type=str, default="./config/eps=10.0.yaml",
help="specify the path of marginal config file in yaml format")

# hyper parameter, the num of update iterations
parser.add_argument("--update_iterations", type=int, default=30,
                   help="specify the num of update iterations")

# target path of synthetic dataset
parser.add_argument("--target_path", type=str, default="out.csv",
help="specify the target path of the synthetic dataset")


args = parser.parse_args()
PRIV_DATA = args.priv_data
PRIV_DATA_NAME = args.priv_data_name
CONFIG_DATA = args.config
PARAMS = args.params
DATA_TYPE = args.datatype
MARGINAL_CONFIG = args.marginal_config
UPDATE_ITERATIONS = args.update_iterations
TARGET_PATH = args.target_path

from .data_loader import *
from .postprocessor import RecordPostprocessor
from dpsyn import DPSyn


def main():
    np.random.seed(0)
    np.random.RandomState(0)
    with open(args.config, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)

    # dataloader initialization
    dataloader = DataLoader()
    dataloader.load_data()

    # default method is dpsyn
    method = 'dpsyn'

    n = args.n
    priv_data = args.priv_data
    priv_data_name = args.priv_data_name
   
    syn_data = run_method(config, dataloader, n)
    # if users set the records' num, we denote it in synthetic dataset's name
    if n!=0:
        print("------------------------> now we synthesize a dataset with ", n, "rows")
        syn_data.to_csv(Path(TARGET_PATH), index=False)
    # the default synthetic dataset name when n=0 
    else:
        syn_data.to_csv(Path(TARGET_PATH), index=False)


def run_method(config, dataloader, n):
    parameters = json.loads(Path(args.params).read_text())
    syn_data = None

    # each item in 'runs' specify one dp task with (eps, delta, sensitivity) 
    # as well as a possible 'max_records' value which bounds the dataset's size
    for r in parameters["runs"]:
        # 'max_records_per_individual' is the global sensitivity value of the designed function f
        #  here in the example f is the count, and you may change as you like
        eps, delta, sensitivity = r['epsilon'], r['delta'], r['max_records_per_individual']

        # we import logger in synthesizer.py
        # we import DPSyn which inherits synthesizer 
        logger.info(f'working on eps={eps}, delta={delta}, and sensitivity={sensitivity}')

        # we will use dpsyn to generate a dataset 
        """I guess it helps by displaying the runtime logic below
        1. DPSyn(Synthesizer)
        it got dataloader, eps, delta, sensitivity
        however, Synthesizer is so simple and crude(oh no it initializes the parameters in __init__)
        2. we call synthesizer.synthesize(fixed_n=n) which is written in dpsyn.py
        3. look at synthesize then
            def synthesize(self, fixed_n=0) -> pd.DataFrame:
            # def obtain_consistent_marginals(self, priv_marginal_config, priv_split_method) -> Marginals:
                noisy_marginals = self.obtain_consistent_marginals()
        4. it calls get_noisy_marginals() which is written in synthesizer.py
            # noisy_marginals = self.get_noisy_marginals(priv_marginal_config, priv_split_method)
        5. look at get_noisy_marginals()
            # we firstly generate punctual marginals
            priv_marginal_sets, epss = self.data.generate_marginal_by_config(self.data.private_data, priv_marginal_config)
            # todo: consider fine-tuned noise-adding methods for one-way and two-way respectively?
            # and now we add noises to get noisy marginals
            noisy_marginals = self.anonymize(priv_marginal_sets, epss, priv_split_method)
        6. look at generate_marginal_by_config() which is written in DataLoader.py
            we need config files like 
        e.g.3.
            priv_all_one_way: (or priv_all_two_way)
            total_eps: xxxxx
        7. look at anonymize() which is written in synthesizer.py 
            def anonymize(self, priv_marginal_sets: Dict, epss: Dict, priv_split_method: Dict) -> Marginals:
            noisy_marginals = {}
            for set_key, marginals in priv_marginal_sets.items():
                eps = epss[set_key]
            # noise_type, noise_param = advanced_composition.get_noise(eps, self.delta, self.sensitivity, len(marginals))
                noise_type = priv_split_method[set_key]
            (1)priv_split_method is hard_coded 
            (2) we decide the noise type by advanced_compisition()


        """
        synthesizer = DPSyn(dataloader,update_iterations, eps, delta, sensitivity=1)
        # tmp returns a DataFrame
        syn_data = synthesizer.synthesize(fixed_n=n)
        



    # post-processing generated data, map records with grouped/binned attribute back to original attributes
    print("********************* START POSTPROCESSING ***********************")
    postprocessor = RecordPostprocessor()
    syn_data = postprocessor.post_process(syn_data, args.config, dataloader.decode_mapping)
    logger.info("------------------------>synthetic data post-processed:")
    print(syn_data)

    return syn_data


if __name__ == "__main__":    
    main()