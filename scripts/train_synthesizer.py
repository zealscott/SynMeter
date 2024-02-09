# this script is used to train the synthesizer
import os
import argparse
from lib.commons import load_config
from lib.info import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="great")
    parser.add_argument("--dataset", "-d", type=str, default="adult")
    parser.add_argument("--cuda", "-c", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    # load template config
    model_config = "exp/{0}/{1}/config.toml".format(args.dataset, args.model)
    config = load_config(os.path.join(ROOT_DIR, model_config))

    # dynamically import model interface
    synthesizer = __import__("synthesizer." + args.model, fromlist=[args.model])
    print("Training {0} on {1}".format(args.model, args.dataset))
    synthesizer.train(config, args.cuda, args.seed)


if __name__ == "__main__":
    main()
