import pandas as pd
import os
from lib.commons import load_json
import numpy as np
import argparse
import scipy.stats as ss
from lib.info import ROOT_DIR

not_available = "-"


def get_ml_res(exp, measures=["rmse", "f1"]):
    ret = {}
    for evaluator, res in exp.items():
        for metric, values in res.items():
            if metric in measures:
                ret[evaluator] = values["mean"]
    return ret


def cal_affinity(syn_res, real_res):
    # calculate the mean affinity between synthetic and real data
    ret = []
    for evaluator, _ in syn_res.items():
        # relative error
        ret.append(abs(syn_res[evaluator] - real_res[evaluator]) / real_res[evaluator])

    return sum(ret) / len(ret)


def exp_ml_affinity_results(measures=["rmse", "f1"]):
    df = pd.DataFrame(columns=["method"] + DATASETS)

    # first, store the real results
    real_res = {}
    for data in DATASETS:
        res_dict = load_json(os.path.join(ROOT_DIR, "exp/{0}/{1}/utility_result.json".format(data, "naive")))
        ml_res = res_dict["ml_performance"]
        res = get_ml_res(ml_res["real"], measures)
        real_res[data] = res

    all_res = {}
    # second, compute the affinity for each method
    for algo in algorithms:
        algo_res = []
        for data in DATASETS:
            if "great" in algo and data == "news":
                res = "-"
                algo_res.append(res)
                continue
            res_dict = load_json(os.path.join(ROOT_DIR, "exp/{0}/{1}/utility_result.json".format(data, algo)))
            ml_res = res_dict["ml_performance"]
            syn_res = get_ml_res(ml_res["synthetic"], measures)
            aff_score = cal_affinity(syn_res, real_res[data])
            res = "{0:.3f}".format(aff_score)
            algo_res.append(res)
        df.loc[len(df)] = [algo] + algo_res
        all_res[algo] = algo_res
    
    # calculate the improvement
    alog = "improvement"
    res = []
    improvement = []
    for i in range(len(DATASETS)):
        if "great" in algo and DATASETS[i] == "news":
            res.append(not_available)
            continue
        ori_res = all_res[algorithms[0]]
        new_res = all_res[algorithms[1]]
        imp = (float(ori_res[i]) - float(new_res[i]))/float(ori_res[i])
        # use percentage
        imp = imp * 100
        improvement.append(imp)
        res.append("{0:.3f}%".format(imp))
    df.loc[len(df)] = [alog] + res
    print("average improvement: {0:.3f}%".format(np.mean(improvement)))
    return df,np.mean(improvement)


def exp_query_results(n_way):
    query_df = pd.DataFrame(columns=["method"] + DATASETS)

    res_all = {}
    for algo in algorithms:
        algo_query_res = []
        res_all[algo] = []
        for data in DATASETS:
            if "great" in algo and data == "news":
                res = "-"
                algo_query_res.append(res)
                res_all[algo].append(res)
                continue
            res_dict = load_json(os.path.join(ROOT_DIR, "exp/{0}/{1}/utility_result.json".format(data, algo)))
            # append query result
            query_res = res_dict["range_query"]
            if f"{n_way}_way_range" in query_res:
                query_res = query_res[f"{n_way}_way_range"]
                res = "{0:.3f}({1:.3f})".format(query_res["mean"], query_res["std"])
            else:
                res = not_available
            algo_query_res.append(res)
            res_all[algo].append(query_res["mean"])
        query_df.loc[len(query_df)] = [algo] + algo_query_res

    #  calculate the improvement
    alog = "improvement"
    res = []
    improvement = []
    for i in range(len(DATASETS)):
        if "great" in algo and DATASETS[i] == "news":
            res.append(not_available)
            continue
        ori_res = res_all[algorithms[0]][i]
        new_res = res_all[algorithms[1]][i]
        imp = (float(ori_res) - float(new_res))/float(ori_res)
        # use percentage
        imp = imp * 100
        improvement.append(imp)
        if DATASETS[i] == "adult":
            print("ori_res: {0:.10f}, new_res: {1:.10f}, imp: {2:.10f}".format(ori_res, new_res, imp))
        res.append("{0:.3f}%".format(imp))
    query_df.loc[len(query_df)] = [alog] + res
    print("average improvement: {0:.3f}%".format(np.mean(improvement)))
    
    return query_df,np.mean(improvement)


def exp_fidelity_results(file = "fidelity_result.json"):
    # column-wise error
    cat_dist_df = pd.DataFrame(columns=["method"] + DATASETS)
    num_dist_df = pd.DataFrame(columns=["method"] + DATASETS)
    # pairwise error
    cat_cat_dist_df = pd.DataFrame(columns=["method"] + DATASETS)
    num_cat_dist_df = pd.DataFrame(columns=["method"] + DATASETS)
    num_num_dist_df = pd.DataFrame(columns=["method"] + DATASETS)

    all_res = {}
    for algo in algorithms:
        all_res[algo] = {}
        algo_cat_dist_res = []
        algo_num_dist_res = []
        algo_cat_cat_dist_res = []
        algo_num_cat_dist_res = []
        algo_num_num_dist_res = []
        for data in DATASETS:
            if "great" in algo and data == "news":
                algo_cat_dist_res.append("-")
                algo_num_dist_res.append("-")
                algo_cat_cat_dist_res.append("-")
                algo_num_cat_dist_res.append("-")
                algo_num_num_dist_res.append("-")
                continue
            res_dict = load_json(os.path.join(ROOT_DIR, "exp/{0}/{1}/{2}".format(data, algo, file)))

            # append cat_error
            if "cat_error" not in res_dict:
                res = not_available
            else:
                cat_dist_res = res_dict["cat_error"]
                res = "{0:.3f}({1:.3f})".format(cat_dist_res["mean"], cat_dist_res["std"])
                if "cat_error" not in all_res[algo]:
                    all_res[algo]["cat_error"] = [cat_dist_res["mean"]]
                else:
                    all_res[algo]["cat_error"].append(cat_dist_res["mean"])
            algo_cat_dist_res.append(res)

            # append cont_error
            if "cont_error" not in res_dict:
                res = not_available
            else:
                num_dist_res = res_dict["cont_error"]
                res = "{0:.3f}({1:.3f})".format(num_dist_res["mean"], num_dist_res["std"])
                if "cont_error" not in all_res[algo]:
                    all_res[algo]["cont_error"] = [num_dist_res["mean"]]
                else:
                    all_res[algo]["cont_error"].append(num_dist_res["mean"])
            algo_num_dist_res.append(res)

            # append cat_cat_error
            if "cat_cat_error" not in res_dict:
                res = not_available
            else:
                cat_cat_dist_res = res_dict["cat_cat_error"]
                res = "{0:.3f}({1:.3f})".format(cat_cat_dist_res["mean"], cat_cat_dist_res["std"])
                if "cat_cat_error" not in all_res[algo]:
                    all_res[algo]["cat_cat_error"] = [cat_cat_dist_res["mean"]]
                else:
                    all_res[algo]["cat_cat_error"].append(cat_cat_dist_res["mean"])
            algo_cat_cat_dist_res.append(res)

            # append cat_cont_error
            if "cat_cont_error" not in res_dict:
                res = not_available
            else:
                num_cat_dist_res = res_dict["cat_cont_error"]
                res = "{0:.3f}({1:.3f})".format(num_cat_dist_res["mean"], num_cat_dist_res["std"])
                if "cat_cont_error" not in all_res[algo]:
                    all_res[algo]["cat_cont_error"] = [num_cat_dist_res["mean"]]
                else:
                    all_res[algo]["cat_cont_error"].append(num_cat_dist_res["mean"])
            algo_num_cat_dist_res.append(res)

            # append cont_cont_error
            if "cont_cont_error" not in res_dict:
                res = not_available
            else:
                num_num_dist_res = res_dict["cont_cont_error"]
                res = "{0:.3f}({1:.3f})".format(num_num_dist_res["mean"], num_num_dist_res["std"])
                if "cont_cont_error" not in all_res[algo]:
                    all_res[algo]["cont_cont_error"] = [num_num_dist_res["mean"]]
                else:
                    all_res[algo]["cont_cont_error"].append(num_num_dist_res["mean"])
            algo_num_num_dist_res.append(res)

        cat_dist_df.loc[len(cat_dist_df)] = [algo] + algo_cat_dist_res
        num_dist_df.loc[len(num_dist_df)] = [algo] + algo_num_dist_res

        cat_cat_dist_df.loc[len(cat_cat_dist_df)] = [algo] + algo_cat_cat_dist_res
        num_cat_dist_df.loc[len(num_cat_dist_df)] = [algo] + algo_num_cat_dist_res
        num_num_dist_df.loc[len(num_num_dist_df)] = [algo] + algo_num_num_dist_res

    types = ["cat_error", "cont_error", "cat_cat_error", "cat_cont_error", "cont_cont_error"]
    
    for error in types:
        # calculate the improvement
        alog = "improvement"
        improvement = []
        ori_res = all_res[algorithms[0]][error]
        new_res = all_res[algorithms[1]][error]
        for i in range(len(ori_res)):
            imp = (float(ori_res[i]) - float(new_res[i]))/float(ori_res[i])
            # use percentage
            imp = imp * 100
            improvement.append(imp)
    print("average improvement: {0:.3f}%".format(np.mean(improvement)))
    
    
    return (
        cat_dist_df,
        num_dist_df,
        cat_cat_dist_df,
        num_cat_dist_df,
        num_num_dist_df,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", "-md", default="tablediffusion")
    parser.add_argument("--utility", "-u", action="store_true", default=False)
    parser.add_argument("--fidelity_test", "-f", action="store_true", default=False)
    parser.add_argument("--fidelity_train", "-ftrain", action="store_true", default=False)

    args = parser.parse_args()

    global algorithms
    algorithms = [args.model + "_ori", args.model + "_eps_1"]
    # algorithms = [args.model + "_ori", args.model]

    global DATASETS
    DATASETS = ["adult","shoppers","phishing","magic","faults","bean","obesity","robot","abalone","news", "insurance","wine"]
    # DATASETS = ["adult","phishing","magic","faults","bean","obesity","robot","abalone", "insurance","wine"]
    
    if args.utility:
        measures = ["rmse", "f1"]
        affinity_df,ml_improv = exp_ml_affinity_results(measures)
        print("=" * 10 + "ml affinity results" + "=" * 10)
        print(affinity_df)
        
        n_way = 3
        query_df,query_improv = exp_query_results(n_way)
        print("=" * 10 + f"{n_way}-way range query" + "=" * 10)
        print(query_df)
        
        print("=" * 10 + "average improvement" + "=" * 10)
        print("average ml + query improvement: {0:.3f}%".format((ml_improv+query_improv)/2))

    if args.fidelity_test:
        (
            cat_dist_df,
            num_dist_df,
            cat_cat_dist_df,
            num_cat_dist_df,
            num_num_dist_df,
        ) = exp_fidelity_results()
        print("=" * 10 + "cat error" + "=" * 10)
        print(cat_dist_df)
        print("=" * 10 + "num error" + "=" * 10)
        print(num_dist_df)
        print("=" * 10 + "cat_cat error" + "=" * 10)
        print(cat_cat_dist_df)
        print("=" * 10 + "cat-num error" + "=" * 10)
        print(num_cat_dist_df)
        print("=" * 10 + "num-num error" + "=" * 10)
        print(num_num_dist_df)

    if args.fidelity_train:
        (
            cat_dist_df,
            num_dist_df,
            cat_cat_dist_df,
            num_cat_dist_df,
            num_num_dist_df,
        ) = exp_fidelity_results("fidelity_train_result.json")
        print("=" * 10 + "cat error" + "=" * 10)
        print(cat_dist_df)
        print("=" * 10 + "num error" + "=" * 10)
        print(num_dist_df)
        print("=" * 10 + "cat_cat error" + "=" * 10)
        print(cat_cat_dist_df)
        print("=" * 10 + "cat-num error" + "=" * 10)
        print(num_cat_dist_df)
        print("=" * 10 + "num-num error" + "=" * 10)
        print(num_num_dist_df)

if __name__ == "__main__":
    main()
