from lib.commons import *
from lib.info import TUNED_PARAMS_PATH
from evaluator.utility.util import split_data_stratify
import numpy as np
from evaluator.utility.cat_boost import train_catboost
from evaluator.utility.tab_transformer import train_tab_transformer
from evaluator.utility.xgb import train_xgb
from evaluator.utility.simple_evaluators import *
from evaluator.utility.query import range_query


def load_data_for_query(data_path, meta_data_path):
    """
    load raw data and column info for marginal query
    """
    data = pd.read_csv(data_path, dtype=object)  # use objective for easily compare
    meta_data = load_json(meta_data_path)

    continue_col_value_dict = {}
    discrete_col_value_dict = {}

    # get discrete column domain
    for col in meta_data["columns"]:
        if col["type"] != "continuous":
            discrete_col_value_dict[col["name"]] = col["i2s"]
        else:
            continue_col_value_dict[col["name"]] = [col["min"], col["max"]]
            # we change the type of column to numeric for range query
            try:
                data[col["name"]] = pd.to_numeric(data[col["name"]])
            except:
                print("{0}, column {1} cannot be converted to numeric".format(data_path, col["name"]))
                continue

    return data, continue_col_value_dict, discrete_col_value_dict


def query_evaluation(args, query_res, n_samples=1000, seed=0, tune=False):
    """
    evaluate the query error for the given dataset, n_way_range = 1, 2, 3
    note that continuous columns are also supported by ramdomly sampling the range
    """
    path_params = args["path_params"]
    real_data_path = path_params["val_data"] if tune else path_params["test_data"]
    real_data, continue_col_value_dict, discrete_col_value_dict = load_data_for_query(
       real_data_path, path_params["meta_data"]
    )
    syn_data, _, _ = load_data_for_query(path_params["out_data"], path_params["meta_data"])

    # sample syn_data to make it have the same number of rows as real_data
    syn_data = syn_data.sample(len(real_data), replace=False, random_state=seed)

    cur_ret = {}
    for n_way_marginal in range(1, 4):
        range_query_error = range_query(
            real_data,
            syn_data,
            discrete_col_value_dict,
            continue_col_value_dict,
            n_way_marginal,
            n_samples,
            seed,
        )
        cur_ret["{0}_way_range".format(n_way_marginal)] = range_query_error

    if tune:
        return cur_ret
    else:
        # add the result to query_res
        if not query_res:
            for metric, value in cur_ret.items():
                query_res[metric] = [value]
        else:
            for metric, value in cur_ret.items():
                query_res[metric].append(value)

        return query_res


def ml_evaluation(config, dataset, cuda, seed, ml_results=[{}, {}], tune=False):
    """
    Evaluate the ml model on real and synthetic data
    train the model with fixed parameters
    when tune = True, never use real_test data
    Evaluator: `xgboost`, `tab_transformer`, `cat_boost`, `lr`, `rf`, `mlp`, `tree`
    """
    improve_reproducibility(seed)
    # prepare data in numpy array
    # real_data: [real_train, real_val, real_test],
    # syn_data: [syn_train, syn_val, real_test]
    real_data, syn_data, task_type = prepare_ml_eval(config, tune)

    n_class = get_n_class(config["path_params"]["meta_data"])

    print("fixed evaluator for ML evaluation")
    syn_results = train_and_test(syn_data, task_type, dataset, n_class, cuda, seed, tune)
    real_results = train_and_test(real_data, task_type, dataset, n_class, cuda, seed, tune)

    if tune:
        return [syn_results, real_results]
    else:
        # add the result to ml_results
        syn_results = add_ml_results(syn_results, ml_results[0])
        real_results = add_ml_results(real_results, ml_results[1])

        return [syn_results, real_results]


def add_ml_results(temp_res, results):
    """add temp_res to results"""
    if not results:
        # {"tvae": {"r2":1, "mse":2}}
        for model, metrics in temp_res.items():
            results[model] = {}
            for metric, value in metrics.items():
                results[model][metric] = [value]
    else:
        for model, metrics in temp_res.items():
            for metric, value in metrics.items():
                results[model][metric].append(value)

    return results


def save_utility_results(ml_results, query_results, utility_result_path):
    syn_ml_results, real_ml_results = ml_results

    for model, metrics in syn_ml_results.items():
        for metric, value in metrics.items():
            syn_ml_results[model][metric] = {}
            syn_ml_results[model][metric]["mean"] = sum(value) / len(value)
            syn_ml_results[model][metric]["std"] = float(np.std(value))

            real_value = real_ml_results[model][metric]
            real_ml_results[model][metric] = {}
            real_ml_results[model][metric]["mean"] = sum(real_value) / len(real_value)
            real_ml_results[model][metric]["std"] = float(np.std(real_value))
    ml_dict = {"synthetic": syn_ml_results, "real": real_ml_results}

    for metric, value in query_results.items():
        query_results[metric] = {}
        query_results[metric]["mean"] = sum(value) / len(value)
        query_results[metric]["std"] = float(np.std(value))

    # save the result with json
    os.makedirs(os.path.dirname(utility_result_path), exist_ok=True)
    with open(utility_result_path, "w") as f:
        json.dump({"ml_performance": ml_dict, "range_query": query_results}, f)

    print("utility result saved to {}".format(utility_result_path))


def prepare_ml_eval(args, tune=False):
    """
    Load real and synthetic data for ML evaluation
    convert the data into numpy array with encoding
    """
    path_params = args["path_params"]
    # load real data
    real_train_data_pd, meta_data, discrete_columns = read_csv(path_params["train_data"], path_params["meta_data"])
    real_val_data_pd, _, _ = read_csv(path_params["val_data"], path_params["meta_data"])
    real_test_data_pd, _, _ = read_csv(path_params["test_data"], path_params["meta_data"])

    # load synthetic data, split into train and val
    syn_data_pd, meta_data, discrete_columns = read_csv(path_params["out_data"], path_params["meta_data"])
    # synthetic data do not need test data, so the size = len(real_train + real_val)
    syn_data_pd = syn_data_pd[: len(real_train_data_pd) + len(real_val_data_pd)]
    syn_train_pd, syn_val_pd = split_data_stratify(data=syn_data_pd, test_size=len(real_val_data_pd))

    # preprocess data, use train+val encoding to transform test data
    real_train, real_val, real_encodings = preprocess(real_train_data_pd, real_val_data_pd, meta_data, discrete_columns)
    real_test = transform_data(real_test_data_pd, real_encodings, meta_data)

    # use real encoding to transform syn data
    syn_train = transform_data(syn_train_pd, real_encodings, meta_data)
    syn_val = transform_data(syn_val_pd, real_encodings, meta_data)

    if tune:
        # ignor real_test for  tuning
        return [real_train, real_val], [syn_train, syn_val], meta_data["task"]
    else:
        return (
            [real_train, real_val, real_test],
            [syn_train, syn_val, real_test],
            meta_data["task"],
        )


def train_and_test(data, task_type, dataset, n_class, cuda, seed, tune=False):
    """
    directly use fiexed parameters to train the model, and test on test data
    """
    result = {}
    if tune:
        # use train and val data for tuning
        train, val = data
        train_data = train
        test = val
    else:
        train, val, test = data
        # combine train and val data since we don't need to tune the model
        train_x = np.concatenate([train[0], val[0]], axis=0)
        train_y = np.concatenate([train[1], val[1]], axis=0)
        train_data = [train_x, train_y]

    # load best lr parameters and fit synthetic data
    print("train lr")
    params_path = TUNED_PARAMS_PATH + "/evaluators/lr/" + "{0}.toml".format(dataset)
    best_lr_params = load_config(params_path)
    _, res = train_lr(best_lr_params, train_data, test, task_type, n_class)
    result["lr"] = res

    # load best rf parameters and fit synthetic data
    params_path = TUNED_PARAMS_PATH + "/evaluators/rf/" + "{0}.toml".format(dataset)
    best_rf_params = load_config(params_path)
    _, res = train_rf(best_rf_params, train_data, test, task_type, n_class)
    result["rf"] = res

    # load best mlp parameters and fit synthetic data
    print("train mlp")
    params_path = TUNED_PARAMS_PATH + "/evaluators/mlp/" + "{0}.toml".format(dataset)
    best_mlp_params = load_config(params_path)
    _, res = train_mlp(best_mlp_params, train_data, test, task_type, n_class)
    result["mlp"] = res

    # load best tree parameters and fit synthetic data
    print("train tree")
    params_path = TUNED_PARAMS_PATH + "/evaluators/tree/" + "{0}.toml".format(dataset)
    best_tree_params = load_config(params_path)
    _, res = train_tree(best_tree_params, train_data, test, task_type, n_class)
    result["tree"] = res

    # load best svm parameters and fit synthetic data
    print("train svm")
    params_path = TUNED_PARAMS_PATH + "/evaluators/svm/" + "{0}.toml".format(dataset)
    best_svm_params = load_config(params_path)
    _, res = train_svm(best_svm_params, train_data, test, task_type, n_class)
    result["svm"] = res

    # load best XGBoost parameters and fit synthetic data
    print("train xgboost")
    params_path = TUNED_PARAMS_PATH + "/evaluators/xgboost/" + "{0}.toml".format(dataset)
    best_xgb_params = load_config(params_path)
    _, res = train_xgb(best_xgb_params, train_data, test, task_type, n_class)
    result["xgboost"] = res

    # load best tab_transformer parameters and fit synthetic data
    print("train tab_transformer")
    params_path = TUNED_PARAMS_PATH + "/evaluators/tab_transformer/" + "{0}.toml".format(dataset)
    best_transformer_params = load_config(params_path)
    _, res = train_tab_transformer(best_transformer_params, train_data, test, task_type, n_class, "cuda:" + cuda)
    result["tab_transformer"] = res

    if tune:
        # ignor cat_boost for fast tuning
        return result

    # load best cat_boost parameters and fit synthetic data
    print("train cat_boost")
    params_path = TUNED_PARAMS_PATH + "/evaluators/cat_boost/" + "{0}.toml".format(dataset)
    best_catboost_params = load_config(params_path)
    _, res = train_catboost(best_catboost_params, train_data, test, task_type, n_class)
    result["cat_boost"] = res

    return result
