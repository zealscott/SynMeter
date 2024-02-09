import copy
from loguru import logger

import numpy as np
from .view import View
"""this file serves for make sure the noisy marginals are consistent, 
i.e., with the same count, submit to the dependency relationship, non-negative counts

"""

class Consistenter:
    class SubsetWithDependency:
        def __init__(self, attributes_set):
            self.attributes_set = attributes_set
            # a set of tuples this object depends on
            self.dependency = set()

    def __init__(self, views, num_categories):
        self.views = views
        self.num_categories = num_categories
        self.iterations = 30

    def compute_dependency(self):
        subsets_with_dependency = {}
        ret_subsets = {}

        for key, view in self.views.items():
            new_subset = self.SubsetWithDependency(view.attributes_set)
            subsets_temp = copy.deepcopy(subsets_with_dependency)

            for subset_key, subset_value in subsets_temp.items():
                attributes_intersection = subset_value.attributes_set & view.attributes_set

                if attributes_intersection:
                    if tuple(attributes_intersection) not in subsets_with_dependency:
                        intersection_subset = self.SubsetWithDependency(attributes_intersection)
                        subsets_with_dependency[tuple(attributes_intersection)] = intersection_subset

                    if not tuple(attributes_intersection) == subset_key:
                        subsets_with_dependency[subset_key].dependency.add(tuple(attributes_intersection))
                    new_subset.dependency.add(tuple(attributes_intersection))

            subsets_with_dependency[tuple(view.attributes_set)] = new_subset

        for subset_key, subset_value in subsets_with_dependency.items():
            if len(subset_key) == 1:
                subset_value.dependency = set()

            ret_subsets[subset_key] = subset_value

        return subsets_with_dependency

    def consist_views(self):
        def find_subset_without_dependency():
            for key, subset in subsets_with_dependency_temp.items():
                if not subset.dependency:
                    return key, subset

            return None, None

        def find_views_containing_target(target):
            result = []

            for key, view in self.views.items():
                if target <= view.attributes_set:
                    result.append(view)

            return result

        # current strategy: if two views do not agree on the levels: v1: 4*2, v2: 2*4, then consist on 2*2
        def consist_on_subset(target):
            target_views = find_views_containing_target(target)

            common_view_indicator = np.zeros(self.num_categories.shape[0])
            for index in target:
                common_view_indicator[index] = 1

            common_view = View(common_view_indicator, self.num_categories)
            common_view.initialize_consist_parameters(len(target_views))

            for index, view in enumerate(target_views):
                common_view.project_from_bigger_view(view, index)

            common_view.calculate_delta()
            # if np.sum(np.absolute(common_view.delta)) > 1000:
            #     print(common_view.attr_one_hot)
            #     print(np.sum(np.absolute(common_view.delta)))
            if np.sum(np.absolute(common_view.delta)) > 1e-3:
                for index, view in enumerate(target_views):
                    view.update_view(common_view, index)

        def remove_subset_from_dependency(target):
            for _, subset in subsets_with_dependency_temp.items():
                if tuple(target.attributes_set) in subset.dependency:
                    subset.dependency.remove(tuple(target.attributes_set))

        # calculate necessary variables
        for key, view in self.views.items():
            view.calculate_tuple_key()
            view.generate_attributes_index_set()
            view.sum = np.sum(view.count)

        # calculate the dependency relationship
        subsets_with_dependency = self.compute_dependency()
        logger.debug("dependency computed")

        # ripple steps needs several iterations
        # for i in range(self.iterations):
        non_negativity = True
        iterations = 0

        while non_negativity and iterations < self.iterations:
            # first make sure summation are the same
            consist_on_subset(set())

            for key, view in self.views.items():
                view.sum = np.sum(view.count)

            subsets_with_dependency_temp = copy.deepcopy(subsets_with_dependency)

            while len(subsets_with_dependency_temp) > 0:
                key, subset = find_subset_without_dependency()

                if not subset:
                    break

                consist_on_subset(subset.attributes_set)
                remove_subset_from_dependency(subset)
                subsets_with_dependency_temp.pop(key, None)

            logger.debug("consist finish")

            nonneg_view_count = 0

            for key, view in self.views.items():
                if (view.count < 0.0).any():
                    view.non_negativity()
                    view.sum = np.sum(view.count)
                else:
                    nonneg_view_count += 1

                if nonneg_view_count == len(self.views):
                    logger.info("finish in %s round" % (iterations,))
                    non_negativity = False

            iterations += 1

            logger.debug("non-negativity finish")

        # calculate normalized count
        for key, view in self.views.items():
            view.sum = np.sum(view.count)
            view.normalize_count = view.count if view.sum <= 0 else view.count / view.sum
