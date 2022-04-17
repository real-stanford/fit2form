import ray
from tqdm import tqdm
import os
from glob import glob


def glob_category(root_dir, category_file, glob_pattern):
    # - imprint dataset
    paths = list()
    with open(category_file, "r") as f:
        for cat_name in f:
            paths += glob(
                os.path.join(root_dir, cat_name.strip(), glob_pattern), recursive=True)
    return paths


def tqdm_remote_get(task_handles, desc=None, max_results=None):
    results = []
    if max_results is None:
        max_results = len(task_handles)
    with tqdm(total=max_results, desc=desc, dynamic_ncols=True) as pbar:
        pbar.update(0)
        while len(task_handles) > 0:
            finished_tasks, task_handles = ray.wait(task_handles)
            results.extend(finished_tasks)
            pbar.update(len(finished_tasks))
            if len(results) > max_results:
                break
    return ray.get(results)
