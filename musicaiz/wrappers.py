import multiprocessing
from itertools import repeat
import time
from typing import Union, List, Any
from pathlib import Path


from musicaiz import utils


def multiprocess_path(
    func,
    path: Union[List[str], Path],
    args: Union[List[Any], None] = None,
    n_jobs=None
) -> str:
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_jobs)

    if isinstance(path, list):
        files_str = path
    else:
        files_str = utils.get_list_files_path(path)

    if args is not None:
        args_it = [repeat(arg) for arg in args]
        results = pool.starmap(
            func,
            zip(files_str, args_it)
        )
    else:
        results = pool.starmap(
            func,
            zip(files_str)
        )
    pool.close()
    pool.join()
    return results
  
  
def timeis(func):
  
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
          
        print(f"Processing time for method {func.__name__}: {end-start} sec")
        return result
    return wrap