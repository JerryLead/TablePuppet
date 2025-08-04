import ray

def ray_group_call(workers, func_name, *args, **kwargs):
    calls = [getattr(w, func_name).remote(*args, **kwargs) for w in workers]
    return ray.get(calls)

def ray_group_call_multiargs(workers, func_name, args_list):
    calls = [getattr(w, func_name).remote(*params)
             for w, params in zip(workers, args_list)]
    return ray.get(calls)