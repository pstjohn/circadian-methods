# Function to assist with the parallel evaluation of class methods.
# use 
# >>> from CommonFiles.Futures import map_
#
# then use 
#
# >>> if __name__ == "__main__":
# >>>     results = map_(func, args)
#
# to get the results in serial or parallel, depending on if
# $ python -m scoop func.py
# was used to call the function in parallel.


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try: func = cls.__dict__[func_name]
        except KeyError: pass
        else: break
        return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# Conditional SCOOP import. Use map_ instead of map for serial or
# parallel evaluation
try:
    from scoop.futures import map as map_
except ImportError:
    map_ = map

