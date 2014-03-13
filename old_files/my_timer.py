from time import time
from functools import update_wrapper

class getfullargspec(object):
    "A quick and dirty replacement for getfullargspec for Python 2.X"
    def __init__(self, f):
        self.args, self.varargs, self.varkw, self.defaults = \
            inspect.getargspec(f)
        self.kwonlyargs = []
        self.kwonlydefaults = None
    def __iter__(self):
        yield self.args
        yield self.varargs
        yield self.varkw
        yield self.defaults
def get_init(cls):
    return cls.__init__.im_func

def time_it(fn):
    # Decorator to time function arguments of a class, return them in a
    # dictionary of times
    def timed(self, *args, **kwargs):
        time0 = time()
        try:
            out = fn(self,*args,**kwargs)
            self.times[fn.func_name] = time() - time0
            return out

        except Exception as e:
            self.times[fn.func_name] = time() - time0
            raise e

    return update_wrapper(timed, fn)
