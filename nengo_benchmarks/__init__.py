all_benchmarks = {}


def register(name):
    def register_with_name(cls):
        all_benchmarks[name] = cls
        return cls

    return register_with_name


from .comm_channel import *
from .conv_cleanup import *
from .convolution import *
from .inhibit import *
from .learning import *
from .lorenz import *
from .matrix_mult import *
from .memory import *
from .memory_recall import *
from .mnist import *
from .oscillator import *
from .parse import *
from .sequence import *
from .sequence_routed import *
