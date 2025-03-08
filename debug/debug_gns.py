from typing import Callable
from model.graph import GraphInstance
# =====================================================
# =*= DEBUG THE GNS SOLVER =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

def debug_printer(mode):
    if mode:
        def debug_print(*args):
            print(*args)
            with open('./log.sh', 'a') as file:
                file.write(*args)
                file.write('\n')
    else:
        def debug_print(*_):
            pass
    return debug_print