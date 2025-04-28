from typing import Tuple
import bisect

# ==========================================
# =*= A QUEUE OF TIME AND ACTION TO TEST =*=
# ==========================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

OUTSOURCING = 0
SCHEDULING = 1
MATERIAL_USE = 2

class Queue:
    def __init__(self):
        self.operation_queue: list[int] = []
        self.item_queue: list[int] = []

    def done(self) -> bool:
        return len(self.operation_queue) == 0 and len(self.item_queue) == 0 # and len(self.item_queue) == 0

    def add_operation(self, operation: int):
        self.operation_queue.append(operation)

    def add_item(self, item: int):
        self.item_queue.append(item)
    
    def remove_operation(self, operation: int):
        self.operation_queue.remove(operation)  
    
    def remove_item(self, item: int):
        self.item_queue.remove(item)