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
        self.time_queue: list[int] = [0]
        self.operation_queue: list[int] = []
        self.item_queue: list[int] = []

    def done(self) -> bool:
        return len(self.operation_queue) == 0 and len(self.item_queue) == 0
    
    def get_next_time(self, pop: bool=True) -> int:
        if pop:
            self.pop_time()
        if self.time_queue:
            return self.time_queue[0]
        return -1
    
    def pop_time(self):
        self.time_queue.pop(0)

    def add_time(self, time: int):
        if time not in self.time_queue:
            bisect.insort(self.time_queue, time)
    
    def add_operation(self, operation: int):
        self.operation_queue.append(operation)

    def add_item(self, item: int):
        self.item_queue.append(item)
    
    def remove_operation(self, operation: int):
        self.operation_queue.remove(operation)  
    
    def remove_item(self, item: int):
        self.item_queue.remove(item)