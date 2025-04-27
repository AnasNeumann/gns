from model.instance import Instance
from tools.common import init_several_1D, init_2D, init_several_2D, init_3D
import random

# ====================================================================
# =*= EXACT (Google OR-Tool) SOLUTION DATA STRUCTURE =*=
# ====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class Solution:
    def __init__(self):
        self.E_start, self.E_outsourced, self.E_prod_start, self.E_validated, self.E_end = [], [], [], [], [] # Elements (p, e)
        self.O_uses_init_quantity, self.O_start, self.O_setup, self.O_end, self.O_executed = [], [], [], [], [] # Execution of operations (p, o, feasible r)
        self.precedes = [] # Relation between operations (p1, p2, o1, o2, feasible r)
        self.D_setup = [] # Design setup (p, o, r, s)
        self.Cmax = -1 # Cmax and objective
        self.obj = []

# ====================================================================
# =*= HEURISTIC SOLUTION DATA STRUCTURE =*=
# ====================================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

YES = 1
NO = 0 

class Operation():
    def __init__(self):
        self.id: int = 0
        self.operation_family: int = 0
        self.simultaneous: bool = False
        self.machine_usage: list[Execution] = []
        self.material_use: list[MaterialUse] = []
        self.in_hours: bool = False
        self.in_days: bool = False
        self.is_design: bool = False
        self.design_value: list[int] = []
        self.item: Item = None

    def get_machine_usage(self, rt: int):
        for execution in self.machine_usage:
            if execution.machine_type.id == rt:
                return execution
        return None
    
    def get_material_use(self, m: int):
        for use in self.material_use:
            if use.material.id == m:
                return use
        return None
    
    def json_display(self):
        return {
            "id": self.id,
            "operation_family": self.operation_family,
            "simultaneous": str(self.simultaneous).lower(),
            "in_hours": str(self.in_hours).lower(),
            "in_days": str(self.in_days).lower(),
            "is_design": str(self.is_design).lower(),
            "item_id": self.item.id,
            "design_value": self.design_value,
            "machine_usage": [r.json_display() for r in self.machine_usage],
            "material_use": [m.json_display() for m in self.material_use]
        }

class Item():
    def __init__(self):
        self.id: int = 0
        self.external: bool = False
        self.start: int = 0
        self.end: int = 0 
        self.outsourced: bool = False
        self.outsourcing_time: int = 0
        self.external_cost: int = 0
        self.children: list[Item] = []
        self.parent: Item = None
        self.project: Project = None
        self.design_ops: list[Operation] = []
        self.production_ops: list[Operation] = []
    
    def json_display(self):
        return {
            "id": self.id,
            "external": str(self.external).lower(),
            "outsourced": str(self.outsourced).lower(),
            "start": self.start,
            "end": self.end,
            "outsourcing_time": self.outsourcing_time,
            "external_cost": self.external_cost,
            "project_id": self.project.id,
            "children": [c.json_display() for c in self.children],
            "design_ops": [d.json_display() for d in self.design_ops],
            "production_ops": [p.json_display() for p in self.production_ops]
        }

class Project():
    def __init__(self):
        self.id: int = 0
        self.head: Item = None
        self.flat_items: list[Item] = []
        self.flat_operations: list[Operation] = []
    
    def json_display(self):
        return {
            "id": self.id,
            "head": self.head.json_display()
        }
    
class RT():
    def __init__(self):
        self.id: int = 0
        self.finite_capacity: bool = False
        self.machines: list[Machine] = []
        self.sequence: list[Execution] = []

    def json_display(self):
        return {
            "id": self.id,
            "finite_capacity": str(self.finite_capacity).lower(),
            "machines": [r.json_display() for r in self.machines]
        }

class Use():
    def __init__(self):
        self.position: int = 0
        self.operation: Operation = None

class Execution(Use):
    def __init__(self):
        Use.__init__(self)
        self.start: int = 0
        self.end: int = 0
        self.selected_machine: Machine = 0
        self.machine_type: RT = None

    def json_display(self):
        return {
            "operation_id": self.operation.id,
            "machine_type_id": self.machine_type.id,
            "selected_machine_id": self.selected_machine.id,
            "start": self.start,
            "end": self.end
        }

class MaterialUse(Use):
    def __init__(self):
        Use.__init__(self)
        self.quantity_needed = 0
        self.material: Material = None
        self.execution_time: int = 0

    def json_display(self):
        return {
            "operation_id": self.operation.id,
            "material_id": self.material.id,
            "quantity_needed": self.quantity_needed,
            "execution_time": self.execution_time
        }

class Resource():
    def __init__(self):
        self.id = 0
        self.resource_type = 0
        self.finite_capacity: bool = False

class Machine(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.type: RT = None
        self.design_setup: list[int] = []
        self.operation_setup: int = 0
        self.finite_capacity = True

    def json_display(self):
        return {
            "id": self.id,
            "operation_setup": self.operation_setup,
            "design_setup": self.design_setup,
            "sequence": [[f"p{e.operation.item.project.id}", f"e{e.operation.item.id}", f"o{e.operation.id}", f"start = {e.start}", f"start = {e.end}"] for e in self.type.sequence if e.selected_machine.id == self.id]
        }

class Material(Resource):
    def __init__(self):
        Resource.__init__(self)
        self.sequence: list[MaterialUse] = []
        self.init_quantity = 0
        self.purchase_time = 0
        self.finite_capacity = False

    def json_display(self):
        return {
            "id": self.id,
            "init_quantity": self.init_quantity,
            "purchase_time": self.purchase_time,
            "sequence": [[f"p{e.operation.item.project.id}", f"e{e.operation.item.id}", f"o{e.operation.id}", f"execution time = {e.execution_time}"] for e in self.sequence]
        }

class HeuristicSolution():
    def __init__(self):
        self.H = 0
        self.M = 0
        self.id = 0
        self.w_makespan = 0
        self.nb_settings = 0
        self.projects: list[Project] = []
        self.machine_types: list[RT] = []
        self.materials: list[Material] = []
        self.flat_resources: list[Resource] = []
        self.total_cost = 0
        self.Cmax = -1
        self.feasible: bool = False

    @staticmethod
    def geneget(obj_list, obj_id):
        for obj in obj_list:
            if obj.id == obj_id:
                return obj
        return None

    def json_display(self):
        return {
            "id": self.id,
            "H": self.H,
            "M": self.M,
            "w_makespan": self.w_makespan,
            "nb_settings": self.nb_settings,
            "total_cost": self.total_cost,
            "Cmax": self.Cmax,
            "feasible": str(self.feasible).lower(),
            "projects": [p.json_display() for p in self.projects],
            "machine_types": [r.json_display() for r in self.machine_types],
            "materials": [m.json_display() for m in self.materials]
        }
    
    def start_operation(self, i: Instance, project: Project, item: Item, o: int):
        p = project.id
        operation: Operation = Operation()
        operation.id = o
        operation.operation_family = i.get_operation_type(p,o)
        operation.simultaneous = i.simultaneous[p][o]
        operation.in_hours = i.in_hours[p][o]
        operation.in_days = i.in_days[p][o]
        operation.is_design = i.is_design[p][o]
        operation.design_value = i.design_value[p][o]
        operation.item = item
        for rt in i.required_rt(p, o):
            resx = i.resources_by_type(rt)
            if resx and i.finite_capacity[i.resources_by_type(rt)[0]]:
                type: RT = self.machine_types[rt]
                execution: Execution = Execution()
                execution.selected_machine = type.machines[0]
                execution.machine_type = type
                execution.operation = operation
                operation.machine_usage.append(execution)
        for r in i.required_resources(p, o):
            if not i.finite_capacity[r]:
                use: MaterialUse = MaterialUse()
                use.quantity_needed = i.quantity_needed[r][p][o]
                use.material = self.flat_resources[r]
                use.operation = operation
                operation.material_use.append(use)
        operation.material_use.sort(key=lambda obj: obj.execution_time)
        operation.machine_usage.sort(key=lambda obj: obj.start)
        if operation.is_design:
            item.design_ops.append(operation)
        else:
            item.production_ops.append(operation)
        project.flat_operations.append(operation)
    
    def reccursive_start_item(self, i: Instance, project: Project, e: int, parent: Item = None, head: bool = False):
        p = project.id
        item: Item = Item()
        item.id = e
        item.external = i.external[p][e]
        item.outsourcing_time = i.outsourcing_time[p][e]
        item.external_cost = i.external_cost[p][e]
        item.parent = parent
        item.project = project
        o_start, o_end = i.get_operations_idx(p, e)
        for o in range(o_start, o_end):
            self.start_operation(i, project, item, o)
        for c in i.get_children(p, e, direct=True):
            child = self.reccursive_start_item(i, project, c, item, head=False)
            item.children.append(child)
        project.flat_items.append(item)
        if head:
            project.head = item
        item.children.sort(key=lambda obj: obj.id)
        item.design_ops.sort(key=lambda obj: obj.id)
        item.production_ops.sort(key=lambda obj: obj.id)
        return item
    
    def start_resources(self, i: Instance):
        for rt in i.loop_rts():
            type: RT = RT()
            type.id = rt
            resx = i.resources_by_type(rt)
            if (not resx or i.finite_capacity[resx[0]]):
                type.finite_capacity = True
            self.machine_types.append(type)
        self.machine_types.sort(key=lambda obj: obj.id)
        for r in i.loop_resources():
            if i.finite_capacity[r]:
                machine: Machine = Machine()
                machine.id = r
                rt = i.get_resource_type(r)
                machine.resource_type = rt
                type = self.machine_types[rt]
                machine.type = type
                machine.design_setup = i.design_setup[r]
                machine.operation_setup = i.operation_setup[r]
                self.flat_resources.append(machine)
                type.machines.append(machine)
            else:
                material: Material = Material()
                material.id = r
                material.resource_type = i.get_resource_type(r)
                material.init_quantity = i.init_quantity[r]
                material.purchase_time = i.purchase_time[r]
                material.quantity_purchased = i.quantity_needed[r]
                self.flat_resources.append(material)
                self.materials.append(material)
        for rt in self.machine_types:
            rt.machines.sort(key=lambda obj: obj.id)
        self.flat_resources.sort(key=lambda obj: obj.id)
        self.materials.sort(key=lambda obj: obj.id)
    
    def start_from_instance(self, i: Instance):
        self.H = i.H
        self.M = i.M
        self.id = i.id
        self.w_makespan
        self.nb_settings = i.nb_settings
        self.start_resources(i)
        for p in i.loop_projects():
            project: Project = Project()
            project.id = p
            self.reccursive_start_item(i, project, i.project_head(p), head=True)
            project.flat_items.sort(key=lambda obj: obj.id)
            project.flat_operations.sort(key=lambda obj: obj.id)
            self.projects.append(project)
        self.projects.sort(key=lambda obj: obj.id)

    def simulate(self, i: Instance):
        # TODO
        pass