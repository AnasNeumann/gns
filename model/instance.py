import json

# =====================================================
# =*= INSTANCE DATA STRUCTURE & ACCESS FUNCTIONS =*=
# =====================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

class Instance:
    def __init__(self, size: int, id: int, w_makespan: int, H: int, **kwargs):
        self.id = id
        self.size = size
        self.H = H
        self.w_makespan = w_makespan  
        
        # Global configuration
        self.M = kwargs.get('M', -1)     
        self.nb_settings = kwargs.get('nb_settings', -1)
        self.nb_HR_types = kwargs.get('nb_HR_types', -1)
        self.nb_human_resources = kwargs.get('nb_human_resources', -1)
        self.nb_production_machine_types = kwargs.get('nb_production_machine_types', -1)
        self.nb_production_machines = kwargs.get('nb_production_machines', -1)
        self.nb_material = kwargs.get('nb_material', -1)
        self.nb_ops_types = kwargs.get('nb_ops_types', -1)
        self.total_elements = kwargs.get('total_elements', -1)
        self.total_operations = kwargs.get('total_operations', -1)
        self.nb_resource_types = kwargs.get('nb_resource_types', -1)
        self.nb_resources = kwargs.get('nb_resources', -1)
        self.E_size = kwargs.get('E_size', []) #p
        self.O_size = kwargs.get('O_size', []) #p
        self.EO_size = kwargs.get('EO_size', []) #p, e

        # Resources
        self.resource_family = kwargs.get('resource_family', []) #r,rt (boolean)
        self.finite_capacity = kwargs.get('finite_capacity', []) #r (boolean)
        self.design_setup = kwargs.get('design_setup', []) #r, s
        self.operation_setup = kwargs.get('operation_setup', []) #r
        self.execution_time = kwargs.get('execution_time', []) #r, p, o

        # Consumable materials
        self.init_quantity = kwargs.get('init_quantity', []) #r
        self.purchase_time = kwargs.get('purchase_time', []) #r
        self.quantity_needed = kwargs.get('quantity_needed', []) #r, p, o

        # Items
        self.assembly = kwargs.get('assembly', []) #p, e1, e2 (boolean)
        self.direct_assembly = kwargs.get('direct_assembly', []) #p, e1, e2 (boolean)
        self.external = kwargs.get('external', []) #p, e (boolean)
        self.outsourcing_time = kwargs.get('outsourcing_time', []) #p, e
        self.external_cost = kwargs.get('external_cost', []) #p, e 

        # Operations
        self.operation_family = kwargs.get('operation_family', []) #p, o, ot (boolean)
        self.simultaneous = kwargs.get('simultaneous', []) #p, o (boolean)
        self.resource_type_needed = kwargs.get('resource_type_needed', []) #p, o, rt (boolean)
        self.in_hours = kwargs.get('in_hours', []) #p, o (boolean)
        self.in_days = kwargs.get('in_days', []) #p, o (boolean)
        self.is_design = kwargs.get('is_design', []) #p, o (boolean)
        self.design_value = kwargs.get('design_value', []) #p, o, s
        self.operations_by_element = kwargs.get('operations_by_element', []) #p, e, o (boolean)
        self.precedence = kwargs.get('precedence', []) #p, e, o1, o2 (boolean)
    
    def get_name(self):
        return self.size+'_'+str(self.id)

    def get_children(self, p: int, e: int, direct: bool=True):
        data = self.direct_assembly if direct else self.assembly
        children = []
        for e2 in range(self.E_size[p]):
            if data[p][e][e2]:
                children.append(e2)
        return children

    def get_direct_parent(self, p: int, e: int):
        for e2 in range(self.E_size[p]):
            if self.direct_assembly[p][e2][e]:
                return e2
        return -1
    
    def get_finie_capacity_resources(self):
        return [r for r in range(self.nb_resources) if self.finite_capacity[r]]
    
    def get_consumable_materials(self):
        return [r for r in range(self.nb_resources) if not self.finite_capacity[r]]

    def get_ancestors(self, p: int, e: int):
        ancestors = []
        for e2 in range(self.E_size[p]):
            if self.assembly[p][e2][e]:
                ancestors.append(e2)
        return ancestors
    
    def get_operations_idx(self, p: int, e: int):
        start = 0
        for e2 in range(0, e):
            start = start + self.EO_size[p][e2]    
        return start, start+self.EO_size[p][e]
    
    def get_operation_type(self, p: int, o: int):
        for ot in range(self.nb_ops_types):
            if self.operation_family[p][o][ot]:
                return ot
        return -1
    
    def get_resource_type(self, r: int):
        for rt in range(self.nb_resource_types):
            if self.resource_family[r][rt]:
                return rt
        return -1
    
    def get_item_of_operation(self, p: int, o: int):
        for e in range(self.E_size[p]):
            if self.operations_by_element[p][e][o]:
                return e
        return -1
    
    def operation_resource_time(self, p: int, o: int, rt: int, max_load: bool):
        resources = self.resources_by_type(rt)
        if not resources or not self.finite_capacity[resources[0]]:
            return 0
        time_rt = []
        for r in self.resources_by_type(rt):
                time_rt.append(self.execution_time[r][p][o])
        if max_load:
            return max(time_rt)
        return sum(time_rt)/len(time_rt) if time_rt else 0

    def operation_time(self, p: int, o: int, total_load: bool):
        times = []
        for rt in self.required_rt(p, o):
                times.append(self.operation_resource_time(p, o, rt, max_load=total_load))
        if total_load:
            return sum(times)
        return max(times)

    def item_processing_time(self, p: int, e: int, total_load:bool):
        design_time = 0
        physical_time = 0
        for o in self.loop_item_operations(p,e):
            if self.is_design[p][o]:
                design_time += self.operation_time(p,o, total_load=total_load)
            else:
                physical_time += self.operation_time(p,o, total_load=total_load)
        return design_time, physical_time

    def require(self, p: int, o: int, r: int):
        for rt in range(self.nb_resource_types):
            if self.resource_family[r][rt]:
                return self.resource_type_needed[p][o][rt]
        return False
    
    def required_rt(self, p: int, o: int):
        rts = []
        for rt in range(self.nb_resource_types):
            if self.resource_type_needed[p][o][rt]:
                rts.append(rt)
        return rts

    def real_time_scale(self, p: int, o: int):
        return 60*self.H if self.in_days[p][o] else 60 if self.in_hours[p][o] else 1

    def get_nb_projects(self):
        return len(self.E_size)
    
    def operations_by_resource(self, r: int):
        return self.operations_by_resource_type(self.get_resource_familly(r))
    
    def operations_by_resource_type(self, rt: int):
        operations = []
        for p in range(self.get_nb_projects()):
            for o in range(self.O_size[p]):
                if self.resource_type_needed[p][o][rt]:
                    operations.append((p, o))
        return operations

    def project_head(self, p: int):
        for e in range(self.E_size[p]):
            is_head = True
            for e2 in range(self.E_size[p]):
                if e2 != e and self.assembly[p][e2][e]:
                    is_head = False
                    break
            if(is_head):
                return e
        return -1
    
    def preds_or_succs(self, p: int, e: int, start: int, end: int, o, design_only: bool=False, physical_only: bool=False, preds: bool=True):
        operations = []
        for other in range(start, end):
            if other!=o and (not design_only or self.is_design[p][other]) \
                and (not physical_only or not self.is_design[p][other]) \
                and ((not preds and self.precedence[p][e][other][o]) or (preds and self.precedence[p][e][o][other])):
                operations.append(other)
        return operations
    
    def succs(self, p: int, e: int, o: int, design_only: bool=False, physical_only: bool=False):
        operations = []
        for other in self.loop_item_operations(p,e):
            if other!=o and (not design_only or self.is_design[p][other]) \
                and (not physical_only or not self.is_design[p][other]) \
                and self.precedence[p][e][other][o]:
                operations.append(other)
        return operations
    
    def last_design_operations(self, p: int, e: int):
        ops = []
        start, end = self.get_operations_idx(p, e)
        for o in range(start, end):
            if self.is_design[p][o]:
                succs = self.preds_or_succs(p, e, start, end, o, design_only=True, physical_only=False, preds=False)
                if not succs:
                    ops.append(o)
        return ops

    def first_operations(self, p: int, e: int):
        ops = []
        start, end = self.get_operations_idx(p, e)
        for o in range(start, end):
            preds = self.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=False, preds=True)
            if not preds:
                ops.append(o)
        return ops
    
    def first_design_operations(self, p: int, e: int):
        ops = []
        start, end = self.get_operations_idx(p, e)
        for o in range(start, end):
            if self.is_design[p][o]:
                preds = self.preds_or_succs(p, e, start, end, o, design_only=True, physical_only=False, preds=True)
                if not preds:
                    ops.append(o)
        return ops

    def first_physical_operations(self, p: int, e: int):
        ops = []
        start, end = self.get_operations_idx(p, e)
        for o in range(start, end):
            if not self.is_design[p][o]:
                preds = self.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=True, preds=True)
                if not preds:
                    ops.append(o)
        return ops

    def last_operations(self, p: int, e: int):
        ops = []
        start, end = self.get_operations_idx(p, e)
        for o in range(start, end):
            succs = self.preds_or_succs(p, e, start, end, o, design_only=False, physical_only=False, preds=False)
            if not succs:
                ops.append(o)
        return ops

    def required_resources(self, p: int, o: int):
        resources = []
        for r in range(self.nb_resources):
            if self.require(p, o, r):
                resources.append(r)
        return resources

    def is_same(self, p1: int, p2: int, o1: int, o2: int):
        return (p1 == p2) and (o1 == o2)

    def get_resource_familly(self, r: int):
        for rf in range(self.nb_resource_types):
            if self.resource_family[r][rf]:
                return rf
        return -1

    def real_time_scale(self, p: int, o: int):
        return 60*self.H if self.in_days[p][o] else 60 if self.in_hours[p][o] else 1

    def resources_by_type(self, rt: int):
        resources = []
        for r in range(self.nb_resources):
            if self.resource_family[r][rt]:
                resources.append(r)
        return resources
    
    def is_last_design(self, p: int, e: int, o: int):
        for o2 in self.last_design_operations(p, e):
            if o2 == o:
                return True
        return False

    def is_last_operation(self, p: int, e: int, o: int):
        for o2 in self.last_operations(p, e):
            if o2 == o:
                return True
        return False
    
    def loop_item_operations(self, p: int, e: int):
        start, end = self.get_operations_idx(p,e)
        return range(start, end)

    def loop_projects(self):
        return range(len(self.E_size))

    def loop_items(self, p: int):
        return range(self.E_size[p])
    
    def loop_resources(self):
        return range(self.nb_resources)
    
    def loop_rts(self):
        return range(self.nb_resource_types)

    def loop_operations(self, p: int):
        return range(self.O_size[p])

    def next_operations(self, p: int, e: int, o: int):
        operations = []
        no_child = True
        if self.is_design[p][o]:
            if self.is_last_design(p, e, o):
                for child in self.get_children(p, e, direct=True):
                    no_child = False
                    operations.extend(self.first_design_operations(p, child))
                if no_child:
                    operations.extend(self.succs(p, e, o, design_only=False, physical_only=True))
            else:
                operations.extend(self.succs(p, e, o, design_only=True, physical_only=False))
        else:
            operations.extend(self.succs(p, e, o, design_only=False, physical_only=True))
        if self.is_last_operation(p, e, o) and no_child:
            parent_found = False
            current = e
            while not parent_found:
                parent = self.get_direct_parent(p, current)
                if parent >= 0:
                    physcal_ops = self.first_physical_operations(p, parent)
                    if physcal_ops:
                        operations.extend(physcal_ops)
                        parent_found = True
                    else:
                        current = parent
                else:
                    parent_found = True
        return operations

    def build_next_and_previous_operations(self):
        next = [[self.next_operations(p, self.get_item_of_operation(p, o), o) for o in self.loop_operations(p)] for p in self.loop_projects()]
        previous = [[[] for _ in self.loop_operations(p)] for p in self.loop_projects()]
        for p in self.loop_projects():
            for o in self.loop_operations(p):
                for successor in next[p][o]:
                    previous[p][successor].append(o)
        return previous, next

    def recursive_display_item(self, p: int, e: int, parent: int, level: int=1):
        operations = []
        children = []
        for child in self.get_children(p, e, True):
            children.append(self.recursive_display_item(p, child, e, level=level+1))
        for o in self.loop_item_operations(p,e):
            resource_types = []
            material_types = []
            for rt in self.required_rt(p, o):
                resources = self.resources_by_type(rt)
                if resources:
                    finite = self.finite_capacity[resources[0]]
                    if finite:
                        r = resources[0]
                        resource_types.append({"RT": rt, "execution_time": self.execution_time[r][p][o]})
                    else:
                        m = resources[0]
                        material_types.append({"RT": rt, "quantity_needed": self.quantity_needed[m][p][o]})
                else:
                    resource_types.append({"RT": rt, "execution_time": -1})
            if material_types:
                operations.append({
                    "operation_id": o,
                    "type_for_setups": self.get_operation_type(p, o),
                    "design_values_for_setups": self.design_value[p][o],
                    "simultaneous": self.simultaneous[p][o],
                    "is_design": self.is_design[p][o],
                    "total_resources": len(resource_types) + len(material_types),
                    "resource_types": resource_types,
                    "material_types": material_types
                })
            else:
                operations.append({
                    "operation_id": o,
                    "type_for_setups": self.get_operation_type(p, o),
                    "design_values_for_setups": self.design_value[p][o],
                    "simultaneous": self.simultaneous[p][o],
                    "is_design": self.is_design[p][o],
                    "total_resources": len(resource_types) + len(material_types),
                    "resource_types": resource_types,
                })
        if self.external[p][e]:
            if children:
                return {
                    "item_id": e, 
                    "EBOM_level": level,
                    "parent": parent,
                    "outsourcing_time": self.outsourcing_time[p][e],
                    "external_cost": self.external_cost[p][e],
                    "nb_operations": len(operations),
                    "operations": operations, 
                    "nb_children": len(children),
                    "children": children
                }
            else:
                return {
                    "item_id": e, 
                    "parent": parent,
                    "outsourcing_time": self.outsourcing_time[p][e],
                    "external_cost": self.external_cost[p][e],
                    "operations": operations, 
                }
        else:
            if children:
                return {"item_id": e, "parent": parent, "nb_children": len(children), "operations": operations, "children": children}
            else:
                return {"item_id": e, "parent": parent, "operations": operations}
            
    def display_resource(self, r: int) -> dict:
        rt: int = self.get_resource_familly(r)
        _nb_rs: int = len(self.resources_by_type(rt)) -1
        if self.finite_capacity[r]:
            return {"r_id": r, "finit_capacity": True, "type": rt, "other_similar": _nb_rs, "operation_setup": self.operation_setup[r], "design_setups": self.design_setup[r]}
        return {"r_id": r, "finit_capacity": False, "type": rt, "other_similar": _nb_rs, "init_quantity": self.init_quantity[r], "purchase_time": self.purchase_time[r]}

    def display(self):
        _resources = []
        for r in self.loop_resources():
            _resources.append(self.display_resource(r))
        _projects = []
        for p in self.loop_projects():
            _projects.append({"project_id:": p, "head": self.recursive_display_item(p, self.project_head(p), -1)})
        return json.dumps({"alpha_for_makespan": self.w_makespan, "design_scale": 60*self.H, "assembly_scale": 60, "production_scale": 1, "nb_projects": len(_projects), "nb_resources": len(_resources), "resources": _resources, "projects": _projects}, indent=4)  