import argparse
import random
import pickle
from model.instance import Instance
from tools.common import init_several_1D, init_several_2D, init_2D, init_1D, init_several_3D, directory, to_bool

# ==============================================================
# =*= RANDOM INSTANCE GENERATOR =*=
# Complete code to partially random instances of model.Instance
# ==============================================================
__author__ = "Anas Neumann - anas.neumann@polymtl.ca"
__version__ = "1.0.0"
__license__ = "Apache 2.0 License"

PROBLEM_SIZES = ['d'] + ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
SIZE = 0
H = 5

# Resources and Types of Resources [4, 6, 11, 15, 19, 25]
NB_HUMAN_RESOURCES = [1] + [1, 2, 3, 4, 5, 8]
NB_RAW_MATERIAL_TYPES = [1] + [1, 2, 2, 3, 4, 5]
NB_PRODUCTION_MACHINE_TYPES = [1] + [2, 3, 4, 6, 8, 9]
NB_PRODUCTION_MACHINES = [1] + [2, 4, 6, 8, 10, 12]
UNKOWN_MACHINE_TYPE = 1

# Types of Operations
NB_DESIGN_OPERATION_TYPES = [1] + [1, 3, 5, 5, 6, 8]
NB_PRODUCTION_OPERATION_TYPES = [1] + [4, 6, 6, 6, 8, 10]
NB_ASSEMBLY_OPERATION_TYPES = [1] + [1, 2, 4, 4, 6, 8]
INIT_QUANTITY = 500
MAX_QUANTITY_USED = 100
NB_TYPES_OF_SETTINGS = [0] + [1, 2, 3, 4, 5, 6]
MAX_SETTINGS_VALUE = 5
MAX_SETUP_TIME = 3
MAX_PROCESSING_TIMES_DESIGN = 5
MAX_PROCESSING_TIMES_ASSEMBLY = 10
MAX_PROCESSING_TIMES_PROD = 40
MIN_OUTSOURCING_PRICE_SHARE = 0.3
MAX_OUTSOURCING_PRICE_SHARE = 1.8
MIN_OUTSOURCING_TIME_SHARE = 0.6
MAX_OUTSOURCING_TIME_SHARE = 1.2

# Current Projects, Elements and Operations
NB_PROJECTS = [1] + [1, 3, 3, 4, 5, 6] # 1, 3, 3, 4, 5, 6 projects
NB_ELTS_PER_PROJECT = [6] + [7, 7, 10, 10, 10, 10] # 7, 21, 30, 40, 50, 60 elements
MEAN_OPS_PER_ELT = [3] + [3, 3, 3, 3, 4, 5]
# 21 (23), 63 (65), 90 (97), 120 (133), 200 (225), 300 (337) tasks

def bias_generator(prop_false: float):
    return random.uniform(0.0,1.000001)>=prop_false

def random_init_array(size: int, min: int, max: int):
    result = []
    for _ in range(size):
        result.append(random.randint(min, max))
    return result

def build_resources(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    i.resource_family = init_2D(i.nb_resources, i.nb_resource_types, False)
    i.design_setup = init_2D(i.nb_resources, i.nb_settings, 0)
    i.finite_capacity = init_1D(i.nb_resources, False)
    i.init_quantity, i.purchase_time, i.operation_setup = init_several_1D(i.nb_resources, 0, 3)
    i.quantity_needed, i.execution_time = init_several_2D(i.nb_resources, nb_projects, -1, 2)
    for r in range(i.nb_resources):
        if r < i.nb_human_resources:
            i.finite_capacity[r] = True
            i.resource_family[r][r] = True
        elif r < i.nb_human_resources+i.nb_production_machines:
            i.finite_capacity[r] = True
            i.operation_setup[r] = random.randint(0, MAX_SETUP_TIME)
            if r - i.nb_human_resources < i.nb_production_machine_types:
                i.resource_family[r][r] = True
            else:
                i.resource_family[r][random.randint(i.nb_human_resources, i.nb_human_resources + i.nb_production_machine_types - 1)] = True
            for s in range(i.nb_settings):
                i.design_setup[r][s] = MAX_SETUP_TIME if bias_generator(0.8) else 0
        else:
            i.purchase_time[r] = H * 60 * random.randint(1,3)
            i.init_quantity[r] = INIT_QUANTITY
            i.resource_family[r][i.nb_production_machine_types + r - i.nb_production_machines] = True
        for p in range(nb_projects):
            i.quantity_needed[r][p], i.execution_time[r][p] = init_several_1D(i.O_size[p], -1, 2)
            for o in range(i.O_size[p]):
                if i.require(p,o,r):
                    if i.finite_capacity[r]:
                        i.quantity_needed[r][p][o] = 0
                        val = MAX_PROCESSING_TIMES_DESIGN if i.in_days[p][o] else MAX_PROCESSING_TIMES_ASSEMBLY if i.in_hours[p][o] else MAX_PROCESSING_TIMES_PROD
                        i.execution_time[r][p][o] = i.real_time_scale(p,o) * random.randint(1, val)
                    else:
                        i.quantity_needed[r][p][o] = random.randint(0, MAX_QUANTITY_USED)
                        i.execution_time[r][p][o] = 0
    return i

def build_operations(i: Instance):
    _unknown_machines: int = UNKOWN_MACHINE_TYPE if SIZE > 0 else 0
    found_unkown_elt = False
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    nb_design_operations_types = NB_DESIGN_OPERATION_TYPES[SIZE]
    nb_assembly_operation_types = NB_ASSEMBLY_OPERATION_TYPES[SIZE]
    nb_production_operation_types = NB_PRODUCTION_OPERATION_TYPES[SIZE]
    i.nb_ops_types = nb_design_operations_types + nb_assembly_operation_types + nb_production_operation_types
    i.design_value, i.is_design, i.in_days, i.in_hours, i.resource_type_needed, i.simultaneous, i.operation_family = init_several_1D(nb_projects, 0, 7)
    for p in range(nb_projects):
        project_has_material = False
        nb_ops = i.O_size[p]
        i.is_design[p], i.in_days[p], i.in_hours[p], i.simultaneous[p], i.operation_family[p] = init_several_1D(nb_ops, False, 5)
        i.operation_family[p], i.resource_type_needed[p] = init_several_2D(nb_ops, i.nb_ops_types, False, 2)
        i.design_value[p] = init_2D(nb_ops, i.nb_settings, [-1])
        for e in range(elts_per_project):
            first, last = i.get_operations_idx(p, e)
            for idx, o in enumerate(range(first, last)):
                if idx > 0:
                    i.design_value[p][o] = random_init_array(size=i.nb_settings, min=0, max=MAX_SETTINGS_VALUE)
                else:
                    i.design_value[p][o] = [0] * i.nb_settings
                ot = random.randint(0, nb_design_operations_types-1) if idx==0 \
                    else random.randint(nb_design_operations_types, nb_assembly_operation_types+nb_design_operations_types-1) if idx==1 \
                    else random.randint(nb_assembly_operation_types+nb_design_operations_types, i.nb_ops_types-1)
                if ot<nb_design_operations_types:
                    i.in_days[p][o] = True
                    i.is_design[p][o] = True
                elif ot<nb_assembly_operation_types+nb_design_operations_types:
                    i.in_hours[p][o] = True
                i.operation_family[p][o][ot] = True
                i.simultaneous[p][o] = bias_generator(0.9)
                maxRT = i.nb_HR_types-1 if (i.in_days[p][o] or i.in_hours[p][o]) else i.nb_resource_types - i.nb_material - _unknown_machines -1
                minRT = 0 if (i.in_days[p][o] or i.in_hours[p][o]) else i.nb_HR_types
                i.resource_type_needed[p][o][random.randint(minRT, maxRT)] = True
                if not i.in_days[p][o] and not i.in_hours[p][o] and (bias_generator(0.7) or not project_has_material):
                    i.resource_type_needed[p][o][random.randint(maxRT+1, maxRT+i.nb_material)] = True
                    project_has_material = True
                if not found_unkown_elt and i.external[p][e] and not i.in_days[p][o] and not i.in_hours[p][o] and len(i.get_children(p,e,True))<=0:
                    found_unkown_elt = True
                    i.resource_type_needed[p][o][maxRT+i.nb_material+_unknown_machines] = True
                    i.external_cost[p][e] = i.external_cost[p][e] * 2
    return i

def build_assembly(i: Instance, p: int, parent: int, ancestors: list[int]):
    ancestors.append(parent)
    for child in i.get_children(p, parent, True):
        for ancestor in ancestors:
            i.assembly[p][ancestor][child] = True
        i = build_assembly(i, p, child, list(ancestors))
    return i

def build_elements(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    i.assembly, i.direct_assembly = init_several_3D(nb_projects, elts_per_project, elts_per_project, False, 2)
    i.outsourcing_time, i.external_cost = init_several_2D(nb_projects, elts_per_project, -1, 2)
    i.external = init_2D(nb_projects, elts_per_project, False)
    mean_elt_op_time = (MAX_PROCESSING_TIMES_DESIGN*60*H) + (MAX_PROCESSING_TIMES_ASSEMBLY*60) + (MEAN_OPS_PER_ELT[SIZE]-2)*MAX_PROCESSING_TIMES_PROD
    mean_elt_op_price = MAX_PROCESSING_TIMES_DESIGN*60*H
    MAX_PRICE = round(mean_elt_op_price*MAX_OUTSOURCING_PRICE_SHARE)
    MIN_PRICE = round(mean_elt_op_price*MIN_OUTSOURCING_PRICE_SHARE)
    MAX_TIME  = round(mean_elt_op_time*MAX_OUTSOURCING_TIME_SHARE)
    MIN_TIME  = round(mean_elt_op_time*MIN_OUTSOURCING_TIME_SHARE)
    i.M = 5 * mean_elt_op_price * NB_ELTS_PER_PROJECT[SIZE] * NB_PROJECTS[SIZE]
    for p in range(nb_projects):
        for e in range(1, elts_per_project):
            i.direct_assembly[p][random.randint(0,e-1)][e] = True
        i = build_assembly(i, p, 0, [])
        for e in range(elts_per_project):
            has_children = i.get_children(p, e, True)
            outsourcable = e>0 and (i.external[p][i.get_direct_parent(p,e)] or ((not has_children and bias_generator(0.05)) or (has_children and bias_generator(0.25))))
            if outsourcable:
                i.external[p][e] = True
                i.outsourcing_time[p][e] = random.randint(MIN_TIME, MAX_TIME)
                i.external_cost[p][e] = random.randint(MIN_PRICE, MAX_PRICE)
    return i

def build_precedence(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    i.operations_by_element, i.precedence = init_several_2D(nb_projects, elts_per_project, [-1], 2)
    for p in range(nb_projects):
        start = 0
        for e in range(elts_per_project):
            i.operations_by_element[p][e] = init_1D(i.O_size[p], False)
            i.precedence[p][e] = init_2D(i.O_size[p], i.O_size[p], False)
            for o in range(start, start + i.EO_size[p][e]):
                i.operations_by_element[p][e][o] = True
                if o > start:
                    i.precedence[p][e][o][o-1] = True
            start = start + i.EO_size[p][e]
    return i

def build_projects(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    mean_ops = MEAN_OPS_PER_ELT[SIZE]
    for p in range(nb_projects):
        ops_p = 0
        for e in range(elts_per_project):
            i.EO_size[p][e] = random.randint(mean_ops-1, mean_ops+1)
            ops_p = ops_p + i.EO_size[p][e]
            i.total_operations = i.total_operations + i.EO_size[p][e]        
        i.O_size[p] = ops_p
    return i

def build_one(size: int, id: int, w_makespan: float):
    _unknown_machines: int = UNKOWN_MACHINE_TYPE if SIZE > 0 else 0
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    i = Instance(size, id, w_makespan, H)
    i.nb_settings = NB_TYPES_OF_SETTINGS[SIZE]
    i.nb_HR_types = NB_HUMAN_RESOURCES[SIZE]
    i.nb_human_resources = NB_HUMAN_RESOURCES[SIZE]
    i.nb_production_machine_types = NB_PRODUCTION_MACHINE_TYPES[SIZE]
    i.nb_production_machines = NB_PRODUCTION_MACHINES[SIZE]
    i.nb_material = NB_RAW_MATERIAL_TYPES[SIZE]
    i.total_elements =  NB_ELTS_PER_PROJECT[SIZE] * NB_PROJECTS[SIZE]
    i.nb_resource_types = i.nb_HR_types + i.nb_production_machine_types + i.nb_material + _unknown_machines
    i.nb_resources = i.nb_human_resources + i.nb_production_machines + i.nb_material
    i.E_size = init_1D(nb_projects, elts_per_project)
    i.EO_size = init_2D(nb_projects, elts_per_project, -1)
    i.O_size = init_1D(nb_projects, -1)
    return build_resources(build_operations(build_elements(build_precedence(build_projects(i)))))

if __name__ == '__main__':
    '''
        TEST WITH
        python instance_generator.py --debug=false --train=150 --test=50
        python instance_generator.py --debug=true
    '''
    parser = argparse.ArgumentParser(description="EPSIII instances generator")
    parser.add_argument("--debug", help="Generate for debug mode", required=True)
    parser.add_argument("--train", help="Number of training instances", required=False)
    parser.add_argument("--test", help="Number of testing instances", required=False)
    args = parser.parse_args()
    _debug: bool = to_bool(args.debug)
    if _debug:
        """
            Debug mode
        """
        SIZE = 0
        instance: Instance = build_one(size='d', id=0, w_makespan=1.00)
        print(instance.display())
        with open(directory.instances+'/test/d/instance_debug.pkl', 'wb') as f:
            pickle.dump(instance, f)
        print("\t Debug instance saved successfully!")
    else:
        """
            Actual generation
        """
        nb_train: int = int(args.train)
        nb_test: int = int(args.test)
        for size, size_folder in enumerate(PROBLEM_SIZES):
            if size > 0:
                SIZE = size
                print("Start size "+size_folder+"("+str(SIZE)+")...")
                for i in range(1, nb_train+nb_test+1):
                    instance = build_one(size_folder, i, round(random.uniform(0.01, 0.99), 2))
                    folder = "train" if i<=nb_train else "test"
                    with open(directory.instances+'/'+folder+'/'+size_folder+'/instance_'+str(i)+'.pkl', 'wb') as f:
                        pickle.dump(instance, f)
                    print("\t Instance #"+instance.get_name()+" saved successfully!")
                print("End size "+size_folder+"("+str(SIZE)+")...")