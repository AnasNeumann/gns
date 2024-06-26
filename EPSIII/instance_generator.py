import argparse
import random
import pickle
from model import Instance, get_direct_children, get_direct_parent, get_operations_idx, get_name, require, real_time_scale

PROBLEM_SIZES = ['s', 'm', 'l', 'xl', 'xxl', 'xxxl']
SIZE = 0
H = 5

# Resources and Types of Resources [4, 6, 11, 15, 19, 25]
NB_HUMAN_RESOURCES = [1, 2, 3, 4, 5, 8]
NB_RAW_MATERIAL_TYPES = [1, 2, 2, 3, 4, 5]
NB_PRODUCTION_MACHINE_TYPES = [2, 3, 4, 6, 8, 9]
NB_PRODUCTION_MACHINES = [2, 4, 6, 8, 10, 12]
UNKOWN_MACHINE_TYPE = 1

# Types of Operations
NB_DESIGN_OPERATION_TYPES = [1, 3, 5, 5, 6, 8]
NB_PRODUCTION_OPERATION_TYPES = [4, 6, 6, 6, 8, 10]
NB_ASSEMBLY_OPERATION_TYPES = [1, 2, 4, 4, 6, 8]
INIT_QUANTITY = 500
MAX_QUANTITY_USED = 100
NB_TYPES_OF_SETTINGS = [1, 2, 3, 4, 5, 6]
MAX_SETTINGS_VALUE = 5
MAX_SETUP_TIME = 3
MAX_PROCESSING_TIMES_DESIGN = 3
MAX_PROCESSING_TIMES_ASSEMBLY = 6
MAX_PROCESSING_TIMES_PROD = 40
MIN_OUTSOURCING_PRICE_SHARE = 0.03
MAX_OUTSOURCING_PRICE_SHARE = 0.15
MIN_OUTSOURCING_TIME_SHARE = 0.8
MAX_OUTSOURCING_TIME_SHARE = 1.2

# Current Projects, Elements and Operations
NB_PROJECTS = [1, 3, 3, 4, 5, 6] # 1, 3, 3, 4, 5, 6 projects
NB_ELTS_PER_PROJECT = [7, 7, 10, 10, 10, 10] # 7, 21, 30, 40, 50, 60 elements
MEAN_OPS_PER_ELT = [3, 3, 3, 3, 4, 5]
# 21 (23), 63 (65), 90 (97), 120 (133), 200 (225), 300 (337) tasks

def bias_generator(prop_false):
    return random.uniform(0.0,1.000001)>=prop_false

def init_array(size, min, max, rdm=True):
    result = []
    for _ in range(size):
        result.append(random.randint(min, max) if rdm else max)
    return result

def build_resources(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    i.resource_family = [[False] * i.nb_resource_types for _ in range(i.nb_resources)]
    i.finite_capacity = [False] * i.nb_resources
    i.design_setup = [[0] * i.nb_settings for _ in range(i.nb_resources)]
    i.operation_setup = [0] * i.nb_resources
    i.execution_time = [[-1] * i.nb_projects for _ in range(i.nb_resources)]
    i.init_quantity =  [0] * i.nb_resources
    i.purchase_time = [0] * i.nb_resources
    i.quantity_needed =  [[-1] * i.nb_projects for _ in range(i.nb_resources)]
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
                i.resourceFamily[r][random.randint(i.nb_human_resources, i.nb_human_resources + i.nb_production_machine_types - 1)] = True
            for s in range(i.nb_settings):
                i.design_setup[r][s] = MAX_SETUP_TIME if bias_generator(0.8) else 0
        else:
            i.purchase_time[r] = H * 60 * random.randint(1,3)
            i.init_quantity[r] = INIT_QUANTITY
            i.resourceFamily[r][i.nb_production_machine_types + r - i.nb_production_machines] = True
        for p in range(nb_projects):
            i.quantity_needed[r][p] = [-1 * i.O_size[p][o]]
            i.execution_time[r][p] = [-1 * i.O_size[p][o]]
            for o in range(i.O_size[p]):
                if require(i,p,o,r):
                    if i.finite_capacity[r]:
                        i.quantity_needed[r][p][o] = random.randint(0, MAX_QUANTITY_USED)
                        i.execution_time[r][p][o] = 0
                    else:
                        i.quantity_needed[r][p][o] = 0
                        val = MAX_PROCESSING_TIMES_DESIGN if i.in_days[p][o] else MAX_PROCESSING_TIMES_ASSEMBLY if i.in_hours[p][o] else MAX_PROCESSING_TIMES_PROD
                        i.execution_time[r][p][o] = real_time_scale(i,p,o) * random.randint(1, val)
    return i

def build_operations(i: Instance):
    found_unkown_elt = False
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    nb_design_operations_types = NB_DESIGN_OPERATION_TYPES[SIZE]
    nb_assembly_operation_types = NB_ASSEMBLY_OPERATION_TYPES[SIZE]
    nb_production_operation_types = NB_PRODUCTION_OPERATION_TYPES[SIZE]
    i.nb_ops_types = nb_design_operations_types + nb_assembly_operation_types + nb_production_operation_types
    i.operation_family = [[-1] for _ in range(nb_projects)]
    i.simultaneous = [[-1] for _ in range(nb_projects)]
    i.resource_type_needed = [[-1] for _ in range(nb_projects)]
    i.in_hours = [[-1] for _ in range(nb_projects)]
    i.in_days = [[-1] for _ in range(nb_projects)]
    i.is_design = [[-1] for _ in range(nb_projects)]
    i.design_value = [[-1] for _ in range(nb_projects)]
    for p in range(nb_projects):
        nb_ops = i.O_size[p]
        i.operation_family[p] = [[False] * i.nb_ops_types for _ in range(nb_ops)]
        i.simultaneous[p] = [False] * nb_ops
        i.resource_type_needed[p]= [[False] * i.nb_resource_types for _ in range(nb_ops)]
        i.in_hours[p] = [False] * nb_ops
        i.in_days[p] = [False] * nb_ops
        i.is_design[p] = [False] * nb_ops
        i.design_value[p] = [[-1] * i.nb_settings for _ in range(nb_ops)]
        for e in range(elts_per_project):
            first, last = get_operations_idx(i, p, e)
            for idx, o in enumerate(range(first, last)):
                if idx > 0:
                    i.design_value[p][o] = init_array(i.nb_settings, 0, MAX_SETTINGS_VALUE)
                ot = random.randint(0, nb_design_operations_types-1) if idx==0 \
                    else random.randint(nb_design_operations_types, nb_assembly_operation_types+nb_design_operations_types-1) if idx==1 \
                    else random.randint(nb_assembly_operation_types+nb_design_operations_types, i.nb_ops_types)
                if ot<nb_design_operations_types:
                    i.in_days[p][o] = True
                    i.is_design[p][o] = True
                elif ot<nb_assembly_operation_types+nb_design_operations_types:
                    i.in_hours[p][o] = True
                i.operation_family[p][o][ot] = True
                i.simultaneous[p][o] = bias_generator(0.9)
                maxRT = i.nb_HR_types-1 if (i.in_days[p][o] or i.in_hours[p][o]) else i.nb_production_machine_types - i.nb_material - UNKOWN_MACHINE_TYPE -1
                minRT = 0 if (i.in_days[p][o] or i.in_hours[p][o]) else i.nb_HR_types
                i.resource_type_needed[p][o][random.randint(minRT, maxRT)] = True
                if not i.in_days[p][o] and not i.in_hours[p][o] and bias_generator(0.8):
                    i.resource_type_needed[p][o][random.randint(maxRT+1, maxRT+i.nb_material)] = True
                if not found_unkown_elt and i.external[p][e] and not i.in_days[p][o] and not i.in_hours[p][o] and len(get_direct_children(i,p,e))<=0:
                    found_unkown_elt = True
                    i.resource_type_needed[p][o][random.randint(maxRT+i.nb_material+UNKOWN_MACHINE_TYPE)] = True
                    i.external_cost[p][e] = i.external_cost[p][e] * 2
    return i

def build_assembly(i: Instance, p, parent, ancestors):
    ancestors.append(parent)
    for children in get_direct_children(i, p, parent):
        for ancestor in ancestors:
            i.assembly[p][ancestor][children] = True
        i = build_assembly(i, p, children, list(ancestors))
    return i

def build_elements(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    i.assembly = [[[False] * elts_per_project for _ in range(elts_per_project)] for _ in range(nb_projects)]
    i.direct_assembly = [[[False] * elts_per_project for _ in range(elts_per_project)] for _ in range(nb_projects)]
    i.external = [[False] * elts_per_project for _ in range(nb_projects)]
    i.outsourcing_time = [[-1] * elts_per_project for _ in range(nb_projects)]
    i.external_cost = [[-1] * elts_per_project for _ in range(nb_projects)]
    mean_elt_op_time = MAX_PROCESSING_TIMES_DESIGN*60;
    MAX_PRICE = round(mean_elt_op_time*MAX_OUTSOURCING_PRICE_SHARE);
    MIN_PRICE = round(mean_elt_op_time*MIN_OUTSOURCING_PRICE_SHARE);
    MAX_TIME  = round(mean_elt_op_time*MAX_OUTSOURCING_TIME_SHARE);
    MIN_TIME  = round(mean_elt_op_time*MIN_OUTSOURCING_TIME_SHARE);
    i.M = 5 * mean_elt_op_time * NB_ELTS_PER_PROJECT[SIZE] * NB_PROJECTS[SIZE];
    for p in range(nb_projects):
        print(i.direct_assembly[p])
        for e in range(1, elts_per_project):
            print(e)
            print(i.direct_assembly[p])
            i.direct_assembly[p][random.randint(0,e-1)][e] = True
            print(i.direct_assembly[p])
            print("===")
        print(i.direct_assembly[p])
        i = build_assembly(i, p, 0, [])
        for e in range(elts_per_project):
            has_children = get_direct_children(i, p, e)
            outsourcable = e>0 and (i.external[p][get_direct_parent(i,p,e)] or ((not has_children and bias_generator(0.05)) or (has_children and bias_generator(0.25))))
            if outsourcable:
                i.external[p][e] = True
                i.outsourcing_time[p][e] = random.randint(MIN_TIME, MAX_TIME)
                i.external_cost[p][e] = random.randint(MIN_PRICE, MAX_PRICE)
    return i

def build_precedence(i: Instance):
    nb_projects = NB_PROJECTS[SIZE]
    elts_per_project = NB_ELTS_PER_PROJECT[SIZE]
    i.operations_by_element = [[-1] * elts_per_project for _ in range(nb_projects)]
    i.precedence = [[-1] * elts_per_project for _ in range(nb_projects)]
    for p in range(nb_projects):
        start = 0
        for e in range(elts_per_project):
            i.operations_by_element[p][e] = [False] * i.O_size[p]
            i.precedence[p][e] = [[False] * i.O_size[p] for _ in range(i.O_size[p])]
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

def build_one(size, id, w_makespan):
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
    i.nb_resource_types = i.nb_HR_types + i.nb_production_machine_types + i.nb_material + UNKOWN_MACHINE_TYPE
    i.nb_resources = i.nb_human_resources + i.nb_production_machines + i.nb_material
    i.E_size = [elts_per_project] * nb_projects
    i.EO_size = [[-1] * elts_per_project for _ in range(nb_projects)]
    i.O_size = [-1] * nb_projects
    return build_resources(build_operations(build_elements(build_precedence(build_projects(i)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EPSIII instances generator")
    parser.add_argument("--train", help="Number of training instances", required=True)
    parser.add_argument("--test", help="Number of testing instances", required=True)
    args = parser.parse_args()
    nb_train = int(args.train)
    nb_test = int(args.test)
    for size, size_folder in enumerate(PROBLEM_SIZES):
        SIZE = size
        print("Start size "+size_folder+"("+str(SIZE)+")...")
        for i in range(nb_train+nb_test):
            instance = build_one(size_folder, i, random.uniform(0.01, 0.99))
            folder = "train" if i<nb_train else "test"
            with open('./EPSIII/instances/'+size_folder+'/'+folder+'/instance_'+str(i)+'.pkl', 'wb') as f:
                pickle.dump(instance, f)
            print("\t Instance #"+get_name(i)+" saved successfully!")
        print("End size "+size_folder+"("+SIZE+")...")