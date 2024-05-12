import sys
import random
import pickle

# STOCHASTIC CONFIGURATION OF THE INSTANCES
NB_TRAIN_INSTANCES = int(sys.argv[1])
NB_TEST_INSTANCES = int(sys.argv[2])
MIN_JOBS = int(sys.argv[3])
MAX_JOBS = int(sys.argv[4])
MIN_TYES_OF_RESOURCES = int(sys.argv[5])
MAX_TYES_OF_RESOURCES = int(sys.argv[6])
MAX_RESOURCES_BY_TYPE = int(sys.argv[7])
MIN_OPERATIONS = 2
MAX_OPERATIONS = int(sys.argv[8])
MIN_PROCESSING_TIME = 1
MAX_PROCESSING_TIME = 15

# INSTANCE GENERATION:
for i in range(NB_TRAIN_INSTANCES + NB_TEST_INSTANCES):

    # RESOURCES GENERATION
    resources = []
    total_res = 0
    for _ in range(random.randint(MIN_TYES_OF_RESOURCES, MAX_TYES_OF_RESOURCES)):
        nb_res_by_type = random.randint(1, MAX_RESOURCES_BY_TYPE)
        resources.append(nb_res_by_type)
        total_res += nb_res_by_type

    # JOB GENERATION
    jobs = []
    total_ops = 0
    for _ in range(random.randint(MIN_OPERATIONS, MAX_OPERATIONS)):
        j = []
        for _ in range(random.randint(MIN_OPERATIONS, MAX_OPERATIONS)):
            processing_time = random.randint(MIN_PROCESSING_TIME, MAX_PROCESSING_TIME)
            resource = random.randint(0, len(resources)-1)
            j.append((resource, processing_time))
            total_ops +=1
        jobs.append(j)

    # SAVE THE INSTANCE
    instance = {"resources": resources, "jobs": jobs, "size": total_ops, "nb_res": total_res}
    folder = "train" if i < NB_TRAIN_INSTANCES else "test"
    with open('./FJS/instances/'+folder+'/instance_'+str(i)+'.pkl', 'wb') as f:
        pickle.dump(instance, f)
    print("Instance #"+str(i)+" saved successfully!")
