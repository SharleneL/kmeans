import sys
import random
import math
import numpy as np

# input format: python kmeans.py [cluster_num]

file_name = "../../HW2_data/HW2_dev.docVectors"
# file_name = "../../HW2_data/fake.docVectors"


def main(argv):
    # save the file into a vector list
    vector_list = list()
    f = open(file_name)
    f_line = f.readline()
    while '' != f_line:
        vector_list.append(get_list(f_line))
        f_line = f.readline()
    f.close()

    # get total cluster number from input
    cluster_num = int(sys.argv[1])

    # # ============================================ #
    # # METHOD#1: generate centroid indexes randomly
    # centroid_index_list = random.sample(list(range(0, len(vector_list)-1)), cluster_num)    # a list of centroids' indexes
    # # generate the centroid list
    # centroid_list = list()
    # for centroid_index in centroid_index_list:
    #     centroid_list.append(vector_list[centroid_index])
    # # ============================================ #


    # ============================================ #
    # METHOD#2: generate centroids indexes by kmeans++
    centroid_list = kpp(vector_list, cluster_num)
    # ============================================ #

    # print centroid_list

    clusters = dict()  # save {class1: vectorList1; class2: vectorList2; ...}
    stop = False
    # for iter_i in range(0, iter_time):  # [0, iter_time-1]
    itr = 0
    while not stop:
        # print "CURRENT ITERATION: " + str(itr)
        itr += 1
        # cluster each vector
        clusters.clear()
        for v_index, vector in enumerate(vector_list):
            max_sim = sys.float_info.min
            classification_res = 0
            for c_index, centroid in enumerate(centroid_list):
                cur_dis = get_dis(vector, centroid)
                if cur_dis > max_sim:
                    max_sim = cur_dis
                    classification_res = c_index
            # NOW: vector is classified to the "classification_res"-th cluster
            # print "VECTOR INDEX -> CLASSIFICATION RES" + str(v_index) + " " + str(classification_res)

            if classification_res in clusters:
                # update
                clusters[classification_res].append(v_index)
            else:
                # add
                clusters[classification_res] = list()
                clusters[classification_res].append(v_index)
        # print clusters

        # recalculate the center for each cluster
        old_centroid_list = list(centroid_list)
        for key, cluster in clusters.iteritems():   # key is the cluster index, cluster is a list of vector_index
            # update the center for current cluster
            v1 = vector_list[cluster[0]]
            index = 1
            while index < len(cluster):
                v2 = vector_list[cluster[index]]

                i1 = 0
                i2 = 0

                while i1 < len(v1) and i2 < len(v2):
                    if v1[i1][0] == v2[i2][0]:
                        v1[i1] = (v1[i1][0], (v1[i1][1] + v2[i2][1]))
                        i1 += 1
                        i2 += 1
                    elif v1[i1][0] > v2[i2][0]:  # add [i2] in front of i1 & move i2 ++
                        v1.insert(i1, (v2[i2][0], v2[i2][1]))
                        i1 += 1
                        i2 += 1
                    else:  # v1[] < v2[]
                        i1 += 1

                    if i1 == len(v1) and i2 < len(v2):
                        while i2 < len(v2):
                            v1.append((v2[i2][0], v2[i2][1]))
                            i2 += 1

                index += 1
            v1 = [(item[0], float(item[1])/float((len(cluster)))) for item in v1]  # new centroid - use float
            centroid_list[key] = v1
        # stop iteration judgement: compare old centroid list & current centroid list
        converge = True
        for cur, old in zip(centroid_list, old_centroid_list):
            if (1 - get_dis(cur, old)) > 10e-3:  # 1 means the 2 centroids are the same
                converge = False
                break
        if converge:
            stop = True


    print "TOTAL ITERATION TIME: " + str(itr)
    # output the res
    for key, cluster in clusters.iteritems():   # key is the cluster index, cluster is a list of vector_index
        for index in cluster:
            print str(index) + " " + str(key)


# the function to generate vector list from string
def get_list(v_str):
    v_list = []
    v_arr = v_str.strip().split()
    for item in v_arr:
        item_arr = item.split(":")
        v_list.append((int(item_arr[0]), int(item_arr[1])))
    v_list.sort(key=lambda tup: tup[0])     # sort by word index, increasing
    return v_list


# the function to calculate the cosine distance between two vectors
def get_dis(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    i1 = 0
    i2 = 0
    res = 0.0
    while i1 < len(v1) and i2 < len(v2):
        if v1[i1][0] == v2[i2][0]:
            res += float(v1[i1][1]) * float(v2[i2][1])
            i1 += 1
            i2 += 1
        elif v1[i1][0] > v2[i2][0]:
            i2 += 1
        else:
            i1 += 1
    return res


# the function to normalize vector
def normalize(v):
    len = 0.0
    for pair in v:
        len += pair[1] * pair[1]
    len = float(math.sqrt(len))
    normalized_v = [(item[0], item[1]/len) for item in v]
    return normalized_v


# the function to generate initial centroids by kmeans++
def kpp(vector_list, cluster_num):
    vector_list_len = len(vector_list)
    centroid_index_list = []
    centroid_list = []
    sim_list = [sys.float_info.min] * vector_list_len   # save all vectors' similarities

    # 1. select the first centroid randomly
    first_cindex = random.sample(list(range(0, vector_list_len-1)), 1)[0]
    centroid_index_list.append(first_cindex)
    centroid_list.append(vector_list[first_cindex])

    # get all cluster_num centroids
    for i in range(1, cluster_num):
        # calculate the sim of each vector & the last centroid in the list
        centroid = centroid_list[-1]
        for v_index, vector in enumerate(vector_list):
            cur_sim = get_dis(vector, centroid)
            sim_list[v_index] = max(cur_sim, sim_list[v_index])
        # NOW: sim_list[i] is vector_list[i]'s max sim with all current centroids
        # do sampling & generate next centroid
        sam_list = map(lambda x: (1-x)**2, sim_list)
        sam_sum = sum(sam_list)
        # normalize
        norm_sam_list = map(lambda x: float(x)/sam_sum, sam_list)
        next_cindex = np.random.choice(vector_list_len, 1, p=norm_sam_list)
        centroid_index_list.append(next_cindex)
        centroid_list.append(vector_list[next_cindex])
    return centroid_list


if __name__ == '__main__':
    main(sys.argv[1:])