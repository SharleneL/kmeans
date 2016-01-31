import sys
import random
import math

# input format: python kmeans.py [cluster_num]

file_name = "../HW2_data/HW2_dev.docVectors"
# file_name = "../HW2_data/fake.docVectors"


def main(argv):
    # save the file into a normalized vector list
    vector_list = list()
    f = open(file_name)
    f_line = f.readline()
    while '' != f_line:
        # vector_list.append(normalize(get_list(f_line)))
        vector_list.append(get_list(f_line))
        f_line = f.readline()
    f.close()

    # get total cluster number from input
    cluster_num = int(sys.argv[1])
    # generate centroid indexes randomly
    # JI DE GAI HUI LAI!!!!
    centroid_index_list = random.sample(list(range(0, len(vector_list)-1)), cluster_num)    # a list of centroids' indexes
    # centroid_index_list = [0, 3]    # a list of centroids' indexes
    # generate the centroid list
    centroid_list = list()
    for centroid_index in centroid_index_list:
        centroid_list.append(vector_list[centroid_index])
    # print "CENTROID LIST:"
    # print centroid_index_list

    # TEST: get_list func
    # print "================V1==================="
    # test1 = "2:1 4:1 3:1 1:1 5:0"
    # print normalize(get_list(test1))
    # test2 = "4:8 3:6"
    # print normalize(get_list(test2))
    # print get_dis(normalize(get_list(test1)), normalize(get_list(test2)))
    # v1 = vector_list[0]
    # print v1
    # vn1 = normalize(v1)
    # print vn1
    # print "================V2==================="
    # print linecache.getline(file_name, 0)
    # v2 = vector_list[1]
    # print v2
    # vn2 = normalize(v2)
    # print vn2
    # print "================DIS==================="
    # print get_dis(v1, v2)
    # TEST END

    iter_time = 20
    clusters = dict()  # save {class1: vectorList1; class2: vectorList2; ...}
    for iter_i in range(0, iter_time):  # [0, iter_time-1]
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
            # print str(v_index) + " " + str(classification_res)

            if classification_res in clusters:
                # update
                clusters[classification_res].append(v_index)
            else:
                # add
                clusters[classification_res] = list()
                clusters[classification_res].append(v_index)
        # print clusters

        # recalculate the center for each cluster
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
            v1 = [(item[0], item[1]/(len(cluster))) for item in v1]  # new centroid
            # print "NEW CENTROID FOR CLUSTER " + str(key)
            # print v1
            centroid_list[key] = v1
        # print "====================/ ITERATE TIME #" + str(iter_i) + " /===================="
        # print centroid_list[0]

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
    v_list.sort(key=lambda tup: tup[0])     # sort by word index, incrreasing
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

if __name__ == '__main__':
    main(sys.argv[1:])