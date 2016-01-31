__author__ = 'luoshalin'

def main():
    # CORPUS STAS FOR THE DEV SET
    dev_file_name = "../HW2_data/HW2_dev.docVectors"
    f_dev = open(dev_file_name)
    line = f_dev.readline()
    first_vector = get_list(line)
    doc_count = 0           # the total number of documents
    word_count = 0          # the total number of words
    word_set = set()        # the total number of unique words
    uniq_word_count = 0     # the average number of unique words per document
    while '' != line:
        vector = get_list(line)
        doc_count += 1
        uniq_word_count += len(vector)
        for item in vector:
            word_count += item[1]
            if item[0] not in word_set:
                word_set.add(item[0])
        line = f_dev.readline()
    f_dev.close()

    print "==========/ DEV DOC: /=========="
    print "Total number of documents:"
    print doc_count
    print "Total number of words:"
    print word_count
    print "Total number of unique words:"
    print len(word_set)
    print "Avg number of uniq words per doc:"
    print uniq_word_count / doc_count

    # CORPUS STAS FOR THE TEST SET
    test_file_name = "../HW2_data/HW2_test.docVectors"
    f_test = open(test_file_name)
    line = f_test.readline()
    doc_count = 0           # the total number of documents
    word_count = 0          # the total number of words
    word_set = set()        # the total number of unique words
    uniq_word_count = 0     # the average number of unique words per document
    while '' != line:
        vector = get_list(line)
        doc_count += 1
        uniq_word_count += len(vector)
        for item in vector:
            word_count += item[1]
            if item[0] not in word_set:
                word_set.add(item[0])
        line = f_test.readline()
    f_test.close()

    print "==========/ TEST DOC: /=========="
    print "Total number of documents:"
    print doc_count
    print "Total number of words:"
    print word_count
    print "Total number of unique words:"
    print len(word_set)
    print "Avg number of uniq words per doc:"
    print uniq_word_count / doc_count

    # CORPUS STAS FOR THE FIRST DOC IN DEV SET
    twice_list = list()
    for item in first_vector:
        if item[1] == 2:
            twice_list.append(item[0])
    print "==========/ FIRST DOC IN DEV SET: /=========="
    print "Total number of unique words:"
    print len(first_vector)
    print "Word ids that occurred exactly twice in the document"
    print twice_list


# the function to generate vector list from string
def get_list(v_str):
    v_list = []
    v_arr = v_str.strip().split()
    for item in v_arr:
        item_arr = item.split(":")
        v_list.append((int(item_arr[0]), int(item_arr[1])))
    v_list.sort(key=lambda tup: tup[0])     # sort by word index, incrreasing
    return v_list


if __name__ == '__main__':
    main()