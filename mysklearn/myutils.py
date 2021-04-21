import random
import math
def create_dictionary(lst):
    freq_lst = {}
    for item in lst:
        if item in freq_lst:
            freq_lst[item] += 1
        else:
            freq_lst[item] = 1
    return freq_lst

def all_same_class(instances):
    #print(instances)
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def select_attribute(instances, available_attributes):
    # for now, we are going to select an attribute randomly
    # TODO: come back after you can build a tree with 
    # random attribute selection and replace with entropy
    # rand_index = random.randrange(0, len(available_attributes))
    # available_attributes[rand_index]

    # dictionary of attributes where value is entropy
    attribute_entropy = {}
    # get entropy for each attribute
    for attribute in available_attributes:
        index = int(attribute.replace("att", ""))
        item_lst = []
        num_lst = []
        # get item examples of the attribute (senior, mid, junior)
        for row in instances:
            if row[index] not in item_lst:
                item_lst.append(row[index])
        E_items = []
        # create dictionary of class probability
        for item in item_lst:
            class_dict = {}
            num = 0
            for row in instances:
                if row[index] == item:
                    num += 1
                    if row[-1] in class_dict:
                        class_dict[row[-1]] += 1
                    else:
                        class_dict[row[-1]] = 1
            num_lst.append(num)
        
            # calculate entropy figures for each type of attribute
            E_item = 0
            for i in class_dict:
                p = class_dict[i]/num
                if p != 0:
                    e = (p * math.log(p, 2))
                    E_item -= e
            E_items.append(E_item * (num / len(instances)))
        # average the values and add to a dictionary
        avg = sum(E_items)
        attribute_entropy[attribute] = avg 
    lowest_entropy_attribute = min(attribute_entropy, key=attribute_entropy.get) 
    return lowest_entropy_attribute

def partition_instances(instances, split_attribute, attribute_domains, header):
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
    # lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    # task: try this!
    attribute_domain.sort()
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def compute_partition_stats(instances, attribute, header):
    # turn instances at class index into a dictionary
    index = header.index(attribute)
    value_lst = []
    for row in instances:
        value_lst.append(row[index])
    value_dict = create_dictionary(value_lst)
    # get max value
    all_values = value_dict.values()
    max_value = max(all_values)
    # get all class labels with that value
    tie_lst = []
    for key in value_dict:
        if value_dict[key] == max_value:
            tie_lst.append(key)
    # break tie and return
    return random.choice(tie_lst), max_value

#def get_occurences(current_instances, available_attributes, attribute_value):
    #return 0

def tdidt(current_instances, available_attributes, attribute_domains, header):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes)
    #print("split_attribute", split_attribute)
    #print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute)

    # cannot split on the same attribute twice in a branch
    # recall: python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    #print("partitions:", partitions)

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        #print("working with partition for:", attribute_value)
        value_subtree = ["Value", attribute_value]
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print(all_same_class(partition))
            #print("CASE 1")
            leaf = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            value_subtree.append(leaf)
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            choice, num = compute_partition_stats(partition, split_attribute, header)
            leaf = ["Leaf", choice, num, len(current_instances)]
            value_subtree.append(leaf)
            tree.append(value_subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            return None
        else: # all base cases are false... recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            if subtree is None:
                # create leaf 
                choice, num = compute_partition_stats(partition, split_attribute, header)
                leaf = ["Leaf", choice, num, len(current_instances)]
                value_subtree.append(leaf)
            else:
                # need to append subtree to value_subtree and appropriately append value subtre
                # to tree
                value_subtree.append(subtree)
            tree.append(value_subtree)  
    return tree

def tdidt_predict(header, tree, instance):
    # returns "True" or "False" if a leaf node is hit
    # None otherwise 
    info_type = tree[0]
    if info_type == "Attribute":
        # get the value of this attribute for the instance
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # recurse, we have a match!!
                return tdidt_predict(header, value_list[2], instance)
    else: # Leaf
        return tree[1] # label


def sum_of_lst(lst):
    sum = 0
    for item in lst:
        sum += item
    return sum

def get_frequency():
    return

def get_bins(cutoffs):
    #stuff_in_string = "Shepherd {} is {} years old.".format(shepherd, age)
    bins = []
    for i in range(0, len(cutoffs) - 1):
        bin = "[{}, {})".format(cutoffs[i], cutoffs[i + 1])
        bins.append(bin)
    return bins

def compute_equal_width_cutoffs(values, num_bins):
    # first compute the range of the values
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    # bin_width is likely a float
    # if your application allows for ints, use them
    # we will use floats
    # np.arange() is like the built in range() but for floats
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    # optionally: might want to round
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 
    
def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1
    
    return freqs

def remove_NA(column):
    clean = []
    for i in column:
        if i != "N/A":
            if i != "NA":
                if i != "":
                    clean.append(i)
    return clean

def convert_mpg_to_rating(table):
    column = table.get_column("mpg", False)
    column.sort()
    rating_lst = []
    for instance in column:
        if instance >= 45:
            rating_lst.append(10)
        elif instance >= 37:
            rating_lst.append(9)
        elif instance >= 31:
            rating_lst.append(8)
        elif instance >= 27:
            rating_lst.append(7)
        elif instance >= 24:
            rating_lst.append(6)
        elif instance >= 20:
            rating_lst.append(5)
        elif instance >= 17:
            rating_lst.append(4)
        elif instance >= 15:
            rating_lst.append(3)
        elif instance == 14:
            rating_lst.append(2)
        else:
            rating_lst.append(1)
    return rating_lst

def get_regression_vals(x_col, y_col):
    x_avg = sum(x_col) / len(x_col)
    y_avg = sum(y_col) / len(y_col)
    numerator = 0
    denominator = 0
    m = 0

    for i in range(0, len(x_col)):
        numerator += (x_col[i] - x_avg) * (y_col[i] - y_avg)
    
    for i in range(0, len(x_col)):
        denominator += (x_col[i] - x_avg) ** 2
    
    m = numerator / denominator 
    b = y_avg - m * x_avg

    numerator_r = numerator
    denominator_r = 0
    denominator_r_x = 0
    denominator_r_y = 0

    for i in range(0, len(x_col)):
        denominator_r_x += (x_col[i] - x_avg) ** 2

    for i in range(0, len(x_col)):
        denominator_r_y += (y_col[i] - y_avg) ** 2
    
    denominator_r = math.sqrt(denominator_r_x * denominator_r_y)

    r = numerator_r / denominator_r
    c = numerator / len(x_col)

    return m, b, r, c

