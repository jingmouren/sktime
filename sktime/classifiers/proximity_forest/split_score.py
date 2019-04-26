import math


def gini(self, root, *nodes):
    root_score = gini_node(self, root)
    root_num_instances = node_size(self, root)
    node_score_sum = 0
    for node in nodes:
        node_score = gini_node(self, node)
        node_num_instances = node_size(self, node)
        node_score *= (node_num_instances / root_num_instances)
        node_score_sum += node_score
    score = root_score - node_score_sum
    return score

def gini_node(self, binned_instances):
    num_instances = node_size(self, binned_instances)
    score = 1
    for class_label, instances in binned_instances.items():
        proportion = len(instances) / num_instances
        score -= math.pow(proportion, 2)
    return score

def ig(self, root, *nodes):
    raise Exception('not implemented yet')

def chi(self, root, *nodes):
    raise Exception('not implemented yet')

def node_size(self, node):
    num_instances = 0
    for class_label, instances in node.items():
        num_instances += len(instances)
    return num_instances