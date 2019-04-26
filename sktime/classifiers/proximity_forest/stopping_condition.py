def pure(self, node): # can we drop the self param? it's needless
    non_zero = False
    for class_label, instances in node.items():
        if len(instances) > 0:
            if non_zero:
                return False
            else:
                non_zero = True
    return True
