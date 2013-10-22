'''
Created on May 21, 2013

@author: Georgiana Dinu, Pham The Nghia
'''
from composes.exception.invalid_argument_error import InvalidArgumentError

def assert_is_instance(object_, class_):
    if not isinstance(object_, class_):
        raise TypeError("expected %s, received %s" % (class_, type(object_)))


def assert_valid_kwargs(kwargs, string_list):

    for key in kwargs.keys():
        if not key in string_list:
            raise InvalidArgumentError("Invalid parameter: %s" % key)


def get_partitions(sorted_list, min_samples):

    prev_idx = 0
    range_list = []
    for i in range(1, len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            if i - prev_idx >= min_samples:
                range_list.append((prev_idx, i))

            prev_idx = i

    if len(sorted_list) - prev_idx >= min_samples:
        range_list.append((prev_idx, len(sorted_list)))

    keys = [sorted_list[range_list[i][0]] for i in xrange(len(range_list))]

    return keys, range_list