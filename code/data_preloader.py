# -*- coding: utf-8 -*-
# @Time : 2024/4/10 4:48 下午
# @Author : Tony
# @File : data_preloader.py
# @Project : contagion_dynamics

from itertools import combinations

"""
Data preloader of the empirical datasets
"""

def read_file(filename):
    with open(filename, 'r') as file:
        return [int(line.strip()) for line in file.readlines()]


def write_file(filename, data):
    with open(filename, 'w') as file:
        for line in data:
            file.write(' '.join(map(str, line)) + '\n')


def integrate_files(nverts_filename, simplices_filename, output_filename):
    nverts = read_file(nverts_filename)
    simplices = read_file(simplices_filename)

    res_hpe = []
    index = 0
    for vertices in nverts:
        res_hpe.append(simplices[index:index + vertices])
        index += vertices

    write_file(output_filename, res_hpe)


def remove_duplicates(input_file, output_file):
    unique_lines = set()

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        unique_lines.add(line.strip())

    with open(output_file, 'w') as f:
        for line in unique_lines:
            f.write(line + '\n')


def generate_pairs(line, n):
    pairs = set()
    for pair in combinations(sorted(line), n):
        pairs.add(tuple(pair))
    return pairs


def process_file(input_file, output_file, n):
    pairs_set = set()
    with open(input_file, 'r') as f:
        for line in f:
            elements = list(map(int, line.split()))
            elements.sort()
            if len(elements) > n:
                pairs = generate_pairs(elements, n)
                pairs_set.update(pairs)
            else:
                pairs_set.add(tuple(elements))

    pairs_list = sorted(list(pairs_set))

    with open(output_file, 'w') as f:
        for pair in pairs_list:
            f.write(' '.join(map(str, pair)) + '\n')







