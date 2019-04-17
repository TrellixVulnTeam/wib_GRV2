#!/usr/bin/env python
# coding:utf-8

import os
import subprocess
import shutil


def get_suffix(input_path):
    dot_position = input_path.rfind('.')
    if dot_position < 0:
        return None
    else:
        return input_path[dot_position + 1:]


def get_filename(input_path, with_suffix=False):
    base_name = os.path.basename(input_path)
    if with_suffix:
        return base_name
    else:
        dot_position = base_name.rfind('.')
        return base_name[0:dot_position]



def load_txt_to_list(input_txt_path, output_line_list):

    file_pid = open(input_txt_path, 'r')
    cur_line = file_pid.readline()
    count = 1
    valid_count = 1
    while cur_line:
        count += 1
        cur_line = cur_line.strip()
        if cur_line == '':
            cur_line = file_pid.readline()
            continue
        output_line_list.append(cur_line)
        cur_line = file_pid.readline()
        valid_count += 1
    print('Total lines:{}       Valid lines:{}'.format(count, valid_count))


def load_text_to_queue(input_txt_path, output_line_queue):
    file_pid = open(input_txt_path, 'r')
    cur_line = file_pid.readline()
    count = 1
    valid_count = 1
    while cur_line:
        count += 1
        cur_line = cur_line.strip()
        if cur_line == '':
            cur_line = file_pid.readline()
            continue
        output_line_queue.put(cur_line)
        cur_line = file_pid.readline()
        valid_count += 1
    print('Total lines:{0: <10d}       Valid lines:{1: <10d}'.format(count, valid_count))


def collect_file_path_to_list(input_dir, output_path_list, suffixs=None):

    count = 0
    for cur_file_name in sorted(os.listdir(input_dir)):
        cur_path = os.path.join(input_dir, cur_file_name)

        if os.path.isfile(cur_path):
            if suffixs is not None:
                if get_suffix(cur_file_name) not in suffixs:
                    continue
            else:
                cur_path = os.path.join(input_dir, cur_file_name)

            count += 1
            output_path_list.append(os.path.abspath(cur_path))
        else:
            collect_file_path_to_list(cur_path, output_path_list, suffixs)

    print('File num:{0: <10d}\tFile path:{1}'.format(count, input_dir))


def collect_file_path_recurisive_to_list(input_dir, output_list, suffixs=None):

    count = 0
    for cur_file_name in os.listdir(input_dir):
        cur_file_path = os.path.join(input_dir, cur_file_name)
        if os.path.isdir(cur_file_path):
            collect_file_path_recurisive_to_list(cur_file_path, output_list, suffixs)
        else:
            if suffixs is not None:
                if get_filename(cur_file_name) is suffixs:
                    output_list.append(cur_file_path)
                    count += 1
            else:
                output_list.append(cur_file_path)
                count += 1

    print('Total Collect Num:{0: <10d}\tFile Path:{1}'.format(len(output_list), input_dir))