#!usr/bin/env python
# coding:utf-8
import sys
sys.path.append('/home/wib/dl/pytorch/caffe2/python')
from caffe2.python import net_drawer
import cv2
import subprocess
import os


def visualizeNet(input_net):
    graph = net_drawer.GetPydotGraph(input_net, rankdir="LR")
    graph.write_png("tempNet.png")
    img1 = cv2.imread("tempNet.png", 1)
    cv2.imshow("tempNet", img1)
    cv2.waitKey(0)


def visual_dot_file(input_file):

    base_dir = os.path.dirname(input_file)
    file_name_with_suffix = os.path.basename(input_file)
    file_name_index = file_name_with_suffix.rfind('.')
    file_name = os.path.join(base_dir, file_name_with_suffix[0:file_name_index])
    subprocess.call('dot -Tpdf {} -o {}.pdf'.format(input_file, file_name), shell=True)
    if os.path.isfile(file_name):
        os.remove(file_name)
    print '[DOT-PDF]save to:{}.pdf'.format(file_name)


def visual_model(input_model, save_dir):
    g = net_drawer.GetPydotGraph(input_model, rankdir="TB")
    dot_path = os.path.join(save_dir, input_model.Proto().name + '.dot')
    g.write_dot(dot_path)
    print '[WRITE-DOT-FILE]:{}'.format(dot_path)
    visual_dot_file(dot_path)
