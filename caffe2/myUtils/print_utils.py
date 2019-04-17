#!/usr/bin/emv python
# coding:utf-8

class ProcessPrinter(object):
    def __init__(self, input_total_num, input_print_times=10):
        self.__print_times = input_print_times
        self.__print_stride = 0
        self.total_num = input_total_num
        self.cal_print_stride()

    def cal_print_stride(self):
        self.__print_stride = max(1, int(self.total_num / (self.__print_times + 0.00001) + 0.5))

    def process_print(self, input_cur_times):
        if input_cur_times % self.__print_stride == 0:
            cur_percentage = float(self.total_num - input_cur_times) / self.total_num * 100
            print('total number:{0: <10d}   cur num:{1: <10d}      percentage:{2: <.3f}'.format(self.total_num,
                                                                                                input_cur_times,
                                                                                                cur_percentage))
