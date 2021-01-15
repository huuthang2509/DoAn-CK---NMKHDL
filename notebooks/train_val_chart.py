import matplotlib.pyplot as plt
import numpy as np

def chart(E_train, E_val, hyper_param,str_label):
    fig, ax1 = plt.subplots()

    ax1.plot(hyper_param, E_train, 'g-o',label='E_train')
    ax1.plot(hyper_param, E_val, 'r-o', label='E_val')


    ax1.set_xlabel(str_label)
    ax1.set_ylabel('Error')
    ax1.set_xscale('log', basex=2)  # re-scale to have equal xtick space
    ax1.set_xticks(hyper_param)               # re-scale to have equal xtick space
    ax1.set_xticklabels(hyper_param)          # re-scale to have equal xtick space
    ax1.legend(loc='upper center')
    ax1.grid()