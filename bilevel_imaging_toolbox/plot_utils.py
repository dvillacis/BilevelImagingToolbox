import numpy as np
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

def plot_history(values):
    plt.plot(values)
    plt.grid()
    plt.show()

def plot_collection(values_list, values_names, title="",save_tikz=False,tikz_path=""):

    plt.style.use('ggplot')

    worst = np.max(values_list)
    db = lambda v : 10*np.log(v/worst)

    fig,ax = plt.subplots()
    i = 0
    for v in values_list:
        ax.plot(db(v),label=values_names[i])
        i+=1

    ax.legend()
    #ax.title = title
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function (log scaled)')
    #plt.show()
    plt.grid(True)
    if save_tikz:
        tikz_save(tikz_path+title+'.tex')
