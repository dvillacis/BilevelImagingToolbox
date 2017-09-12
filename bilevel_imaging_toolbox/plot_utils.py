import matplotlib.pyplot as plt

def plot_history(values):
    plt.plot(values)
    plt.grid()
    plt.show()

def plot_collection(values_list, values_names):
    for v in values_list:
        plt.plot(v)
    plt.grid()
    plt.show()
