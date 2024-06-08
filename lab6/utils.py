import matplotlib.pyplot as plt

def tqdm_bar(pbar, desp):
    pbar.set_description(desp , refresh=False)
    pbar.refresh()
    
def plot_curve(title, epoch, y1, curve1, y2=None, curve2=None, path=None, loc='upper right'):
    x = range(0, epoch)
    plt.title(title)
    plt.plot(x, y1, 'b', label=curve1)
    if y2:
        plt.plot(x, y2, 'r', label=curve2)
    plt.legend(loc=loc)
    plt.savefig(path)
    plt.clf()