import matplotlib.pyplot as plt

def tqdm_bar(pbar, desp):
    pbar.set_description(desp , refresh=False)
    pbar.refresh()
    
def plot_curve(title, epoch, y1, y2, curve1, curve2, path, loc='upper right'):
    x = range(0, epoch)
    plt.title(title)
    plt.plot(x, y1, 'b', label=curve1)
    plt.plot(x, y2, 'r', label=curve2)
    plt.legend(loc=loc)
    plt.savefig(path)
    plt.clf()