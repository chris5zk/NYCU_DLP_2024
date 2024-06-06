
def tqdm_bar(pbar, desp):
    pbar.set_description(desp , refresh=False)
    pbar.refresh()