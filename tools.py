import numpy as np

def set_split(X, y, proportion, expend=False):
    index = np.arange(len(X))
    np.random.shuffle(index)
    sum_p = sum(proportion)
    proportion = [p / sum_p for p in proportion]
    prop=0
    N = len(index)
    X_splits = []
    y_splits = []
    for i in range(len(proportion)):
        start_index = round( N*prop )
        end_index = round( N*(prop+proportion[i]) )
        
        set_index = index[start_index: end_index]
        X_splits.append( X[set_index] )
        y_splits.append( y[set_index] )
        
        prop += proportion[i]
    
    if expend:
        return *X_splits, *y_splits
    else:
        return X_splits, y_splits
    