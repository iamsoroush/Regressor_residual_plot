import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_data(X, y):
	"""A function for splitting Data to train and test sets in a stratified manner
	"""
    bins = np.linspace(0, len(y), 100)
    y_binned = np.digitize(y, bins)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y_binned)
    return x_train, x_test, y_train, y_test

def residual_plot(model, X, Y):
    """This function plots residual-plot for a regressor.
	
	X, y : np.ndarray
	model : estimator object. Should have 'fit' and 'predict' methods.
	"""
    x_train, x_test, y_train, y_test = split_data(X, Y)
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    res_train = y_train - y_pred_train
    res_test = y_test - y_pred_test
    
    fig, [ax0, ax1] = plt.subplots(2, 1, figsize=(14, 10))
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    tableau20 = [(i[0]/255., i[1]/255., i[2]/255.) for i in tableau20]
    %matplotlib inline
    
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    ################################
    # Plot res-plot for training set
    x = StandardScaler().fit_transform(y_pred_train.reshape(-1, 1))
    y = StandardScaler().fit_transform(res_train.reshape(-1, 1))
    fig1 = plt.figure(figsize=(14, 10))
    fig1.suptitle('Residual plot for training set')
    
    # start with a rectangular Figure
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # the scatter plot:
    axScatter.scatter(x, y, color=tableau20[0], alpha=0.5)
    
    # now determine nice limits by hand:
    n_bins = 100

    x_limp = x.max() + x.std()
    x_limn = x.min() - x.std()
    y_limp = y.max() + y.std()
    y_limn = y.min() - y.std()

    axScatter.set_xlim((x_limn, x_limp))
    axScatter.set_ylim((y_limn, y_limp))
    axScatter.set_xlabel('Estimated output')
    axScatter.set_ylabel('Residuals')

    axHistx.hist(x, bins=n_bins, color=tableau20[1], alpha=0.75)
    axHisty.hist(y, bins=n_bins, orientation='horizontal', color=tableau20[2], alpha=0.75)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    
    ################################
    # Plot res-plot for testing set
    x = StandardScaler().fit_transform(y_pred_test.reshape(-1, 1))
    y = StandardScaler().fit_transform(res_test.reshape(-1, 1))
    fig2 = plt.figure(figsize=(14, 10))
    fig2.suptitle('Residual plot for testing set')
    
    # start with a rectangular Figure
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # the scatter plot:
    axScatter.scatter(x, y, color=tableau20[0], alpha=0.5)
    
    # now determine nice limits by hand:
    n_bins = 100

    x_limp = x.max() + x.std()
    x_limn = x.min() - x.std()
    y_limp = y.max() + y.std()
    y_limn = y.min() - y.std()

    axScatter.set_xlim((x_limn, x_limp))
    axScatter.set_ylim((y_limn, y_limp))
    axScatter.set_xlabel('Estimated output')
    axScatter.set_ylabel('Residuals')

    axHistx.hist(x, bins=n_bins, color=tableau20[1], alpha=0.75)
    axHisty.hist(y, bins=n_bins, orientation='horizontal', color=tableau20[2], alpha=0.75)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    plt.show()