import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA


def plot_scatter(data,
                 label=None,
                 title=None,
                 x_label='x',
                 y_label='y',
                 show_index=None,
                 pca=True,
                 n_components=2,
                 save_path=None):
    """
    Plot 2nd diem data by scatter.
    use pca decrease dimension default(use `MDS` by `pca=False`).

    :param data: instance data
    :param label: instance label
    :param title: plot title
    :param x_label: plot x_label
    :param y_label: plot y_label
    :param show_index: plot instance index
    :param pca: use pca default, False to use `MDS`
    :param n_components: plot by 2 or 3 dimension
    :param save_path: save plot path
    :return: None
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # dimensionality reduction
    if data.shape[1] > n_components:
        if pca:
            pca = PCA(n_components=n_components)
        else:
            pca = MDS(n_components=n_components, max_iter=99, n_init=1)
        data = pca.fit_transform(scale(data))

    # label to color
    if label is None:
        label = 'k'
        label_count = 1
    else:
        label_set = set(label)
        label_count = len(label_set)
        colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(label_set)))
        colors = {k: v for k, v in zip(label_set, colors)}
        label = [colors[i] for i in label]

    # plot instance
    fig = plt.figure()
    if n_components != 2:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=label)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(x=data[:, 0], y=data[:, 1], c=label)

        # plot index
        if show_index is not None:
            if show_index is True:
                show_index = list(range(data.shape[0]))

            for index, x, y in zip(show_index, data[:, 0], data[:, 1]):
                plt.annotate(index, (x, y), alpha=0.15)

    title_text = 'instance count: %d, label count: %d' % (data.shape[0], label_count)
    if title is not None:
        title_text = '\n'.join([title, title_text])

    plt.title(title_text)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if save_path is not None:
        plt.savefig('%s\%s.png' % (save_path, title), format='png')
    else:
        # plt.show()
        pass
