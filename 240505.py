from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn import init
import torchvision
from torchvision import transforms


df = sns.load_dataset("mpg")
g = sns.FacetGrid(data=df, col="origin")

# ボックスプロットを上に重ねてプロット
for ax in g.axes.flat:
    sns.histplot(y=df["mpg"], ax=ax)
    ax2 = ax.twiny()
    sns.boxplot(y=df["mpg"], ax=ax2)


def cramers_v(x, y):
    cont_table = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(cont_table, correction=False)[0]
    min_d = min(cont_table.shape) - 1
    n = len(x)
    v = np.sqrt(chi2/(min_d*n))
    return v


import seaborn as sns
import matplotlib.pyplot as plt

# サンプルデータセットを読み込む
tips = sns.load_dataset('tips')

# FacetGridを作成する
g = sns.FacetGrid(tips, col='day', aspect=0.5)

# 各サブプロットにヒストグラムを描画する
# g.map_dataframe(sns.histplot, y='total_bill', bins=10)

# 左端以外のサブプロットのy軸の色を変更する
for ax in g.axes.flat:
    sns.histplot(y=df["tip"], ax=ax, alpha=0.5, element="step")
    ax2 = ax.twiny()
    sns.boxplot(y=df["tip"], ax=ax2, width=0.3, boxprops={"alpha": 0.7})


    if ax != g.axes.flat[0]:
        ax.spines['left'].set_color('white')
        ax.tick_params(axis='y', colors='white')
        
        ax2.spines['left'].set_color('white')
        ax2.tick_params(axis='x', colors='white')
    ax2.spines['top'].set_color('white')
    ax2.spines['right'].set_color('white')

# 結果を表示する
plt.show()

class GreedyFeatureSelection():
    """
    # Greedy feature selection
    gfs = GreedyFeatureSelection(pipeline=pipeline, cv=cv)
    gfs.select_feature(X, y)
    # スコアの結果と選択された特徴量を確認
    print(gfs.scores)
    print(gfs.selected_features)
    """

    def __init__(self, pipeline, cv):
        self.pipeline = pipeline
        self.cv = cv
        self.selected_features = []
        self.scores = [0]

    def select_feature(self, X, y):

        all_features = X.columns

        while True:
            # print('greedy selection started')
            best_score = self.scores[-1]
            candidate_feature = None
            for feature in all_features:
                if feature in self.selected_features:
                    continue
                # print(f'{feature} started')
                features = self.selected_features + [feature]
                X_train = X[features]
                # 評価
                score = cross_val_score(
                    self.pipeline, X_train, y, cv=self.cv).mean()
                # print(f'{features} score: {score}')
                if score > best_score:
                    # print(f'best score updated {best_score} -> {score}')
                    best_score = score
                    candidate_feature = feature

            if candidate_feature is not None:
                # print(f'========{candidate_feature} is selected=============')
                self.scores.append(best_score)
                self.selected_features.append(candidate_feature)
            else:
                break


# カスタムのDatasetを作る
class MyDataset(Dataset):  # Datasetクラスを継承する
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        return X, y


conv_model = nn.Sequential(
    # 1x28x28
    nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    # 4x14x14
    nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    # 8x7x7
    nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    # 16x4x4
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    # 32x2x2 -> GAP -> 32 x 1 x 1

    nn.Flatten(),
    # # 128 -> 32
    nn.Linear(128, 10)
    # nn.Linear(32, 10)
    # 10
)

# kaiming初期化
for layer in conv_model:
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight)


class ActivationStatistics:
    """
    A utility class for gathering statistics on activations produced by ReLU layers in a model.

    Attributes:
    - model (nn.Module): The model whose activations are to be tracked.
    - act_means (List[List[float]]): List of means for each ReLU layer's activations.
    - act_stds (List[List[float]]): List of standard deviations for each ReLU layer's activations.

    Methods:
    - register_hook: Register hooks on ReLU layers of the model to gather statistics.
    - save_out_stats: Callback method to save statistics of activations.
    - get_statistics: Return collected activation means and standard deviations.
    - plot_statistics: Plot the activation statistics using matplotlib.

    Usage:
        model = ... # some PyTorch model
        act_stats = ActivationStatistics(model)
        ... # Run the model, gather statistics
        act_stats.plot_statistics()
    """

    def __init__(self, model):
        self.model = model
        self.act_means = [[]
                          for module in self.model if isinstance(module, nn.ReLU)]
        self.act_stds = [[]
                         for module in self.model if isinstance(module, nn.ReLU)]
        self.register_hook()

    def register_hook(self):
        relu_layers = [
            module for module in self.model if isinstance(module, nn.ReLU)]
        for i, relu in enumerate(relu_layers):
            relu.register_forward_hook(partial(self.save_out_stats, i))

    def save_out_stats(self, i, module, inp, out):
        self.act_means[i].append(out.detach().cpu().mean().item())
        self.act_stds[i].append(out.detach().cpu().std().item())

    def get_statistics(self):
        return self.act_means, self.act_stds

    def plot_statistics(self):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        for act_mean in self.act_means:
            axs[0].plot(act_mean)
        axs[0].set_title('Activation means')
        axs[0].legend(range(len(self.act_means)))

        for act_std in self.act_stds:
            axs[1].plot(act_std)
        axs[1].set_title('Activation stds')
        axs[1].legend(range(len(self.act_stds)))

        plt.show()


def lr_finder(model, train_loader, loss_func, lr_multiplier=1.2):
    """
    Find an optimal learning rate using the learning rate range test.

    Parameters:
    - model: PyTorch model.
    - train_loader: DataLoader for training data.
    - loss_func: PyTorch loss function.
    - lr_multiplier: Multiplier to increase the learning rate at each step.

    Returns:
    - lrs: List of tested learning rates.
    - losses: List of losses corresponding to the learning rates.
    """
    lr = 1e-8
    max_lr = 10
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    lrs = []

    for train_batch, data in enumerate(train_loader):
        X, y = data

        opt.zero_grad()
        # forward
        preds = model(X)
        loss = loss_func(preds, y)
        losses.append(loss.item())
        lrs.append(lr)

        # backward
        loss.backward()
        opt.step()

        lr *= lr_multiplier

        for param_group in opt.param_groups:
            param_group['lr'] = lr
        if lr > max_lr:
            break

    return lrs, losses
