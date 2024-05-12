import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

df = sns.load_dataset("mpg")

colors = ["#ff6347", "#4682b4"]  # トマト色からスチールブルー

# LinearSegmentedColormapを使用して連続カラーパレットを作成します。
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

df_plot = df[["mpg", "cylinders", "displacement"]].head(20)

fig, axes = plt.subplots(ncols=3)
axes = axes.flatten()
for ax, origin in zip(axes, df["origin"].unique()):
    ax = sns.heatmap(df_plot[df.head(20)["origin"]==origin], cmap=custom_cmap, ax=ax, cbar=False, vmin=0, vmax=500)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

# カラーバーの追加
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # カラーバーの位置とサイズを調整
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=500))
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax)

# plt.tight_layout()
