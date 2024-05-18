import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import math


# Download stock data
df = yf.download("AAPL", start="2020-01-01", end="2021-01-01").reset_index()
a = np.arange(1, 254) % 10
df["category"] = a
df["category"] = df["category"].astype(str)
df["Date"] = pd.to_datetime(df["Date"])
df["week"] = df["Date"].dt.isocalendar().week
df["week"] = df["week"] % 6
df = df.loc[df["week"] < 5]
df["week"] = df["week"].astype(str)
df.head()
df = df.sort_values("Open").reset_index(drop=True)

def plot_scatter_by_category(df, category, x, y, border, title, pptx):
    sns.set_theme(context="notebook", style="darkgrid")
    df_id_merge = df[["id", x]].value_counts().reset_index()[["id", x]]
    figure, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharey=True)
    plt.subplots_adjust(hspace=0.3)
    axes = iter(axes.flatten())
    color = sns.color_palette("deep", n_colors=df[category].nunique())
    for cat, c in zip(sorted(df[category].unique()), color):
        ax = next(axes)
        df_plot = df[df[category]==cat].sort_values(x)
        df_agg = df_plot.groupby("id")[y].mean().reset_index()
        df_agg = df_agg.merge(df_id_merge, on="id")
        ax = sns.scatterplot(df_agg, x=x, y=y, ax=ax, color=c)
        df_moving_average = df_plot.groupby("id")[y].mean().reset_index()
        df_moving_average = df_moving_average.merge(df_id_merge, on="id").drop("id", axis=1)
        df_moving_average = df_moving_average.sort_values(x).rolling(window=3, on=x).mean()
        ax = sns.lineplot(df_moving_average, x=x, y=y, ax=ax)
        ax.set_title(cat)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        # ax.set_xlabel(None)
        # ax.set_xticklabels([])
        # ax.hlines(border, xmin=-0.5, xmax=df_plot[x].nunique()-0.5, colors="black")

        # describe = df_plot.groupby(x)[[y]].describe().T.reset_index().drop(columns="level_0").set_index("level_1").rename_axis(None, axis=0)
        # describe = describe.map(lambda x: f"{x:.3g}")
        # describe = describe.loc[["count", "mean", "50%"]]
        # describe.index = ["Count", "Avg", "Median"]
        # table = ax.table(cellText=describe.values,
        #                     colLabels=["week "+col for col in describe.columns],
        #                     rowLabels=describe.index,
        #                     rowColours=["darkgray"]*describe.shape[0],
        #                     colColours=["darkgray"]*describe.shape[1],
        #                     cellColours=[["white"]*describe.shape[1] if i%2==0 else ["lightgray"]*describe.shape[1] for i in range(describe.shape[0])],
        #                     )
        # table.auto_set_font_size(False)
        # table.set_fontsize(15) 
        # table.scale(1, 1.6)
    
    figure.suptitle(title, fontsize=30)

    for ax in axes:
        ax.set_visible(False)
    
    figure.savefig(f"./{title}.png")
    figure.tight_layout()
    plt.show()
    # pptx.save(f"./{title}.png")

# plot_scatter_by_category(df, "category", x="week", y="Open", title="Title", pptx=None)


def plot_scatters_by_category(df, category_col, x, y, title, pptx):
    total_rows = math.ceil(df[category_col].nunique() / 3)
    n_figures = math.ceil(total_rows / 2)
    uniques = sorted(df[category_col].unique())
    for i in range(n_figures):
        unique = uniques[i*6:(i+1)*6]
        df_plot = df[df[category_col].isin(unique)]
        if n_figures:
            plot_scatter_by_category(df_plot, category_col, x, y, 100, title+f"_{i+1}", pptx)
        else:
            plot_scatter_by_category(df_plot, category_col, x, y, 100, title, pptx)

plot_scatters_by_category(df, "category", "Date", "Open", "A", None)