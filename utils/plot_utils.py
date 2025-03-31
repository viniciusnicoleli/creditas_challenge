from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.ticker import FuncFormatter
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp
import seaborn as sns
import pandas as pd
import numpy as np
import joypy
import os

def plot_result_metric(df_plot, xlabel, ylabel,title, huer=None):
    plt.figure(figsize=(12,8))
    sns.lineplot(data=df_plot, x='value', y='metric', hue=huer)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

class plotter:
    def plotar_nested_boxplot(df: pd.DataFrame, cols: list) -> None:
        """Função que plota gráfico de boxplot além de devolver
        quais são os valores que se apresentaram como outliers
        permitindo vermos o tamanho, qtde e valores.
        """
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
        axes = axes.flatten()

        i = 0
        outliers = pd.DataFrame()
        for col in cols:
            palette = sns.color_palette("Set2", n_colors=len(df[col].unique()))
            sns.boxplot(df[col], ax=axes[i], palette=palette)
            axes[i].set_title(f'{col} Boxplot') 

            # Coleta os outliers que foram apresentados no gráfico.
            stats = boxplot_stats(df[col])
            outliers[col] = [outlier for outlier in stats[0]['fliers']] + [None] * (len(df) - len(stats[0]['fliers']))

            i += 1

        plt.tight_layout()
        plt.show()

        return outliers
    
    def plotar_dist(df: pd.DataFrame, nrow=2, ncols=5) -> None:
        """Função que plota gráficos de distribuição
        em uma única célula
        """
        fig, axes = plt.subplots(nrows=nrow, ncols=ncols, figsize=(18, 10))
        axes = axes.flatten()

        for i, column in enumerate(df.columns):
            df[column].hist(ax=axes[i],
                            edgecolor='white',
                            color='#3366FF'
                        )
            
            axes[i].set_title(f'{column}') 
            axes[i].set_xlabel('') 
            axes[i].set_ylabel('Frequency') 

            axes[i].ticklabel_format(style='plain', axis='y')
            
        plt.tight_layout()
        plt.show()

    def corrplot(df: pd.DataFrame, figsize=(12,8)) -> None:
        """Plota a correlação de Spearman das variáveis de um DataFrame
        """
        correlation_matrix = df.corr(method='spearman')

        plt.figure(figsize=figsize)
        # Criei uns filtros adicionais para embelezar
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlação de Spearman')
        plt.show()


    @staticmethod
    def plotar_nested_dist_cat(df: pd.DataFrame, cols: list) -> None:
        """Função que plota gráfico de distribuição nested
        """
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 8))
        axes = axes.flatten()

        i = 0
        for col in cols:
            palette = sns.color_palette("Set2", n_colors=len(df[col].unique()))
            sns.histplot(x=df['target'], hue=df[col],
                          ax=axes[i], kde=False,element='poly',palette=palette)
            axes[i].set_title(f'target cat por {col}') 
            i += 1

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plotar_nested_dist(df: pd.DataFrame, cols: list, nrow, ncol) -> None:
        """Função que plota gráfico de distribuição nested
        """
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(18, 8))
        axes = axes.flatten()

        i = 0
        for col in cols:
            sns.histplot(x=df[col], hue=df['target'],
                            ax=axes[i], kde=True,element='poly')
            axes[i].set_title(f'{col} pela target') 
            i += 1

        plt.tight_layout()
        plt.show()

    def plot_ecdf_features(self,df: pd.DataFrame, hue: str, figsize):
        plt.figure(figsize=figsize)

        num_features = df.shape[1]
        num_cols = int(np.ceil(np.sqrt(num_features)))
        num_rows = int(np.ceil(num_features / num_cols))

        norm = Normalize(vmin=0, vmax=1)

        for i, feature in enumerate(df.columns, 1):
            plt.subplot(num_rows, num_cols, i)

            ks_stat = 0

            if df[hue].nunique() == 2:
                group1 = df[df[hue] == 0][feature].dropna()
                group2 = df[df[hue] == 1][feature].dropna()
                ks_stat, _ = ks_2samp(group1, group2)

            sns.ecdfplot(data=df, x=feature, hue=hue, palette={0: "red", 1: "blue"})
            plt.title(f"{feature}")

            circle_color = plt.cm.Reds(norm(ks_stat))
            circle = Circle((0.9, 0.1), 0.03, color=circle_color, transform=plt.gca().transAxes, clip_on=False)
            plt.gca().add_patch(circle)

            plt.text(0.9, 0.1, f"{ks_stat:.2f}", ha="center", va="center", fontsize=10, transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.show()

    def plot_ecdf_splits(
                        self,
                        X_first: pd.DataFrame,
                        X_second: pd.DataFrame,
                        y_first: List[int],
                        y_second: List[int],
                        interest: int,
                        figsize_to_plot: Tuple[int,int]):

        """

        Plota a ECDF dos splits relacionado ao seu interesse [0,1]

        """
        # Targets

        X_first['target'] = y_first
        X_first['label'] = 1
        X_second['target'] = y_second
        X_second['label'] = 0

        # DataFrame final de concatenação

        df_plot = pd.concat([X_first, X_second],axis=0).reset_index(drop=True)
        df_plot = df_plot.query(f"target == {interest}")

        print('######################')
        print('Label - 1: Significa o X_first')
        print('Label - 0: Significa o X_second')
        print('Threshold: Considerar acima de 0.33 como atenção.')
        print('######################')

        self.plot_ecdf_features(df=df_plot,
                        hue='label',
                        figsize=figsize_to_plot)
        
