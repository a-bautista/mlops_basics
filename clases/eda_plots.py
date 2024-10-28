import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from math import ceil

class Plots:
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def summary_statistics(self):
        print(self.data.describe(include='all'))
    
    def plot_numeric_distributions(self):
        numeric_columns = self.data.select_dtypes(include=['int64']).columns
        fig, axes = plt.subplots(ceil(len(numeric_columns) / 3), 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(numeric_columns):
            sns.histplot(self.data[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    def plot_binary_counts(self):
        binary_columns = self.data.select_dtypes(include=['bool']).columns
        fig, axes = plt.subplots(ceil(len(binary_columns) / 3), 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, col in enumerate(binary_columns):
            sns.countplot(x=self.data[col], hue=self.data[self.target], ax=axes[i])
            axes[i].set_title(f'Count of {col} by {self.target}')
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self):
        numeric_columns = self.data.select_dtypes(include=['int64']).columns
        fig, axes = plt.subplots(ceil(len(numeric_columns) / 3), 3, figsize=(15, 10))
        axes = axes.flatten()
        for i, num_col in enumerate(numeric_columns):
            sns.boxplot(x=self.data[self.target], y=self.data[num_col], ax=axes[i])
        plt.tight_layout()
        plt.show()

    def plot_crosstab(self):
        categorical_columns = self.data.select_dtypes(include=['category']).columns
        for cat_col in categorical_columns:
            crosstab = pd.crosstab(self.data[cat_col], self.data[self.target])
            sns.heatmap(crosstab, annot=True, cmap='Blues')
            plt.show()

    def plot_correlation_heatmap(self):
        numeric_data = self.data.select_dtypes(include=['int64'])
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()