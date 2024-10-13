import seaborn as sns
import matplotlib.pyplot as plt

class PlotGraph:
    def __init__(self, data):
        self.data = data
        self.summary_statistics(data)
        self.plot_numeric_distributions(data)
        self.plot_binary_counts(data, 'Diabetes_binary')
        self.plot_boxplots(data, 'Diabetes_binary')
        self.plot_crosstab(data, 'Diabetes_binary')
        self.plot_correlation_heatmap(data)

    def summary_statistics(self, data):
        print("Summary Statistics:")
        print(data.describe(include='all'))
    
    def plot_numeric_distributions(self, data):
        numeric_columns = data.select_dtypes(include=['int64']).columns
        for col in numeric_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(data[col], bins=30, kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid()
            plt.show()

    def plot_binary_counts(self,data, target):
            binary_columns = data.select_dtypes(include=['bool']).columns
            for col in binary_columns:
              plt.figure(figsize=(10, 5))
              sns.countplot(x=data[col], hue=data[target])
              plt.title(f'Count of {col} by {target}')
              plt.xlabel(col)
              plt.ylabel('Count')
              plt.legend(title=target)
              plt.grid()
              plt.show()

    def plot_boxplots(self,data, target):
            numeric_columns = data.select_dtypes(include=['int64']).columns
            for num_col in numeric_columns:
              plt.figure(figsize=(10, 5))
              sns.boxplot(x=data[target], y=data[num_col])
              plt.title(f'Boxplot of {num_col} by {target}')
              plt.xlabel(target)
              plt.ylabel(num_col)
              plt.grid()
              plt.show()

    def plot_crosstab(self,data, target):
            categorical_columns = data.select_dtypes(include=['category']).columns
            for cat_col in categorical_columns:
              crosstab = pd.crosstab(data[cat_col], data[target])
              print(f'Crosstab for {cat_col} vs {target}:')
              print(crosstab)
              sns.heatmap(crosstab, annot=True, fmt="d", cmap='Blues')
              plt.title(f'Crosstab Heatmap of {cat_col} by {target}')
              plt.ylabel(cat_col)
              plt.xlabel(target)
              plt.show()

    def plot_correlation_heatmap(self,data):
            numeric_data = data.select_dtypes(include=['int64'])
            correlation_matrix = numeric_data.corr()
            plt.figure(figsize=(15, 15))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title('Correlation Heatmap')
            plt.show()