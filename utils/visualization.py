import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def boxplot_by_category(data, num_cols, target, suptitle):
    # Calculate number of rows needed
    n_rows = (len(num_cols) + 2) // 3  # Round up to nearest integer

    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6*n_rows))
    fig.suptitle(suptitle, fontsize=16)

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Create a box plot for each numerical column
    for i, col in enumerate(num_cols):
        sns.boxplot(x=target, y=col, data=data, ax=axes[i])
        axes[i].set_xlabel(target)
        axes[i].set_ylabel(col)

    # Remove excess subplots
    for i in range(len(num_cols), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def stacked_barplot_by_category(data, cat_cols, target, suptitle, top_N=10):
    # Calculate number of rows needed
    n_rows = (len(cat_cols) + 2) // 3  # Round up to nearest integer

    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 6*n_rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        # Get the top 10 categories
        top_categories = data[col].value_counts().nlargest(top_N).index.to_list()
        
        # Filter the data and reset the category levels
        filtered_data = data[data[col].isin(top_categories)].copy()
        filtered_data[col] = filtered_data[col].cat.remove_unused_categories()

        # Calculate counts
        count_data = filtered_data.groupby([col, target]).size().unstack(fill_value=0)
        # Sort the data by total count
        count_data = count_data.sort_values(by=count_data.columns.tolist(), ascending=False)

        # Plot counts
        count_data.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(f'Count')

        # Add a note about top 10 categories if applicable
        if data[col].nunique() > top_N:
            # Rotate x-axis labels
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            # axes[i].text(0.5, -0.20, f'Showing top {top_N} out of {train[col].nunique()} categories', 
            #              ha='center', va='center', transform=axes[i].transAxes, fontsize=10)
            axes[i].set_xlabel(f'{col} (Showing top {top_N} out of {data[col].nunique()})')
        else:
            # Rotate x-axis labels
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0, ha='center')

    # Remove excess subplots
    for i in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[i])

    # Add an overall title to the figure
    fig.suptitle(suptitle, fontsize=16, y=0.95)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust the top margin to accommodate the suptitle
    plt.show()
