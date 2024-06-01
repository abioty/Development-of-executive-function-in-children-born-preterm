import os
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns

# Define the working directory
os.chdir(os.path.expanduser("~/Desktop/fc-sc-data/NMF-Y"))

# Load the CSV data
data = pd.read_csv("le-fc.csv")

# Define the range of components to consider
n_components_range = range(2, 31)  # Adjust this range as needed

# Initialize variables
relative_errors = []
threshold_relative_error = 0.02  # 5% relative error threshold
optimal_n_components = None  # Placeholder for the optimal number of components

# Calculate the norm of the original data for normalization
original_norm = np.linalg.norm(data.values)

# Determine the optimal number of components based on relative error threshold
for n_components in n_components_range:
    nmf_model = NMF(n_components=n_components, init='nndsvd', random_state=123, max_iter=100000, tol=1e-4)
    W = nmf_model.fit_transform(data)
    H = nmf_model.components_
    reconstruction_error = np.linalg.norm(data - np.dot(W, H), 'fro')
    relative_error = reconstruction_error / original_norm
    relative_errors.append(relative_error)
    if relative_error < threshold_relative_error:
        optimal_n_components = n_components
        break

# Check if an optimal number of components was found
if optimal_n_components is None:
    print("Did not achieve desired relative error with given range of components.")
else:
    print(f"Optimal number of components based on relative error threshold: {optimal_n_components}")

    # Proceed with the optimal number of components
    nmf_model = NMF(n_components=optimal_n_components, init='nndsvd', random_state=123, max_iter=100000, tol=1e-4)
    W = nmf_model.fit_transform(data)
    H = nmf_model.components_

    # Build DataFrame of variables contributing at least 1% for each component
    top_variables_list = []
    min_contribution_threshold = 0.01  # 1% contribution threshold

    for component_idx, component_scores in enumerate(H):
        total_score = np.sum(component_scores)
        for variable_idx, score in enumerate(component_scores):
            if score / total_score >= min_contribution_threshold:
                top_variables_list.append({'Component': component_idx + 1,
                                           'Variable': data.columns[variable_idx],
                                           'Score': score})

    top_variables_df = pd.DataFrame(top_variables_list)

    # Sort the DataFrame by Component and Score in descending order
    top_variables_df = top_variables_df.sort_values(by=['Component', 'Score'], ascending=[True, False])

    # Export top contributing variables to a CSV file
    top_variables_df.to_csv('top_variables_scores_le-fc.csv', index=False)
    print("Top contributing variables exported to top_variables_scores_le-fc.csv")

    # Create FacetGrid for plotting with 4 components per row and increased height
    g = sns.FacetGrid(top_variables_df, col='Component', col_wrap=4, height=5, sharex=False, sharey=False)
    g.map_dataframe(lambda data, **kwargs: sns.barplot(x='Score', y='Variable', data=data, palette='viridis', orient='h', **kwargs))
    g.set_titles('Network Component {col_name}')
    g.set_xlabels('Score')
    g.set_ylabels('Top Contributing Variables')

    # Annotate the scores for each component
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.2f}", (p.get_width(), p.get_y() + p.get_height() / 2),
                        xytext=(5, 0), textcoords='offset points', ha='left', va='center')

    # Adjust y-axis labels for all columns
    for ax in g.axes.flat:
        ax.set_yticklabels(ax.get_yticklabels(), ha='right')

    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.tight_layout()
    plt.savefig('components_plot_le-fc.png', dpi=300)
    plt.show()

    # Feature extraction: Create a DataFrame using the W matrix from the NMF model
    features_df = pd.DataFrame(W, columns=[f'Comp_fc_le{i + 1}' for i in range(optimal_n_components)])
    features_df.to_csv('features_le_fc.csv', index=False)
    print("Features exported to features_le_fc.csv")




 