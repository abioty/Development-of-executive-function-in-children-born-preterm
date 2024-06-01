import os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr



# Set seed for reproducibility
np.random.seed(123)

# Load data function
def load_data(file_name):
    return pd.read_csv(file_name)

# Canonical Correlation Analysis function
def calculate_canonical_loadings(X, Y):
    cca = CCA(n_components=35, max_iter=1000)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    return np.corrcoef(X_c.T, Y_c.T)[:X_c.shape[1], X_c.shape[1]:], cca

# Permutation test function for CCA
def permutation_test_cca(X, Y, observed_corr, n_iterations=50):
    p_values = np.zeros(observed_corr.shape[0])
    permuted_corr = np.zeros((n_iterations, observed_corr.shape[0]))
    cca = CCA(n_components=35, max_iter=1000)
    for i in range(n_iterations):
        shuffled_X = np.random.permutation(X)
        shuffled_Y = np.random.permutation(Y)
        cca.fit(shuffled_X, shuffled_Y)
        X_c, Y_c = cca.transform(shuffled_X, shuffled_Y)
        permuted_corr[i] = np.diag(np.corrcoef(X_c.T, Y_c.T)[:X_c.shape[1], X_c.shape[1]:])
    p_values = 1 - np.array([percentileofscore(permuted_corr[:, i], observed_corr[i]) / 100 for i in range(observed_corr.shape[0])])
    return p_values

# Function to get top variable names for significant components
def get_top_variable_names_for_significant_components(data, weights, significant_indices, threshold=0.5):
    top_variable_names = {}
    column_names = data.columns
    for index in significant_indices:
        weight_vector = weights[:, index]
        top_variable_indices = np.where(np.abs(weight_vector) >= threshold)[0]
        top_variable_names[index] = column_names[top_variable_indices].tolist()
    return top_variable_names

# Main execution
if __name__ == "__main__":
    desktop_path = os.path.expanduser("~/Desktop/fc-sc-data/NMF-Y")
    os.chdir(desktop_path)
    
    # Load FC and SC data
    fc_data = load_data("FC_top_vars.csv")
    sc_data = load_data("SC_top_vars.csv")
    
    # Standardize data
    scaler = StandardScaler()
    fc_data_standardized = scaler.fit_transform(fc_data)
    sc_data_standardized = scaler.fit_transform(sc_data)
    
    # Calculate canonical loadings
    observed_corr, cca = calculate_canonical_loadings(fc_data_standardized, sc_data_standardized)
    
    # Perform permutation test
    p_values = permutation_test_cca(fc_data_standardized, sc_data_standardized, np.diag(observed_corr))
    
    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    significant_indices = np.where(pvals_corrected < 0.05)[0]

    if len(significant_indices) > 0:
        print(f"Significant components at indices: {significant_indices}")

        # Calculate R^2 for the significant components
        r_squared_values = np.square(np.diag(observed_corr)[significant_indices])
        print(f"Coefficient of Determination (R^2) for significant components: {r_squared_values}")

        # Print only the p-values corresponding to the significant components
        print(f"Permutation test p-values for significant components: {p_values[significant_indices]}")

        # Get top variable names for FC and SC
        top_fc_variable_names = get_top_variable_names_for_significant_components(fc_data, cca.x_weights_, significant_indices, threshold=0.2)
        top_sc_variable_names = get_top_variable_names_for_significant_components(sc_data, cca.y_weights_, significant_indices, threshold=0.2)
        print(f"Top variable names for significant components in FC: {top_fc_variable_names}")
        print(f"Top variable names for significant components in SC: {top_sc_variable_names}")

        # Initialize dictionaries to store coefficients for FC and SC
        coef_fc_per_var = {}
        coef_sc_per_var = {}

        # Process significant components
        for component_index in significant_indices:
            top_fc_var_names = top_fc_variable_names[component_index]
            top_sc_var_names = top_sc_variable_names[component_index]

            top_fc_var_indices = [fc_data.columns.get_loc(var) for var in top_fc_var_names]
            top_sc_var_indices = [sc_data.columns.get_loc(var) for var in top_sc_var_names]

            coef_fc_per_var[component_index] = cca.x_weights_[top_fc_var_indices, component_index]
            coef_sc_per_var[component_index] = cca.y_weights_[top_sc_var_indices, component_index]

        print(f"Canonical coefficients for each top variable in FC for significant components: {coef_fc_per_var}")
        print(f"Canonical coefficients for each top variable in SC for significant components: {coef_sc_per_var}")

       # Define components of interest
        components_of_interest = [17, 26]
        
        for component_of_interest in components_of_interest:
            if component_of_interest in significant_indices:
                canonical_correlation = np.diag(observed_corr)[component_of_interest]
                fc_weights = cca.x_weights_[:, component_of_interest]
                sc_weights = cca.y_weights_[:, component_of_interest]

                fc_variable_names = top_fc_variable_names[component_of_interest]
                sc_variable_names = top_sc_variable_names[component_of_interest]

                fc_variable_weights = fc_weights[[fc_data.columns.get_loc(var) for var in fc_variable_names]]
                sc_variable_weights = sc_weights[[sc_data.columns.get_loc(var) for var in sc_variable_names]]

                fc_data_export = pd.DataFrame({
                    'component': component_of_interest,
                    'variable': fc_variable_names,
                    'type': 'FC',
                    'canonical_correlation': canonical_correlation,
                    'canonical_weight': fc_variable_weights
                })

                sc_data_export = pd.DataFrame({
                    'component': component_of_interest,
                    'variable': sc_variable_names,
                    'type': 'SC',
                    'canonical_correlation': canonical_correlation,
                    'canonical_weight': sc_variable_weights
                })

                all_data_export = pd.concat([fc_data_export, sc_data_export], ignore_index=True)

                csv_file_path = os.path.join(desktop_path, f'significant_variables_components_{component_of_interest}.csv')
                all_data_export.to_csv(csv_file_path, index=False)
                print(f"CSV file saved to: {csv_file_path}")
            else:
                print(f"Component {component_of_interest} is not among the significant components.")


#Plotting 


# Assuming fc_data_standardized and sc_data_standardized are your standardized original datasets
cca = CCA(n_components=35)
cca.fit(fc_data_standardized, sc_data_standardized)
fc_transformed, sc_transformed = cca.transform(fc_data_standardized, sc_data_standardized)

# Specified indices and corresponding permutation p-values for significant components
significant_indices = [17, 26]
perm_test_p_values = pvals_corrected[significant_indices]

# Manually formatted top variable names for each significant component
top_fc_variable_names = {
    17: 'aCG l_bc, aLTL (WM) l_bc, mSTG l_co, pM/ITG l_co, \npM/ITG r_co, Caudate r_co, pSTG r_le, Hippo l_le',
    26: 'pM/ITG(WM) l_bc, LOTG/aFuG l_bc, OL l_co, FL l_co, \npPHG(WM) l_co, OL(WM) r_co, Caudate r_co, PL(WM) r_le, Hippo l_le'
}
top_sc_variable_names = {
    17: 'PL(WM) l_asp, PL l_asp, mSTG(WM) l_asp, \nPL r_asp, pCG l_asp, FL (WM) r_co, OL(WM) l_co',
    26: 'Insula r_asp, mSTG(WM) r_asp, PL(WM) l_asp, \nmSTG(WM) l_asp, FL l_asp, aM/ITG r_asp, Cerebellum r _co'
}

# Create a 1x2 grid plot (one row, two columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjusted figure size for horizontal layout

for i, index in enumerate(significant_indices):
    # Extract the transformed components for the selected index
    x = fc_transformed[:, index]
    y = sc_transformed[:, index]

    # Calculate R-squared value
    pearson_corr, _ = pearsonr(x, y)
    r_squared_value = pearson_corr ** 2

    # Map p-values to marker sizes, lower p-value results in larger size.
    weights = 50 * np.exp(-10 * perm_test_p_values[i])
    alphas = 1 - perm_test_p_values[i]  # Lower p-value, higher transparency

    ax = axes[i]
    sc = ax.scatter(
        x, y, 
        c=y, 
        s=weights,  # Marker size based on p-values
        alpha=alphas,  # Transparency based on p-values
        cmap='viridis', 
        edgecolor='k'
    )

    sns.regplot(
        x=x, 
        y=y, 
        ax=ax, 
        scatter=False, 
        line_kws={'color': plt.cm.YlGnBu(0.7), 'lw': 3}
    )

    # Additional text from top variable names
    additional_x_text = top_fc_variable_names[index]
    additional_y_text = top_sc_variable_names[index]

    ax.set_xlabel(f'FC SCORE: {additional_x_text}', fontsize=12, labelpad=16)
    ax.set_ylabel(f'SC SCORE: {additional_y_text}', fontsize=12, labelpad=16)

    # Annotation with permutation test p-value and R-squared value
    ax.annotate(f'PermTest p-value: {perm_test_p_values[i]:.3f}\n$R^2$: {r_squared_value:.3f}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=14, 
                verticalalignment='top')

    # Label the plots as 'a' and 'b'
    ax.text(-0.1, 1.05, chr(97 + i), transform=ax.transAxes, size=20, weight='bold')

# Adjust the layout and margins to fit all contents
plt.tight_layout(pad=3)

plt.savefig("sc-fc-significant-component2.tif", dpi=300, bbox_inches='tight')
plt.show()



 