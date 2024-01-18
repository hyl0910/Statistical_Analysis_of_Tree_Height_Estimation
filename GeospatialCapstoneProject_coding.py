import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from scipy.stats import bartlett
import seaborn as sns
from termcolor import colored
from tabulate import tabulate


file=r"C:/NRE_5585/Data/Height_Validation_Data.xls"
with pd.ExcelFile(file) as xls:
    df = pd.read_excel(xls, 'Height_Validation_Data')

residuals = df['LiDAR_Ht'] - df['Field_Ht']

# Function to perform Bartlett test for equal variance
def bartlett_test(residuals, species_col):
    groups = [residuals[df['Spp'] == species] for species in df[species_col].unique() if len(df[df[species_col] == species]) >= 24]
    return stats.bartlett(*groups)

fig, axes = plt.subplots(1, 2, figsize=(7, 5))
#
# Scatterplot
axes[0].scatter(range(len(residuals)), residuals)
axes[0].set_ylabel("Residuals (m)")
axes[0].set_title("Plot of Residuals")

# Perform Bartlett test for equal variance for each species with at least 24 members
bartlett_result = bartlett_test(residuals, 'Spp')

# Display Bartlett test result on the plot
axes[0].text(0.1, max(residuals) - 1, f'Bartlett p-value: {bartlett_result.pvalue:.4f}', fontsize=8, color='red')


#
# # Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_xlabel("Quantiles")
axes[1].set_ylabel("Ordered Values")
axes[1].set_title("Probability Plot")
#

# Linear regression
X = df['LiDAR_Ht'].values.reshape(-1, 1)
y = df['Field_Ht']
regression_model = stats.linregress(X.flatten(), y)
y_pred = regression_model.slope * X.flatten() + regression_model.intercept
r_squared = regression_model.rvalue ** 2

# Display R-squared on the scatterplot
axes[1].text(-1, -13, f'R-squared: {r_squared:.4f}', fontsize=8, color='red')


# Perform normality tests
shapiro_test_result = stats.shapiro(residuals)
ks_test_result = stats.kstest(residuals, 'norm')
anderson_test_result = stats.anderson(residuals)

# Display normality test results on the Q-Q plot
axes[1].text(-1, -10, f'Shapiro: {shapiro_test_result.pvalue:.1f}', fontsize=8, color="red")
axes[1].text(-1, -11, f'KS p-value: {ks_test_result.pvalue:.1f}', fontsize=8,color="red")
axes[1].text(-1, -12, f'AD statistic: {anderson_test_result.statistic:.1f}', fontsize=8,color="red")
#axes[1].text(-5, -7, f'Anderson-Darling critical values: {anderson_test_result.critical_values}', fontsize=8)


plt.tight_layout()
plt.show()

X = df['LiDAR_Ht'].values.reshape(-1, 1)
y = df['Field_Ht']
#
# # Create a linear regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Make predictions using the regression model
y_pred = regression_model.predict(X)

# Calculate the R-squared and p-value
slope, intercept, r_value, p_value, std_err = stats.linregress(df['LiDAR_Ht'], df['Field_Ht'])
r_squared = r_value ** 2

# Create the scatter plot
plt.scatter(X, y, label='Actual Data', color='b')
plt.plot(X, y_pred, label='Regression Line', color='r')
plt.xlabel('LiDAR Height (m)')
plt.ylabel('Field Height (m)')
plt.title('LiDAR vs. Field Heights')

# Display the regression equation, R-squared, and p-value
equation = f'y = {slope:.3f}x + {intercept:.3f}'
r_squared_text = f'r-sqr: {r_squared:.3f}'
p_value_text = f'p-value: {p_value:.1f}'
#
plt.text(X.min(), y.max(), equation, fontsize=8)
plt.text(X.min(), y.max() - 1, r_squared_text, fontsize=8)
plt.text(X.min(), y.max() - 2, p_value_text, fontsize=8)
plt.show()

species_counts = df['Spp'].value_counts()
species_to_include = species_counts[species_counts >= 30].index
filtered_df = df[df['Spp'].isin(species_to_include)]

# Create a grouped boxplot for the selected species
plt.figure(figsize=(9,10))  # Set the overall figure size

# Subplot 1: Grouped Boxplot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, and this is the first plot
sns.boxplot(x='Spp', y=residuals, data=filtered_df, notch=True)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)
anova_result_species = stats.f_oneway(*[residuals[df['Spp'] == species] for species in df['Spp'].unique()])
# Annotate the boxplot with ANOVA p-value
plt.text(0.5, 0.30, f'p-value={anova_result_species.pvalue:.3f}', ha='center', va='center', transform=plt.gca().transAxes, color='red')


# Set the title and labels
plt.title("Boxplot by Species")
plt.ylabel("Residuals")

# Subplot 2: Boxplots for SppType
plt.subplot(2, 1, 2)  # 2 rows, 1 column, and this is the second plot

for species_type in df['SppType'].unique():
    species_type_residuals = residuals[df['SppType'] == species_type]

    # Check if the data is normal or not
    if stats.shapiro(species_type_residuals)[1] < 0.05:
        # Use the non-parametric test (Wilcoxon rank-sum test) if data is not normal
        test_result = stats.ranksums(species_type_residuals, species_type_residuals)

    else:
        # Use ANOVA test if data is normal
        test_result = stats.f_oneway(species_type_residuals, species_type_residuals)

    # Create boxplots with test results
    sns.boxplot(x=df['SppType'], y=species_type_residuals,width=0.5,notch=True)
    plt.text(0.5, 0.95, f'p-value={test_result[1]:.3f}', ha='center', va='center',
             transform=plt.gca().transAxes, color='red')

    plt.title(f"Boxplots for species type")

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()

species_to_test = filtered_df['Spp'].unique()
pairwise_test_results = pd.DataFrame(index=species_to_test, columns=species_to_test, dtype=float)

# Initialize a set to keep track of filled cells
filled_cells = set()

for i in range(len(species_to_test)):
    for j in range(i + 1, len(species_to_test)):
        # Check if the cell has already been filled
        if (species_to_test[i], species_to_test[j]) in filled_cells or (species_to_test[j], species_to_test[i]) in filled_cells:
            continue

        species_i_residuals = residuals[df['Spp'] == species_to_test[i]]
        species_j_residuals = residuals[df['Spp'] == species_to_test[j]]

        # Check if the data is normal or not
        is_i_residuals_normal = stats.shapiro(species_i_residuals)[1] < 0.05
        is_j_residuals_normal = stats.shapiro(species_j_residuals)[1] < 0.05

        if is_i_residuals_normal or is_j_residuals_normal:
            # Use the non-parametric test (Wilcoxon rank-sum test) if data is not normal
            test_statistic, p_value = stats.ranksums(species_i_residuals, species_j_residuals)
        else:
            # Use 2-sample t-test if data is normal
            test_statistic, p_value = stats.ttest_ind(species_i_residuals, species_j_residuals)

        # Mark the cells as filled
        filled_cells.add((species_to_test[i], species_to_test[j]))
        filled_cells.add((species_to_test[j], species_to_test[i]))

        pairwise_test_results.at[species_to_test[i], species_to_test[j]] = p_value
        pairwise_test_results.at[species_to_test[j], species_to_test[i]] = p_value


pairwise_test_results_styled = pairwise_test_results.applymap(lambda val: 1 if pd.isna(val) else val)
# Define the significant color formatting function
significant_color = lambda val: f"\x1b[31m{val:.3f}\x1b[0m" if float(val) < 0.05 else f"{val:.3f}"

# Apply the color formatting to the entire DataFrame
pairwise_test_results_styled = pairwise_test_results_styled.applymap(significant_color)

# Create a new DataFrame with 1 instead of NaN and apply formatting
formatted_results = pd.DataFrame(index=pairwise_test_results_styled.index, columns=pairwise_test_results_styled.columns, dtype=str)
for col in formatted_results.columns:
    for idx in formatted_results.index:
        if col == idx:
            formatted_results.at[idx, col] = '1'
        else:
            formatted_results.at[idx, col] = pairwise_test_results_styled.at[idx, col]

# Convert the DataFrame to a tabular format
tabular_results = tabulate(formatted_results, headers='keys', tablefmt='pretty')

# Print the pairwise test results
print("Pairwise 2-sample tests by species")
print(tabular_results)












# # Highlight significant results in red
# significant_color = lambda val: f"\x1b[31m{val:.3f}\x1b[0m" if float(val) < 0.05 else f"{val:.3f}"
#
# # Apply the color formatting to the entire DataFrame
# pairwise_test_results_styled = pairwise_test_results.applymap(significant_color)
#
# # Define the custom function to apply formatting
# def format_cell(val, row, col):
#     if pd.notna(val):
#         return f"\x1b[31m{val:.3f}\x1b[0m" if float(val) < 0.05 else f"{val:.3f}"
#     else:
#         return val
#
# # Apply the formatting function to the entire DataFrame
# pairwise_test_results_styled = pairwise_test_results.style.format(format_cell)
#
# # Convert the DataFrame to a formatted table using tabulate
# formatted_table = tabulate(pairwise_test_results_styled, tablefmt="grid", headers="keys", showindex=True, numalign="center")
#
# # Print the formatted table
# print("Pairwise 2-sample tests by species")
# print(formatted_table)





# # Highlight significant results in red
# significant_color = lambda val: f"\x1b[31m{val:.3f}\x1b[0m" if float(val) < 0.05 else f"{val:.3f}"
#
# pairwise_test_results_styled = pairwise_test_results.applymap(lambda val: 1 if pd.isna(val) else val)
#
# # Highlight significant results in red
# sisignificant_color = lambda val: f"\x1b[31m{val:.3f}\x1b[0m" if float(val) < 0.05 else f"{val:.3f}"
#
# # Apply the color formatting to the entire DataFrame
# pairwise_test_results_styled = pairwise_test_results_styled.applymap(significant_color)
#
# # Convert the DataFrame to a tabular format
# tabular_results = tabulate(pairwise_test_results_styled, headers='keys', tablefmt='pretty')
#
# # Print the pairwise test results
# print("Pairwise 2-sample tests by species")
# print(tabular_results)
#
#


