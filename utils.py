import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
import seaborn as sns;sns.set()
import warnings
warnings.filterwarnings('ignore')
#import plotly.express as px
from datetime import datetime,time

import math


def central_tendency_stats(data, column):
    """Calculate and display all measures of central tendency for a column."""
    values = data[column]

    # Calculate different measures
    mean_val = values.mean()
    median_val = values.median()
    mode_val = values.mode()[0]  # Mode can have multiple values, take the first one

    # Weighted mean example (using price as weights for demonstration)
    # Compute weighted average of each column values.
    # For all columns except, price, use values from price as weights. FOr price, use jordan_min_price as weights (to avoid self-weighting)
    if column != 'price':
        weights = data['price']
        weighted_mean = np.average(values, weights=weights)
    else:
        weighted_mean = np.average(values, weights=data['jordan_min_price'])

    # Display results
    print(f"Statistics for {column}:")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Mode: {mode_val:.4f}")
    print(f"Weighted Mean: {weighted_mean:.4f}")

    # Visual representation of mean, median and mode
    plt.figure(figsize=(20, 5))
    sns.histplot(values, kde=True)
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='g', linestyle='-.', label=f'Median: {median_val:.2f}')
    plt.axvline(mode_val, color='b', linestyle=':', label=f'Mode: {mode_val:.2f}')
    plt.title(f'Distribution of {column} with Central Tendency Measures')
    plt.legend()
    plt.show()

    # Explanation
    print("\nInterpretation:")
    if abs(mean_val - median_val) < 0.01 * abs(mean_val):
        print("The mean and median are very close, suggesting a relatively symmetric distribution.")
    elif mean_val > median_val:
        print("The mean is greater than the median, suggesting a right-skewed (positively skewed) distribution.")
        print("This means there are some larger values pulling the mean upward.")
    else:
        print("The median is greater than the mean, suggesting a left-skewed (negatively skewed) distribution.")
        print("This means there are some smaller values pulling the mean downward.")

    if abs(mode_val - median_val) < 0.01 * abs(median_val):
        print("The mode is close to the median, which is typical in many distributions.")
    print("\n")

    print("==" * 50)


def dispersion_stats(data, column):
    """Calculate and display all measures of dispersion for a column."""
    values = data[column]

    # Calculate different measures
    range_val = values.max() - values.min()
    variance = values.var()
    std_dev = values.std()

    # Calculate quartiles and IQR
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    # Calculate coefficient of variation (CV)
    cv = (std_dev / values.mean()) * 100

    # Display results
    print(f"Dispersion Statistics for {column}:")
    print(f"Range: {range_val:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Interquartile Range (IQR): {iqr:.4f}")
    print(f"Coefficient of Variation: {cv:.2f}%")

    # Visual representation of dispersion
    plt.figure(figsize=(20, 5))

    # Create two subplots
    plt.subplot(1, 2, 1)
    sns.boxplot(y=values)
    plt.title(f'Boxplot of {column}')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    sns.histplot(values, kde=True)
    plt.axvline(values.mean(), color='r', linestyle='--',
                label=f'Mean: {values.mean():.2f}')
    plt.axvline(values.mean() + std_dev, color='g', linestyle='-.',
               label=f'Mean + SD: {values.mean() + std_dev:.2f}')
    plt.axvline(values.mean() - std_dev, color='g', linestyle='-.',
               label=f'Mean - SD: {values.mean() - std_dev:.2f}')
    plt.title(f'Distribution of {column} with Standard Deviation')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Explanation
    print("\nInterpretation:")
    print(f"Range: The difference between the maximum and minimum values is {range_val:.4f}.")
    print(f"Standard Deviation: On average, values deviate from the mean by approximately {std_dev:.4f}.")
    print(f"IQR: The middle 50% of the data falls within a range of {iqr:.4f}.")

    # Interpretation of Coefficient of Variation
    if cv < 10:
        print("Coefficient of Variation: The data shows low variability relative to the mean.")
    elif cv < 30:
        print("Coefficient of Variation: The data shows moderate variability relative to the mean.")
    else:
        print("Coefficient of Variation: The data shows high variability relative to the mean.")
    print("\n")

    print("==" * 50)


def compare_means_by_caregory_col(data, features, category_col):
    """Compare means of features between caregory_col."""

    grouped_means = data.groupby(category_col)[features].mean()

    # Plot
    plt.figure(figsize=(20, 4))
    grouped_means.T.plot(kind='bar', rot=45,title=f'Mean of {features} by {category_col}')
    plt.title(f'Comparison of Feature {features} Means by {category_col}')
    plt.ylabel(f'{features} Mean Value')
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Return the grouped means for further analysis
    return np.round(grouped_means,2)


######################################
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.stats.correlation_tools import corr_nearest, corr_clipped
import warnings
from itertools import combinations
import requests
from io import StringIO

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import math
######################################



    
def explore_correlation_data(data, continuous_vars):
    """
    Exploration of the dataset focusing on relationships between continuous variables.

    """
    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS FOR CORRELATION")
    print("=" * 70)

    # Basic descriptive statistics
    print("\nDescriptive Statistics for Continuous Variables:")
    print("-" * 50)

    desc_stats = data[continuous_vars].describe().round(2)
    print(desc_stats)

    # Calculate coefficient of variation for each variable
    # Remember that CV helps to understand the extent of dispersion or variability around the mean of a dataset.
    print(f"\nCoefficient of Variation (CV = std/mean):")
    print("-" * 40)
    for var in continuous_vars:
        mean_val = data[var].mean()
        std_val = data[var].std()
        cv = (std_val / mean_val) * 100
        print(f"{var:<20}: {cv:>6.1f}%")

    # Check distributions for normality (important for Pearson correlation)
    print(f"\nNormality Assessment (Shapiro-Wilk Test):")
    print("-" * 45)
    normality_results = {}

    for var in continuous_vars:
        # Use sample if data is too large for Shapiro-Wilk
        sample_data = data[var].dropna()
        if len(sample_data) > 5000:
            sample_data = sample_data.sample(5000, random_state=42)

        stat, p_value = stats.shapiro(sample_data)
        normality_results[var] = {'statistic': stat, 'p_value': p_value}

        interpretation = "Normal" if p_value > 0.05 else "Non-normal"
        print(f"{var:<20}: W = {stat:.4f}, p = {p_value:.4f} ({interpretation})")

    # Create different visualizations
    n_vars = len(continuous_vars)
    fig, axes = plt.subplots(3, n_vars, figsize=(4*n_vars, 12))
    fig.suptitle('Exploratory Data Analysis: Continuous Variables', fontsize=16)

    for i, var in enumerate(continuous_vars):
        var_data = data[var].dropna()

        # Row 1: Histograms with normal overlay
        axes[0, i].hist(var_data, bins=30, density=True, alpha=0.7,
                       color='skyblue', edgecolor='black')

        # Overlay normal distribution
        mu, sigma = stats.norm.fit(var_data)
        x = np.linspace(var_data.min(), var_data.max(), 100)
        axes[0, i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-',
                       linewidth=2, label='Normal fit')

        axes[0, i].set_title(f'Distribution: {var}')
        axes[0, i].set_xlabel(var)
        axes[0, i].set_ylabel('Density')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Row 2: Q-Q plots for normality
        stats.probplot(var_data, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'Q-Q Plot: {var}')
        axes[1, i].grid(True, alpha=0.3)

        # Row 3: Box plots
        axes[2, i].boxplot(var_data, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[2, i].set_title(f'Box Plot: {var}')
        axes[2, i].set_ylabel(var)
        axes[2, i].grid(True, alpha=0.3)

        # Add outlier information
        Q1 = var_data.quantile(0.25)
        Q3 = var_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = var_data[(var_data < lower_bound) | (var_data > upper_bound)]

        axes[2, i].text(0.5, 0.95, f'Outliers: {len(outliers)}',
                       transform=axes[2, i].transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Identify potential outliers
    print(f"\nOutlier Analysis - quantile-based methods: ")
    print("-" * 20)

    outlier_summary = {}
    for var in continuous_vars:
        var_data = data[var].dropna()
        Q1 = var_data.quantile(0.25)
        Q3 = var_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = var_data[(var_data < lower_bound) | (var_data > upper_bound)]
        outlier_percentage = (len(outliers) / len(var_data)) * 100

        outlier_summary[var] = {
            'count': len(outliers),
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

        print(f"{var:<20}: {len(outliers):>3} outliers ({outlier_percentage:>5.1f}%)")

    # Species-specific analysis
    if 'p_color' in data.columns:
        print(f"\nSpecies-specific Means:")
        print("-" * 25)
        species_means = data.groupby('p_color')[continuous_vars].mean().round(2)
        print(species_means)

        # Visualize species differences
        
        n_cols = 2
        n_rows = (len(continuous_vars) + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

        fig.suptitle('Variable Distributions by p_color', fontsize=16)

        for i, var in enumerate(continuous_vars):
            row = i // n_cols
            col = i % n_cols    

            sns.boxplot(data=data, x='p_color', y=var, ax=axes[row, col])
            axes[row, col].set_title(f'{var} by p_color')
            axes[row, col].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    return {
        'descriptive_stats': desc_stats,
        'normality_results': normality_results,
        'outlier_summary': outlier_summary,
        'species_means': species_means if 'species' in data.columns else None
    }
def calculate_basic_correlations(data, continuous_vars):
    """
    Calculate and interpret basic Pearson correlations between continuous variables.

    """
    print("\n" + "=" * 70)
    print("BASIC PEARSON CORRELATIONS")
    print("=" * 70)

    # Calculate correlation matrix
    corr_matrix = data[continuous_vars].corr()

    print("Correlation Matrix:")
    print("-" * 20)
    print(corr_matrix.round(3))

    # Extract unique pairs and their correlations
    print(f"\nPairwise Correlations:")
    print("-" * 25)

    correlation_results = []

    for i in range(len(continuous_vars)):
        for j in range(i+1, len(continuous_vars)):
            var1 = continuous_vars[i]
            var2 = continuous_vars[j]

            # FIXED: Calculate correlation with proper alignment
            # Get valid pairs first to ensure alignment
            valid_pairs = data[[var1, var2]].dropna()

            # Calculate correlation and significance test on aligned data
            r, p_value = pearsonr(valid_pairs[var1], valid_pairs[var2])

            # Calculate sample size
            n = len(valid_pairs)

            # Interpret strength
            if abs(r) >= 0.7:
                strength = "Strong"
            elif abs(r) >= 0.3:
                strength = "Moderate"
            else:
                strength = "Weak"

            # Interpret direction
            direction = "Positive" if r > 0 else "Negative"

            # Significance
            significance = "Significant" if p_value < 0.05 else "Not significant"

            correlation_results.append({
                'var1': var1,
                'var2': var2,
                'correlation': r,
                'p_value': p_value,
                'n': n,
                'strength': strength,
                'direction': direction,
                'significance': significance
            })

            print(f"{var1} ↔ {var2}:")
            print(f"  r = {r:>7.3f}, p = {p_value:>7.4f}, n = {n:>3}")
            print(f"  {strength} {direction.lower()} correlation ({significance.lower()})")
            print()

    # Create correlation heatmap
    plt.figure(figsize=(30, 8))

    # Create mask for upper triangle to show only lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Pearson Correlation Coefficient'})

    plt.title('Correlation Matrix Heatmap', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"Correlation Summary:")
    print("-" * 20)

    correlations_flat = [abs(result['correlation']) for result in correlation_results]
    print(f"Number of correlations: {len(correlations_flat)}")
    print(f"Mean |correlation|: {np.mean(correlations_flat):.3f}")
    print(f"Max |correlation|: {np.max(correlations_flat):.3f}")
    print(f"Min |correlation|: {np.min(correlations_flat):.3f}")

    # Count by strength
    strength_counts = {}
    for result in correlation_results:
        strength = result['strength']
        strength_counts[strength] = strength_counts.get(strength, 0) + 1

    print(f"\nCorrelations by strength:")
    for strength, count in strength_counts.items():
        percentage = (count / len(correlation_results)) * 100
        print(f"  {strength}: {count} ({percentage:.1f}%)")

    # Significant correlations
    significant_corrs = [r for r in correlation_results if r['significance'] == 'Significant']
    print(f"\nSignificant correlations: {len(significant_corrs)} out of {len(correlation_results)}")

    return {
        'correlation_matrix': corr_matrix,
        'pairwise_results': correlation_results,
        'summary_stats': {
            'mean_abs_correlation': np.mean(correlations_flat),
            'max_abs_correlation': np.max(correlations_flat),
            'min_abs_correlation': np.min(correlations_flat),
            'strength_distribution': strength_counts,
            'significant_count': len(significant_corrs)
        }
    }

def check_correlation_assumptions(data, var1, var2, create_plots=True):
    """
    Assumption checking for Pearson correlation between two variables.
    """
    print(f"\n" + "=" * 60)
    print(f"ASSUMPTION CHECKING: {var1} vs {var2}")
    print("=" * 60)

    # Get clean data for the pair
    clean_data = data[[var1, var2]].dropna()
    x = clean_data[var1]
    y = clean_data[var2]
    n = len(clean_data)

    print(f"Sample size: {n} complete pairs")

    # 1. Linear relationship check
    print(f"\n1. LINEAR RELATIONSHIP")
    print("-" * 25)

    # Calculate Pearson correlation
    r_pearson, p_pearson = pearsonr(x, y)

    # Fit polynomial models to test for non-linearity
    #Note: Remember that R-squared indicates how well the model fits the data
    # Linear model (degree 1)
    linear_coef = np.polyfit(x, y, 1)
    linear_pred = np.polyval(linear_coef, x)
    linear_r2 = 1 - (np.sum((y - linear_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # Quadratic model (degree 2)
    quad_coef = np.polyfit(x, y, 2)
    quad_pred = np.polyval(quad_coef, x)
    quad_r2 = 1 - (np.sum((y - quad_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    # Test for non-linearity
    r2_improvement = quad_r2 - linear_r2
    linearity_ok = r2_improvement < 0.05  # Arbitrary threshold

    print(f"Linear R² = {linear_r2:.4f}")
    print(f"Quadratic R² = {quad_r2:.4f}")
    print(f"R² improvement = {r2_improvement:.4f}")
    print(f"Linearity assumption: {'SATISFIED' if linearity_ok else 'QUESTIONABLE'}")

    # 2. Normality check
    print(f"\n2. NORMALITY")
    print("-" * 15)

    # Shapiro-Wilk tests (use sample if too large)
    x_sample = x.sample(min(5000, len(x)), random_state=42) if len(x) > 5000 else x
    y_sample = y.sample(min(5000, len(y)), random_state=42) if len(y) > 5000 else y

    shapiro_x = stats.shapiro(x_sample)
    shapiro_y = stats.shapiro(y_sample)

    normality_x = shapiro_x.pvalue > 0.05
    normality_y = shapiro_y.pvalue > 0.05

    print(f"{var1}: W = {shapiro_x.statistic:.4f}, p = {shapiro_x.pvalue:.4f} ({'Normal' if normality_x else 'Non-normal'})")
    print(f"{var2}: W = {shapiro_y.statistic:.4f}, p = {shapiro_y.pvalue:.4f} ({'Normal' if normality_y else 'Non-normal'})")

    # 3. Homoscedasticity check
    print(f"\n3. HOMOSCEDASTICITY")
    print("-" * 20)

    # Calculate residuals from linear regression
    residuals = y - linear_pred

    # Breusch-Pagan test for heteroscedasticity
    # Simple version: correlation between |residuals| and fitted values
    abs_residuals = np.abs(residuals)
    fitted_values = linear_pred

    hetero_corr, hetero_p = pearsonr(fitted_values, abs_residuals)
    homoscedasticity_ok = hetero_p > 0.05

    print(f"Breusch-Pagan-like test:")
    print(f"Correlation |residuals| vs fitted: r = {hetero_corr:.4f}, p = {hetero_p:.4f}")
    print(f"Homoscedasticity assumption: {'SATISFIED' if homoscedasticity_ok else 'VIOLATED'}")

    # 4. Outlier detection
    print(f"\n4. OUTLIER DETECTION")
    print("-" * 20)

    # Z-score method for univariate outliers
    #Note: Remember that Z-score indicates how many standard deviations a data point is from the mean of a distribution
    z_scores_x = np.abs(stats.zscore(x))
    z_scores_y = np.abs(stats.zscore(y))

    outliers_x = np.sum(z_scores_x > 3)
    outliers_y = np.sum(z_scores_y > 3)

    # Bivariate outliers using Mahalanobis distance
    data_array = np.column_stack([x, y])    # Combine x and y into a 2D array
    mean_vec = np.mean(data_array, axis=0)   # Calculate mean of x and mean of y
    cov_matrix = np.cov(data_array.T)    # Calculate covariance matrix

    try:
        inv_cov = np.linalg.inv(cov_matrix)   # Invert the covariance matrix
        mahal_dist = []
        for i in range(len(data_array)):
            diff = data_array[i] - mean_vec   # How far this point is from the center
            mahal_dist.append(np.sqrt(diff.T @ inv_cov @ diff))  # Mahalanobis distance formula

        mahal_dist = np.array(mahal_dist)
        # Chi-square critical value for 2 variables at 99.9% confidence
        chi2_critical = stats.chi2.ppf(0.999, df=2)    # Get chi-square threshold for 99.9% confidence
        bivariate_outliers = np.sum(mahal_dist**2 > chi2_critical)  # Count outliers

    except np.linalg.LinAlgError:
        bivariate_outliers = 0
        mahal_dist = np.zeros(len(data_array))

    total_outliers = max(outliers_x, outliers_y, bivariate_outliers)
    outliers_ok = total_outliers < (0.05 * n)  # Less than 5% outliers

    print(f"Univariate outliers (|z| > 3):")
    print(f"  {var1}: {outliers_x} ({outliers_x/n*100:.1f}%)")
    print(f"  {var2}: {outliers_y} ({outliers_y/n*100:.1f}%)")
    print(f"Bivariate outliers (Mahalanobis): {bivariate_outliers} ({bivariate_outliers/n*100:.1f}%)")
    print(f"Outlier assumption: {'SATISFIED' if outliers_ok else 'ATTENTION NEEDED'}")

    # Create diagnostic plots if requested
    if create_plots:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Correlation Assumptions: {var1} vs {var2}', fontsize=16)

        # Plot 1: Scatter plot with linear and quadratic fits
        axes[0, 0].scatter(x, y, alpha=0.6, s=30)

        # Linear fit
        x_line = np.linspace(x.min(), x.max(), 100)
        y_linear = np.polyval(linear_coef, x_line)
        y_quad = np.polyval(quad_coef, x_line)

        axes[0, 0].plot(x_line, y_linear, 'r-', linewidth=2, label=f'Linear (R² = {linear_r2:.3f})')
        axes[0, 0].plot(x_line, y_quad, 'g--', linewidth=2, label=f'Quadratic (R² = {quad_r2:.3f})')

        axes[0, 0].set_xlabel(var1)
        axes[0, 0].set_ylabel(var2)
        axes[0, 0].set_title('Linearity Check')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Residuals vs Fitted (homoscedasticity)
        axes[0, 1].scatter(fitted_values, residuals, alpha=0.6, s=30)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Fitted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Fitted (Homoscedasticity)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Q-Q plot for X variable
        stats.probplot(x, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title(f'Q-Q Plot: {var1}')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Q-Q plot for Y variable
        stats.probplot(y, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot: {var2}')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Outlier detection (Mahalanobis distance)
        if len(mahal_dist) > 0:
            axes[1, 1].scatter(range(len(mahal_dist)), mahal_dist**2, alpha=0.6, s=30)
            axes[1, 1].axhline(y=chi2_critical, color='red', linestyle='--',
                              label=f'Critical value = {chi2_critical:.1f}')
            axes[1, 1].set_xlabel('Observation Index')
            axes[1, 1].set_ylabel('Squared Mahalanobis Distance')
            axes[1, 1].set_title('Bivariate Outlier Detection')
            axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 2].scatter(fitted_values, sqrt_abs_residuals, alpha=0.6, s=30)
        axes[1, 2].set_xlabel('Fitted Values')
        axes[1, 2].set_ylabel('√|Residuals|')
        axes[1, 2].set_title('Scale-Location Plot')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Summary
    assumptions_met = sum([linearity_ok, normality_x, normality_y, homoscedasticity_ok, outliers_ok])
    total_assumptions = 5

    print(f"\n" + "=" * 50)
    print("ASSUMPTION CHECK SUMMARY")
    print("=" * 50)
    print(f"Assumptions satisfied: {assumptions_met}/{total_assumptions}")

    if assumptions_met >= 4:
        recommendation = "Pearson correlation is appropriate"
    elif assumptions_met >= 2:
        recommendation = "Consider both Pearson and Spearman correlations"
    else:
        recommendation = "Spearman correlation recommended"

    print(f"Recommendation: {recommendation}")

    return {
        'linearity': linearity_ok,
        'normality_x': normality_x,
        'normality_y': normality_y,
        'homoscedasticity': homoscedasticity_ok,
        'outliers_ok': outliers_ok,
        'assumptions_met': assumptions_met,
        'recommendation': recommendation,
        'pearson_r': r_pearson,
        'sample_size': n,
        'outlier_counts': {
            'univariate_x': outliers_x,
            'univariate_y': outliers_y,
            'bivariate': bivariate_outliers
        }
    }


def perform_spearman_correlation(data, continuous_vars):
    """
    Perform Spearman's rank correlation analysis as a non-parametric alternative.
    """
    print(f"\n" + "=" * 70)
    print("SPEARMAN'S RANK CORRELATION ANALYSIS")
    print("=" * 70)

    print("Spearman's rank correlation (ρ) measures monotonic relationships")
    print("It's robust to outliers and doesn't assume normality or linearity")
    print()

    # Calculate Spearman correlation matrix - using pandas .corr function - all pairs results simultaneously
    # Note: The .corr(method='spearman' method automatically converts data to ranks
    spearman_matrix = data[continuous_vars].corr(method='spearman')

    print("Spearman Correlation Matrix:")
    print("-" * 30)
    print(spearman_matrix.round(3))

    # Compare with Pearson correlations
    pearson_matrix = data[continuous_vars].corr(method='pearson')

# Just so we can print p-values, let us do this individually, for each pair of variables at a time
# using the scipy.stats functions, which returns the p-values
    print(f"\nSpearman vs Pearson Correlations:")
    print("-" * 35)

    spearman_results = []

    for i in range(len(continuous_vars)):
        for j in range(i+1, len(continuous_vars)):
            var1 = continuous_vars[i]
            var2 = continuous_vars[j]

            # Get aligned data first, then calculate correlations
            clean_data = data[[var1, var2]].dropna()
            x = clean_data[var1]
            y = clean_data[var2]

            # Spearman correlation (using function from scipy.stats)
            rho, p_spearman = spearmanr(x, y)

            # Pearson correlation for comparison (using function from scipy.stats)
            r_pearson, p_pearson = pearsonr(x, y)

            # Calculate difference
            diff = abs(rho - r_pearson)

            spearman_results.append({
                'var1': var1,
                'var2': var2,
                'spearman_rho': rho,
                'spearman_p': p_spearman,
                'pearson_r': r_pearson,
                'pearson_p': p_pearson,
                'difference': diff,
                'sample_size': len(clean_data)
            })

            # Interpret difference
            if diff < 0.1:
                agreement = "High agreement"
            elif diff < 0.2:
                agreement = "Moderate agreement"
            else:
                agreement = "Poor agreement"

            print(f"{var1} ↔ {var2}:")
            print(f"  Spearman ρ = {rho:>7.3f} (p = {p_spearman:>6.4f})")
            print(f"  Pearson  r = {r_pearson:>7.3f} (p = {p_pearson:>6.4f})")
            print(f"  Difference = {diff:>7.3f} ({agreement})")
            print()

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Spearman vs Pearson Correlation Comparison', fontsize=16)

    # Plot 1: Spearman correlation heatmap
    mask = np.triu(np.ones_like(spearman_matrix, dtype=bool))
    sns.heatmap(spearman_matrix,
                mask=mask,
                annot=True, annot_kws={"size": 9},cmap='RdBu_r', center=0,
                square=True, fmt='.3f', ax=axes[0, 0],
                cbar_kws={'label': 'Spearman ρ'})
    axes[0, 0].set_title('Spearman Correlation Matrix')

    # Plot 2: Pearson correlation heatmap
    sns.heatmap(pearson_matrix,
                mask=mask,
                annot=True, annot_kws={"size": 9},cmap='RdBu_r', center=0,
                square=True, fmt='.3f', ax=axes[0, 1],
                cbar_kws={'label': 'Pearson r'})
    axes[0, 1].set_title('Pearson Correlation Matrix')

    # Plot 3: Spearman vs Pearson scatter plot
    spearman_values = []
    pearson_values = []

    for result in spearman_results:
        spearman_values.append(result['spearman_rho'])
        pearson_values.append(result['pearson_r'])

    axes[1, 0].scatter(pearson_values, spearman_values, alpha=0.7, s=60)

    # Add diagonal line (perfect agreement)
    min_val = min(min(spearman_values), min(pearson_values))
    max_val = max(max(spearman_values), max(pearson_values))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--',
                   linewidth=2, label='Perfect agreement')

    axes[1, 0].set_xlabel('Pearson r')
    axes[1, 0].set_ylabel('Spearman ρ')
    axes[1, 0].set_title('Spearman vs Pearson Correlations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Difference magnitude histogram
    differences = [result['difference'] for result in spearman_results]
    axes[1, 1].hist(differences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('|Spearman ρ - Pearson r|')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Correlation Differences')
    axes[1, 1].grid(True, alpha=0.3)

    # Add statistics
    mean_diff = np.mean(differences)
    max_diff = np.max(differences)
    axes[1, 1].axvline(mean_diff, color='red', linestyle='--',
                      label=f'Mean difference = {mean_diff:.3f}')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print(f"Comparison Summary:")
    print("-" * 20)
    print(f"Mean absolute difference: {mean_diff:.3f}")
    print(f"Maximum difference: {max_diff:.3f}")
    print(f"Number of pairs with |diff| > 0.1: {sum(1 for d in differences if d > 0.1)}")
    print(f"Number of pairs with |diff| > 0.2: {sum(1 for d in differences if d > 0.2)}")

    return {
        'spearman_matrix': spearman_matrix,
        'pairwise_results': spearman_results,
        'comparison_stats': {
            'mean_difference': mean_diff,
            'max_difference': max_diff,
            'differences': differences
        }
    }


def analyze_data_correlations_comprehensive(data, continuous_vars):
    """
    Comprehensive correlation analysis of numerical features with both methods.
    """
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("=" * 70)

    # Check assumptions for key variable pairs
    key_pairs = [
    ('price', 'demand'),
    ('price', 'supply'),
    ('total_volume', 'demand'),
    ('total_volume', 'supply'),
    ('brazil', 'vietnam'),
    ('india', 'china'),
    ('indonesia', 'vietnam'),
    ('jordan_max_price', 'jordan_min_price'),
    ('vietnam_season', 'price'),
    ('vietnam_season', 'total_volume')
]

    assumption_results = {}

    for var1, var2 in key_pairs:
        print(f"\n Analyzing {var1} vs {var2}")
        assumption_results[f"{var1}_{var2}"] = check_correlation_assumptions(
            data, var1, var2, create_plots=True
        )

    # Summary of assumption checks
    print(f"\n" + "=" * 50)
    print("ASSUMPTION CHECK SUMMARY FOR ALL PAIRS")
    print("=" * 50)

    for pair_name, results in assumption_results.items():
        # FIXED: More robust string splitting
        parts = pair_name.split('_')
        # Reconstruct variable names properly
        if len(parts) >= 4:  # e.g., "bill_length_mm_body_mass_g"
            var1 = '_'.join(parts[:-3])  # Everything except last 3 parts
            var2 = '_'.join(parts[-3:])  # Last 3 parts
        else:
            var1, var2 = pair_name.split('_', 1)

        var1_clean = var1.replace('_', ' ')
        var2_clean = var2.replace('_', ' ')

        print(f"\n{var1_clean} ↔ {var2_clean}:")
        print(f"  Assumptions met: {results['assumptions_met']}/5")
        print(f"  Recommendation: {results['recommendation']}")
        print(f"  Pearson r = {results['pearson_r']:.3f}")

    return assumption_results



#############################################################################
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.stats.correlation_tools import corr_nearest, corr_clipped
import warnings
from itertools import combinations
import requests
from io import StringIO

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import math
##################################################################################


def demonstrate_partial_correlation_concepts():
    """
    Demonstrate partial correlation concepts with simulated data.
    """
    print(f"\n" + "=" * 70)
    print("PARTIAL CORRELATION CONCEPTS: SIMULATED EXAMPLES")
    print("=" * 70)

    np.random.seed(42)
    n = 200

    # Scenario 1: True confounding
    print(f"\nScenario 1: True Confounding")
    print("-" * 30)

    # Z (confounder) influences both X and Y
    z1 = np.random.normal(0, 1, n)
    x1 = 0.8 * z1 + np.random.normal(0, 0.6, n)  # X depends on Z
    y1 = 0.7 * z1 + np.random.normal(0, 0.7, n)  # Y depends on Z
    # X and Y are NOT directly related, only through Z

    # Scenario 2: Partial relationship
    print(f"Scenario 2: Partial Relationship")
    print("-" * 32)

    # Z influences both X and Y, but X and Y also have direct relationship
    z2 = np.random.normal(0, 1, n)
    x2 = 0.6 * z2 + np.random.normal(0, 0.8, n)  # X depends on Z
    y2 = 0.5 * z2 + 0.4 * x2 + np.random.normal(0, 0.6, n)  # Y depends on both Z and X

    # Scenario 3: Suppressor variable
    print(f"Scenario 3: Suppressor Variable")
    print("-" * 32)

    # Z suppresses the true X-Y relationship
    x3 = np.random.normal(0, 1, n)
    z3 = 0.6 * x3 + np.random.normal(0, 0.8, n)  # Z depends on X
    y3 = 0.8 * x3 - 0.5 * z3 + np.random.normal(0, 0.5, n)  # Y depends positively on X, negatively on Z

    scenarios = [
        ("True Confounding", x1, y1, z1, "X and Y related only through Z"),
        ("Partial Relationship", x2, y2, z2, "X and Y have both direct and indirect relationships"),
        ("Suppressor Variable", x3, y3, z3, "Z suppresses the true X-Y relationship")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Partial Correlation Scenarios', fontsize=16)

    for i, (title, x, y, z, description) in enumerate(scenarios):
        # Calculate correlations
        r_xy_simple, _ = pearsonr(x, y)

        # Calculate partial correlation manually
        r_xz, _ = pearsonr(x, z)
        r_yz, _ = pearsonr(y, z)

        numerator = r_xy_simple - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        r_xy_partial = numerator / denominator if denominator != 0 else np.nan

        # Plot 1: X vs Y scatter plot
        axes[i, 0].scatter(x, y, alpha=0.6, s=30, c=z, cmap='viridis')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        axes[i, 0].set_title(f'{title}\nSimple r = {r_xy_simple:.3f}')
        axes[i, 0].grid(True, alpha=0.3)

        # Plot 2: Partial correlation (residuals)
        # Regress X on Z, get residuals
        slope_xz = np.cov(x, z)[0, 1] / np.var(z)
        x_residuals = x - (slope_xz * z)

        # Regress Y on Z, get residuals
        slope_yz = np.cov(y, z)[0, 1] / np.var(z)
        y_residuals = y - (slope_yz * z)

        axes[i, 1].scatter(x_residuals, y_residuals, alpha=0.6, s=30)
        axes[i, 1].set_xlabel('X residuals (controlling for Z)')
        axes[i, 1].set_ylabel('Y residuals (controlling for Z)')
        axes[i, 1].set_title(f'Partial r = {r_xy_partial:.3f}')
        axes[i, 1].grid(True, alpha=0.3)

        # Add regression line to residuals plot
        if not np.isnan(r_xy_partial):
            z_line = np.polyfit(x_residuals, y_residuals, 1)
            p_line = np.poly1d(z_line)
            x_line = np.linspace(x_residuals.min(), x_residuals.max(), 100)
            axes[i, 1].plot(x_line, p_line(x_line), 'r-', linewidth=2)

        # Plot 3: Comparison bar chart
        correlations = [r_xy_simple, r_xy_partial]
        labels = ['Simple', 'Partial']
        colors = ['skyblue', 'lightcoral']

        bars = axes[i, 2].bar(labels, correlations, color=colors, alpha=0.7)
        axes[i, 2].set_ylabel('Correlation Coefficient')
        axes[i, 2].set_title('Simple vs Partial Correlation')
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_ylim(-1, 1)

        # Add value labels on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            axes[i, 2].text(bar.get_x() + bar.get_width()/2., height + 0.02 if height >= 0 else height - 0.05,
                           f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

        # Add scenario description
        axes[i, 0].text(0.02, 0.98, description, transform=axes[i, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       verticalalignment='top', fontsize=9)

        print(f"{title}:")
        print(f"  Simple correlation: {r_xy_simple:.3f}")
        print(f"  Partial correlation: {r_xy_partial:.3f}")
        print(f"  Difference: {r_xy_simple - r_xy_partial:.3f}")
        print(f"  Interpretation: {description}")
        print()

    plt.tight_layout()
    plt.show()




def calculate_partial_correlation(data, x_var, y_var, control_vars):
    """
    Calculate partial correlation between two variables controlling for others.

    """
    print(f"\nCalculating partial correlation:")
    print(f"X variable: {x_var}")
    print(f"Y variable: {y_var}")
    print(f"Controlling for: {', '.join(control_vars)}")

    # Get clean data
    all_vars = [x_var, y_var] + control_vars
    clean_data = data[all_vars].dropna()
    n = len(clean_data)

    print(f"Sample size: {n}")

    # Method 1: Using regression residuals (most intuitive)
    print(f"\nMethod 1: Regression Residuals Approach")
    print("-" * 45)

    # Step 1: Regress X on control variables, get residuals
    X_controls = clean_data[control_vars]
    X_target = clean_data[x_var]

    model_x = LinearRegression()
    model_x.fit(X_controls, X_target)
    x_residuals = X_target - model_x.predict(X_controls)

    # Step 2: Regress Y on control variables, get residuals
    Y_target = clean_data[y_var]

    model_y = LinearRegression()
    model_y.fit(X_controls, Y_target)
    y_residuals = Y_target - model_y.predict(X_controls)

    # Step 3: Correlate the residuals
    partial_r_residuals, p_value_residuals = pearsonr(x_residuals, y_residuals)

    print(f"Partial correlation (residuals method): {partial_r_residuals:.4f}")
    print(f"P-value: {p_value_residuals:.6f}")

    # Method 2: Direct formula for single control variable:
    # The direct formula method can only handle one control variable at a time.
    # But in our code we are Controlling for: p_color_red, p_color_green, p_color_yellow (3 control variables),
    # So this method will be skipped
    if len(control_vars) == 1:
        print(f"\nMethod 2: Direct Formula (Single Control)")
        print("-" * 45)

        z_var = control_vars[0]

        # Calculate simple correlations
        r_xy, _ = pearsonr(clean_data[x_var], clean_data[y_var])
        r_xz, _ = pearsonr(clean_data[x_var], clean_data[z_var])
        r_yz, _ = pearsonr(clean_data[y_var], clean_data[z_var])

        # Apply partial correlation formula
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

        if denominator != 0:
            partial_r_formula = numerator / denominator
        else:
            partial_r_formula = np.nan

        print(f"Simple correlations:")
        print(f"  r({x_var}, {y_var}) = {r_xy:.4f}")
        print(f"  r({x_var}, {z_var}) = {r_xz:.4f}")
        print(f"  r({y_var}, {z_var}) = {r_yz:.4f}")
        print(f"Partial correlation (formula): {partial_r_formula:.4f}")

        # Verify methods agree
        diff = abs(partial_r_residuals - partial_r_formula)
        print(f"Difference between methods: {diff:.6f}")

    # Calculate degrees of freedom and significance test
    df = n - len(control_vars) - 2

    # t-statistic for partial correlation
    if abs(partial_r_residuals) < 1:
        t_stat = partial_r_residuals * np.sqrt(df / (1 - partial_r_residuals**2))
        p_value_t = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    else:
        t_stat = np.inf
        p_value_t = 0.0

    print(f"\nSignificance test:")
    print(f"Degrees of freedom: {df}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value_t:.6f}")

    # Effect size interpretation
    if abs(partial_r_residuals) >= 0.5:
        effect_size = "Large"
    elif abs(partial_r_residuals) >= 0.3:
        effect_size = "Medium"
    elif abs(partial_r_residuals) >= 0.1:
        effect_size = "Small"
    else:
        effect_size = "Negligible"

    print(f"Effect size: {effect_size}")

    return {
        'partial_correlation': partial_r_residuals,
        'p_value': p_value_residuals,
        't_statistic': t_stat,
        'degrees_of_freedom': df,
        'effect_size': effect_size,
        'sample_size': n,
        'x_residuals': x_residuals,
        'y_residuals': y_residuals,
        'method_comparison': {
            'residuals_method': partial_r_residuals,
            'formula_method': partial_r_formula if len(control_vars) == 1 else None
        }
    }

# one-hot encoding categorical variables
def create_dummy_variables(data, categorical_var):
    """
    Create dummy variables for categorical variable to use in partial correlation.

    """
    # Create dummy variables (drop first to avoid multicollinearity): one-hot encoding categorical variables
    dummies = pd.get_dummies(data[categorical_var], prefix=categorical_var, drop_first=True)
    return dummies

def analyze_p_color_effects(data, continuous_vars):
    """
    Analyze how p_color affects correlations between morphological variables.

    """
    print(f"\n" + "=" * 70)
    print("p_color EFFECTS ON Pepper MORPHOLOGY CORRELATIONS")
    print("=" * 70)

    # Create p_color dummy variables
    p_color_dummies = create_dummy_variables(data, 'p_color')

    # Combine with continuous variables
    analysis_data = pd.concat([data[continuous_vars], p_color_dummies], axis=1)
    analysis_data = analysis_data.dropna()

    print(f"p_color dummy variables created:")
    for col in p_color_dummies.columns:
        print(f"  • {col}")

    # Analyze key morphological relationships
    key_relationships = [
    ('price', 'demand', 'Price ↔ Demand'),
    ('price', 'supply', 'Price ↔ Supply'),
    ('demand', 'supply', 'Demand ↔ Supply'),
    
    ('total_volume', 'demand', 'Total Volume ↔ Demand'),
    ('total_volume', 'supply', 'Total Volume ↔ Supply'),
    
    ('jordan_max_price', 'jordan_min_price', 'Max Price ↔ Min Price'),
    
    ('vietnam', 'brazil', 'Vietnam ↔ Brazil Production'),
    ('vietnam', 'india', 'Vietnam ↔ India Production'),
    ('indonesia', 'vietnam', 'Indonesia ↔ Vietnam Production'),
    ('china', 'india', 'China ↔ India Production'),
    
    ('vietnam_season', 'price', 'Season ↔ Price'),
    ('vietnam_season', 'total_volume', 'Season ↔ Volume')
]

    results = {}

    print(f"\nAnalyzing relationships with and without p_color control:")
    print("=" * 55)

    for x_var, y_var, description in key_relationships:
        print(f"\n{description}")
        print("-" * len(description))


        # Simple correlation (no control) - ensure same sample as partial correlation
        clean_pair_data = analysis_data[[x_var, y_var]].dropna()
        simple_r, simple_p = pearsonr(clean_pair_data[x_var], clean_pair_data[y_var])

        # Partial correlation controlling for p_color
        control_vars = list(p_color_dummies.columns)
        partial_result = calculate_partial_correlation(analysis_data, x_var, y_var, control_vars)

        # Calculate the difference
        difference = simple_r - partial_result['partial_correlation']

        print(f"\nSummary:")
        print(f"Simple correlation:    r = {simple_r:.4f} (p = {simple_p:.4f})")
        print(f"Partial correlation:   r = {partial_result['partial_correlation']:.4f} (p = {partial_result['p_value']:.4f})")
        print(f"Difference:            Δr = {difference:.4f}")

        # Interpretation
        if abs(difference) > 0.2:
            interpretation = "Large p_color effect - p_color is a major confounder"
        elif abs(difference) > 0.1:
            interpretation = "Moderate p_color effect - some confounding present"
        else:
            interpretation = "Small p_color effect - relationship is largely within-p_color"

        print(f"Interpretation: {interpretation}")

        results[f"{x_var}_{y_var}"] = {
            'description': description,
            'simple_correlation': simple_r,
            'simple_p_value': simple_p,
            'partial_correlation': partial_result['partial_correlation'],
            'partial_p_value': partial_result['p_value'],
            'difference': difference,
            'interpretation': interpretation,
            'x_residuals': partial_result['x_residuals'],
            'y_residuals': partial_result['y_residuals']
        }

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, len(key_relationships), figsize=(5*len(key_relationships), 20))
    fig.suptitle('Simple vs Partial Correlations: p_color Effects', fontsize=16)

    for i, (x_var, y_var, description) in enumerate(key_relationships):
        result = results[f"{x_var}_{y_var}"]

        # Top row: Simple correlations with p_color colored
        sns.scatterplot(data=data, x=x_var, y=y_var, hue='p_color', ax=axes[0, i], alpha=0.7,palette={'red': 'red', 'green': 'green', 'yellow': 'yellow'})

        # Add overall regression line
        x_vals = data[x_var].dropna()
        y_vals = data[y_var].dropna()
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        axes[0, i].plot(x_vals, p(x_vals), "r--", alpha=0.8, linewidth=2, label='Overall trend')

        axes[0, i].set_title(f'Simple Correlation\nr = {result["simple_correlation"]:.3f}')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        # Bottom row: Partial correlation (residuals plot)
        axes[1, i].scatter(result['x_residuals'], result['y_residuals'], alpha=0.6, s=30)

        # Add regression line for residuals
        z_res = np.polyfit(result['x_residuals'], result['y_residuals'], 1)
        p_res = np.poly1d(z_res)
        x_res_line = np.linspace(result['x_residuals'].min(), result['x_residuals'].max(), 100)
        axes[1, i].plot(x_res_line, p_res(x_res_line), "r-", linewidth=2)

        axes[1, i].set_title(f'Partial Correlation\nr = {result["partial_correlation"]:.3f}')
        axes[1, i].set_xlabel(f'{x_var} (residuals)')
        axes[1, i].set_ylabel(f'{y_var} (residuals)')
        axes[1, i].grid(True, alpha=0.3)

        # Add difference annotation
        diff_text = f'Δr = {result["difference"]:.3f}'
        axes[1, i].text(0.05, 0.95, diff_text, transform=axes[1, i].transAxes,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                       verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return results

def perform_within_p_color_correlations(data, continuous_vars):
    """
    Calculate correlations within each p_color separately.
    """
    print(f"\n" + "=" * 70)
    print("WITHIN-P_COLOR CORRELATION ANALYSIS")
    print("=" * 70)

    p_color_list = data['p_color'].unique()
    within_p_color_results = {}

    print(f"Analyzing correlations within each p_color:")

    for p_color in p_color_list:
        print(f"\n{p_color.upper()} PEPPER")
        print("=" * (len(p_color) + 9))

        p_color_data = data[data['p_color'] == p_color][continuous_vars].dropna()
        n_p_color = len(p_color_data)

        print(f"Sample size: {n_p_color}")

        # Calculate correlation matrix for this p_color
        p_color_corr = p_color_data.corr()

        print(f"\nCorrelation matrix:")
        print(p_color_corr.round(3))

        # Store results
        within_p_color_results[p_color] = {
            'correlation_matrix': p_color_corr,
            'sample_size': n_p_color,
            'data': p_color_data
        }

    # Create comparison visualization
    fig, axes = plt.subplots(1, len(p_color_list), figsize=(6*len(p_color_list), 7))
    fig.suptitle('Within-P_Color Correlation Matrices', fontsize=16)

    # Ensure axes is a list for consistent indexing
    if len(p_color_list) == 1:
        axes = [axes]

    for i, p_color in enumerate(p_color_list):
        corr_matrix = within_p_color_results[p_color]['correlation_matrix']

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True, annot_kws={"size": 7},cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', ax=axes[i],
                   cbar_kws={'label': 'Pearson r'})

        axes[i].set_title(f'{p_color}\n(n = {within_p_color_results[p_color]["sample_size"]})')

    plt.tight_layout()
    plt.show()
    # Compare correlations across p_color
    print(f"\nComparison across p_color:")
    print("-" * 30)

    # Get all unique variable pairs
    var_pairs = []
    for i in range(len(continuous_vars)):
        for j in range(i+1, len(continuous_vars)):
            var_pairs.append((continuous_vars[i], continuous_vars[j]))

    comparison_df = pd.DataFrame(index=[f"{var1} ↔ {var2}" for var1, var2 in var_pairs],
                                columns=p_color_list)

    for var1, var2 in var_pairs:
        pair_name = f"{var1} ↔ {var2}"
        for p_color in p_color_list:
            corr_val = within_p_color_results[p_color]['correlation_matrix'].loc[var1, var2]
            comparison_df.loc[pair_name, p_color] = corr_val

    print(comparison_df.round(3))

    # Calculate variance in correlations across p_color
    print(f"\nVariability in correlations across p_color:")
    print("-" * 45)

    for pair_name in comparison_df.index:
        correlations = comparison_df.loc[pair_name].values
        var_corr = np.var(correlations)
        range_corr = np.max(correlations) - np.min(correlations)

        print(f"{pair_name:<35}: variance = {var_corr:.4f}, range = {range_corr:.3f}")

    return within_p_color_results, comparison_df




def fishers_z_transform(r):
    """
    Apply Fisher's Z transformation to correlation coefficient.
    Formula: Z = 0.5 × ln((1 + r) / (1 - r))

    """
    # Handle edge cases where r is exactly ±1
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fishers_z(z):
    """
    Convert Fisher's Z back to correlation coefficient.

    """
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)



def correlation_confidence_interval(r, n, confidence_level=0.95):
    """
    Calculate confidence interval for a correlation coefficient using Fisher's Z transformation.

    """
    # Fisher's Z transformation
    z = fishers_z_transform(r)

    # Standard error of Z (1 over sq. root n-3)
    se_z = 1 / np.sqrt(n - 3)

    # Critical value for confidence interval
    alpha = 1 - confidence_level
    z_critical = stats.norm.ppf(1 - alpha/2)

    # Confidence interval for Z
    z_lower = z - z_critical * se_z
    z_upper = z + z_critical * se_z

    # Transform back to correlation scale
    r_lower = inverse_fishers_z(z_lower)
    r_upper = inverse_fishers_z(z_upper)

    return r_lower, r_upper, z, se_z

def comprehensive_correlation_matrix_analysis(data, continuous_vars, confidence_level=0.95):
    """
    Perform comprehensive correlation matrix analysis with confidence intervals and significance testing.
    """
    print(f"\n" + "=" * 70)
    print("COMPREHENSIVE CORRELATION MATRIX ANALYSIS")
    print("=" * 70)

    # Calculate correlation matrix with all pairwise statistics
    n_vars = len(continuous_vars)
    results_matrix = {}

    # Initialize result storage
    correlations = np.zeros((n_vars, n_vars))
    p_values = np.zeros((n_vars, n_vars))
    lower_bounds = np.zeros((n_vars, n_vars))
    upper_bounds = np.zeros((n_vars, n_vars))
    sample_sizes = np.zeros((n_vars, n_vars))

    print(f"Calculating correlations with {confidence_level*100:.0f}% confidence intervals:")
    print("-" * 60)

    detailed_results = []

    for i, var1 in enumerate(continuous_vars):
        for j, var2 in enumerate(continuous_vars):
            if i == j:
                # Diagonal elements
                correlations[i, j] = 1.0
                p_values[i, j] = 0.0
                lower_bounds[i, j] = 1.0
                upper_bounds[i, j] = 1.0
                sample_sizes[i, j] = len(data[var1].dropna())
            else:
                # Off-diagonal elements
                clean_data = data[[var1, var2]].dropna()
                x = clean_data[var1]
                y = clean_data[var2]
                n = len(clean_data)

                # Calculate correlation and significance
                r, p = pearsonr(x, y)

                # Calculate confidence interval
                r_lower, r_upper, fisher_z, se_z = correlation_confidence_interval(r, n, confidence_level)

                # Store results
                correlations[i, j] = r
                p_values[i, j] = p
                lower_bounds[i, j] = r_lower
                upper_bounds[i, j] = r_upper
                sample_sizes[i, j] = n

                # Store detailed results for unique pairs
                if i < j:
                    detailed_results.append({
                        'var1': var1,
                        'var2': var2,
                        'correlation': r,
                        'p_value': p,
                        'ci_lower': r_lower,
                        'ci_upper': r_upper,
                        'fisher_z': fisher_z,
                        'se_z': se_z,
                        'sample_size': n,
                        'ci_width': r_upper - r_lower
                    })

                    print(f"{var1} ↔ {var2}:")
                    print(f"  r = {r:>7.3f} [{r_lower:>6.3f}, {r_upper:>6.3f}], p = {p:>7.4f}, n = {n:>3}")

    # Create DataFrames for easy manipulation
    corr_df = pd.DataFrame(correlations, index=continuous_vars, columns=continuous_vars)
    p_values_df = pd.DataFrame(p_values, index=continuous_vars, columns=continuous_vars)

    print(f"\nCorrelation Matrix:")
    print("-" * 20)
    print(corr_df.round(3))

    # Multiple comparisons correction
    print(f"\nMultiple Comparisons Correction:")
    print("-" * 35)

    # Extract p-values for unique pairs (upper triangle)
    unique_p_values = []
    for result in detailed_results:
        unique_p_values.append(result['p_value'])

    # Apply different correction methods
    # We use: statsmodels.stats.multitest.multipletests where these can be inputs
    # Bonferroni correction works by dividing your desired significance level (like 0.05)
    # by the number of comparisons you're making. So if you're testing 6 pairs, the new threshold
    # becomes 0.05 ÷ 6 = 0.0083 — meaning each individual p-value must be less than 0.0083 to be considered significant.
    corrections = {
        'Bonferroni': 'bonferroni',
        'Holm': 'holm',
        'FDR (Benjamini-Hochberg)': 'fdr_bh',
        'FDR (Benjamini-Yekutieli)': 'fdr_by'
    }

    for method_name, method_code in corrections.items():
        rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(unique_p_values, method=method_code)

        # Update detailed results
        for i, result in enumerate(detailed_results):
            result[f'p_adjusted_{method_code}'] = p_adjusted[i]
            result[f'significant_{method_code}'] = rejected[i]

        significant_count = sum(rejected)
        print(f"{method_name:<25}: {significant_count:>2} significant out of {len(unique_p_values)}")

    return {
        'correlation_matrix': corr_df,
        'p_values_matrix': p_values_df,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'detailed_results': detailed_results,
        'sample_sizes': sample_sizes
    }



def create_advanced_correlation_visualizations(correlation_analysis, continuous_vars):
    """
    Create advanced visualizations for correlation analysis.
    """
    print(f"\n" + "=" * 70)
    print("ADVANCED CORRELATION VISUALIZATIONS")
    print("=" * 70)

    corr_matrix = correlation_analysis['correlation_matrix']
    p_values_matrix = correlation_analysis['p_values_matrix']
    detailed_results = correlation_analysis['detailed_results']

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))

    # 1. Enhanced correlation heatmap with significance stars
    ax1 = plt.subplot(2, 3, 1)

    # Create annotation matrix with significance stars
    annotations = corr_matrix.round(3).astype(str)

    for i in range(len(continuous_vars)):
        for j in range(len(continuous_vars)):
            if i != j:
                p_val = p_values_matrix.iloc[i, j]
                corr_val = corr_matrix.iloc[i, j]

                # Add significance stars
                if p_val < 0.001:
                    star = "***"
                elif p_val < 0.01:
                    star = "**"
                elif p_val < 0.05:
                    star = "*"
                else:
                    star = ""

                annotations.iloc[i, j] = f"{corr_val:.3f}{star}"

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix,
                mask=mask,
                annot=annotations, fmt='', cmap='RdBu_r', center=0,
                square=True, ax=ax1, cbar_kws={'label': 'Pearson r'})
    ax1.set_title('Correlation Matrix with Significance\n(* p<0.05, ** p<0.01, *** p<0.001)')

    # 2. Confidence interval visualization
    ax2 = plt.subplot(2, 3, 2)

    # Extract data for plotting
    correlations = [result['correlation'] for result in detailed_results]
    ci_lowers = [result['ci_lower'] for result in detailed_results]
    ci_uppers = [result['ci_upper'] for result in detailed_results]
    pair_names = [f"{result['var1'][:8]}↔{result['var2'][:8]}" for result in detailed_results]

    y_pos = np.arange(len(pair_names))

    # Create horizontal error bars
    ax2.errorbar(correlations, y_pos, xerr=[np.array(correlations) - np.array(ci_lowers),
                                           np.array(ci_uppers) - np.array(correlations)],
                fmt='o', capsize=5, capthick=2, markersize=8)

    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pair_names)
    ax2.set_xlabel('Correlation Coefficient')
    ax2.set_title('Correlations with 95% Confidence Intervals')
    ax2.grid(True, alpha=0.3)

    # 3. P-value distribution
    ax3 = plt.subplot(2, 3, 3)

    p_vals = [result['p_value'] for result in detailed_results]
    ax3.hist(p_vals, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
    ax3.set_xlabel('P-values')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of P-values')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Confidence interval widths
    ax4 = plt.subplot(2, 3, 4)

    ci_widths = [result['ci_upper'] - result['ci_lower'] for result in detailed_results]
    sample_sizes = [result['sample_size'] for result in detailed_results]

    ax4.scatter(sample_sizes, ci_widths, alpha=0.7, s=60)
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('95% CI Width')
    ax4.set_title('CI Width vs Sample Size')
    ax4.grid(True, alpha=0.3)

    # Add trendline
    z = np.polyfit(sample_sizes, ci_widths, 1)
    p = np.poly1d(z)
    ax4.plot(sample_sizes, p(sample_sizes), "r--", alpha=0.8)

    # 5. Multiple comparisons impact
    ax5 = plt.subplot(2, 3, 5)

    methods = ['Uncorrected', 'Bonferroni', 'Holm', 'FDR (BH)']
    significant_counts = []

    # Uncorrected
    uncorrected_sig = sum(1 for result in detailed_results if result['p_value'] < 0.05)
    significant_counts.append(uncorrected_sig)

    # Corrected methods
    for method in ['bonferroni', 'holm', 'fdr_bh']:
        method_sig = sum(1 for result in detailed_results if result[f'significant_{method}'])
        significant_counts.append(method_sig)

    bars = ax5.bar(methods, significant_counts, alpha=0.7,
                   color=['red', 'orange', 'yellow', 'lightgreen'])
    ax5.set_ylabel('Number of Significant Correlations')
    ax5.set_title('Impact of Multiple Comparisons Correction')
    ax5.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, significant_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{count}', ha='center', va='bottom')

    # 6. Correlation strength distribution
    ax6 = plt.subplot(2, 3, 6)

    abs_correlations = [abs(result['correlation']) for result in detailed_results]

    # Create bins for correlation strength
    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['Negligible\n(0-0.1)', 'Small\n(0.1-0.3)', 'Medium\n(0.3-0.5)',
              'Large\n(0.5-0.7)', 'Very Large\n(0.7-0.9)', 'Nearly Perfect\n(0.9-1.0)']

    counts, _ = np.histogram(abs_correlations, bins=bins)

    bars = ax6.bar(range(len(counts)), counts, alpha=0.7,
                   color=['lightblue', 'lightgreen', 'yellow', 'orange', 'red', 'darkred'])
    ax6.set_xticks(range(len(counts)))
    ax6.set_xticklabels(labels, rotation=45, ha='right')
    ax6.set_ylabel('Number of Correlations')
    ax6.set_title('Distribution of Correlation Strengths')
    ax6.grid(True, alpha=0.3)

    # Add percentage labels
    total_corrs = len(abs_correlations)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            percentage = (count / total_corrs) * 100
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{percentage:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()






def hierarchical_clustering_correlations(correlation_matrix, continuous_vars):
    """
    Perform hierarchical clustering on correlation matrix to identify variable groups.
    """
    print(f"\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING OF CORRELATION MATRIX")
    print("=" * 70)

    # Convert correlation matrix to distance matrix
    # Distance = 1 - |correlation|
    distance_matrix = 1 - np.abs(correlation_matrix.values)

    # Ensure distance matrix is symmetric and has zero diagonal (distance to itself is obviously zero)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Convert to condensed form for scipy
    condensed_distances = squareform(distance_matrix) # 2D to 1D. Linkage function expects distances in 1D format—not a full 2D matrix

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='ward') # Performs hierarchical agglomerative clustering using the Ward method.

    # Create dendrogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot dendrogram
    dendrogram_result = dendrogram(linkage_matrix, labels=continuous_vars,
                                  orientation='top', ax=ax1)
    ax1.set_title('Hierarchical Clustering of Variables\n(Based on Correlation Distance)')
    ax1.set_ylabel('Distance')
    ax1.tick_params(axis='x', rotation=45)

    # Reorder correlation matrix based on clustering
    cluster_order = dendrogram_result['leaves']
    reordered_vars = [continuous_vars[i] for i in cluster_order]
    reordered_corr = correlation_matrix.loc[reordered_vars, reordered_vars]

    # Plot reordered correlation matrix
    # mask = np.triu(np.ones_like(reordered_corr, dtype=bool))
    sns.heatmap(reordered_corr,
                #mask=mask,
                annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', ax=ax2,
                cbar_kws={'label': 'Pearson r'})
    ax2.set_title('Reordered Correlation Matrix\n(Clustered Variables)')

    plt.tight_layout()
    plt.show()

    # Extract clusters at different levels
    print(f"Variable clustering at different cut levels:")
    print("-" * 45)

    for n_clusters in [2, 3, 4]:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        print(f"\n{n_clusters} clusters:")
        for cluster_id in range(1, n_clusters + 1):
            cluster_vars = [continuous_vars[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            print(f"  Cluster {cluster_id}: {cluster_vars}")

    return {
        'linkage_matrix': linkage_matrix,
        'distance_matrix': distance_matrix,
        'cluster_order': cluster_order,
        'reordered_correlation_matrix': reordered_corr
    }



def bootstrap_correlation_confidence_intervals(data, var1, var2, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for correlation coefficient.

    """
    print(f"\n" + "=" * 60)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS: {var1} vs {var2}")
    print("=" * 60)

    # Get clean data
    clean_data = data[[var1, var2]].dropna()
    x = clean_data[var1].values
    y = clean_data[var2].values
    n = len(clean_data)

    print(f"Original sample size: {n}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Confidence level: {confidence_level*100:.0f}%")

    # Calculate original correlation
    original_r, original_p = pearsonr(x, y)

    # Bootstrap sampling
    bootstrap_correlations = []

    np.random.seed(42)  # For reproducibility

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]

        # Calculate correlation for bootstrap sample
        r_boot, _ = pearsonr(x_boot, y_boot)
        bootstrap_correlations.append(r_boot)

    bootstrap_correlations = np.array(bootstrap_correlations)

    # Calculate confidence intervals using percentile method
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower_boot = np.percentile(bootstrap_correlations, lower_percentile)
    ci_upper_boot = np.percentile(bootstrap_correlations, upper_percentile)

    # Calculate Fisher's Z confidence interval for comparison
    ci_lower_fisher, ci_upper_fisher, _, _ = correlation_confidence_interval(original_r, n, confidence_level)

    # Statistics
    boot_mean = np.mean(bootstrap_correlations)
    boot_std = np.std(bootstrap_correlations)
    boot_bias = boot_mean - original_r

    print(f"\nResults:")
    print("-" * 10)
    print(f"Original correlation: {original_r:.4f}")
    print(f"Bootstrap mean: {boot_mean:.4f}")
    print(f"Bootstrap bias: {boot_bias:.4f}")
    print(f"Bootstrap std: {boot_std:.4f}")

    print(f"\nConfidence Intervals:")
    print("-" * 22)
    print(f"Bootstrap CI:   [{ci_lower_boot:.4f}, {ci_upper_boot:.4f}]")
    print(f"Fisher's Z CI:  [{ci_lower_fisher:.4f}, {ci_upper_fisher:.4f}]")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Bootstrap distribution
    ax1.hist(bootstrap_correlations, bins=50, alpha=0.7, density=True,
            color='skyblue', edgecolor='black')

    # Add lines for original correlation and confidence interval
    ax1.axvline(original_r, color='red', linestyle='-', linewidth=2, label=f'Original r = {original_r:.3f}')
    ax1.axvline(ci_lower_boot, color='orange', linestyle='--', linewidth=2)
    ax1.axvline(ci_upper_boot, color='orange', linestyle='--', linewidth=2,
               label=f'Bootstrap 95% CI')
    ax1.axvline(boot_mean, color='green', linestyle=':', linewidth=2, label=f'Bootstrap mean = {boot_mean:.3f}')

    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Bootstrap Distribution of Correlations\n{var1} vs {var2}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Q-Q plot to check normality of bootstrap distribution
    stats.probplot(bootstrap_correlations, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Bootstrap Distribution Normality')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        'original_correlation': original_r,
        'original_p_value': original_p,
        'bootstrap_correlations': bootstrap_correlations,
        'bootstrap_mean': boot_mean,
        'bootstrap_std': boot_std,
        'bootstrap_bias': boot_bias,
        'ci_lower_bootstrap': ci_lower_boot,
        'ci_upper_bootstrap': ci_upper_boot,
        'ci_lower_fisher': ci_lower_fisher,
        'ci_upper_fisher': ci_upper_fisher
    }


def power_analysis_correlation(effect_sizes=None, sample_sizes=None, alpha=0.05, power=0.8):
    """
    Perform power analysis for correlation studies.

    """
    print(f"\n" + "=" * 70)
    print("POWER ANALYSIS FOR CORRELATION STUDIES")
    print("=" * 70)

    if effect_sizes is None:
        effect_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    if sample_sizes is None:
        sample_sizes = list(range(10, 201, 10))

    # Cohen's guidelines for effect sizes
    print(f"Cohen's guidelines for correlation effect sizes:")
    print(f"Small effect:    |r| = 0.10")
    print(f"Medium effect:   |r| = 0.30")
    print(f"Large effect:    |r| = 0.50")

    print(f"\nPower analysis parameters:")
    print(f"Alpha level: {alpha}")
    print(f"Desired power: {power}")

    # Function to calculate power for given r and n
    def calculate_power(r, n, alpha=0.05):
        if n <= 3:
            return 0.0

        # Calculate non-centrality parameter
        z_r = fishers_z_transform(r)
        se_z = 1 / np.sqrt(n - 3)

        # Critical value
        z_critical = stats.norm.ppf(1 - alpha/2)

        # Power calculation using non-central distribution
        # This is an approximation
        delta = abs(z_r) / se_z
        power = 1 - stats.norm.cdf(z_critical - delta) + stats.norm.cdf(-z_critical - delta)

        return min(power, 1.0)

    # Function to calculate required sample size
    def calculate_required_n(r, power=0.8, alpha=0.05):
        if abs(r) < 0.01:
            return float('inf')

        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # Fisher's Z transformation
        z_r = fishers_z_transform(r)

        # Required sample size formula
        n_required = ((z_alpha + z_beta) / z_r)**2 + 3

        return max(4, int(np.ceil(n_required)))

    # Calculate power curves
    print(f"\nSample size requirements for {power*100:.0f}% power:")
    print("-" * 45)

    required_ns = []
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        req_n = calculate_required_n(r, power, alpha)
        required_ns.append(req_n)
        print(f"r = {r:.1f}: n = {req_n:>4}")

    # Create power analysis visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Power Analysis for Correlation Studies', fontsize=16)

    # Plot 1: Power vs Sample Size for different effect sizes
    for r in [0.1, 0.2, 0.3, 0.4, 0.5]:
        powers = [calculate_power(r, n, alpha) for n in sample_sizes]
        ax1.plot(sample_sizes, powers, label=f'r = {r:.1f}', linewidth=2)

    ax1.axhline(y=power, color='red', linestyle='--', label=f'{power*100:.0f}% Power')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Statistical Power')
    ax1.set_title('Power vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Required Sample Size vs Effect Size
    effect_range = np.arange(0.1, 0.8, 0.05)
    required_sample_sizes = [calculate_required_n(r, power, alpha) for r in effect_range]

    # Cap extremely large sample sizes for visualization
    required_sample_sizes = [min(n, 1000) for n in required_sample_sizes]

    ax2.plot(effect_range, required_sample_sizes, 'b-', linewidth=2, marker='o')
    ax2.set_xlabel('Effect Size (|r|)')
    ax2.set_ylabel('Required Sample Size')
    ax2.set_title(f'Sample Size for {power*100:.0f}% Power')
    ax2.grid(True, alpha=0.3)

    # Add Cohen's guidelines
    ax2.axvline(x=0.1, color='green', linestyle=':', alpha=0.7, label='Small')
    ax2.axvline(x=0.3, color='orange', linestyle=':', alpha=0.7, label='Medium')
    ax2.axvline(x=0.5, color='red', linestyle=':', alpha=0.7, label='Large')
    ax2.legend(title='Effect Size')

    # Plot 3: Power vs Effect Size for different sample sizes
    sample_size_examples = [30, 50, 100, 200]
    effect_range_fine = np.arange(0.05, 0.8, 0.02)

    for n in sample_size_examples:
        powers = [calculate_power(r, n, alpha) for r in effect_range_fine]
        ax3.plot(effect_range_fine, powers, label=f'n = {n}', linewidth=2)

    ax3.axhline(y=power, color='red', linestyle='--', label=f'{power*100:.0f}% Power')
    ax3.set_xlabel('Effect Size (|r|)')
    ax3.set_ylabel('Statistical Power')
    ax3.set_title('Power vs Effect Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)

    # Plot 4: Minimum detectable effect size
    min_detectable_r = []
    for n in sample_sizes:
        # Binary search for minimum r that achieves desired power
        low, high = 0.01, 0.99
        while high - low > 0.001:
            mid = (low + high) / 2
            if calculate_power(mid, n, alpha) >= power:
                high = mid
            else:
                low = mid
        min_detectable_r.append(high)

    ax4.plot(sample_sizes, min_detectable_r, 'g-', linewidth=2, marker='s')
    ax4.set_xlabel('Sample Size')
    ax4.set_ylabel('Minimum Detectable Effect Size')
    ax4.set_title(f'Minimum Detectable |r| for {power*100:.0f}% Power')
    ax4.grid(True, alpha=0.3)

    # Add Cohen's guidelines
    ax4.axhline(y=0.1, color='green', linestyle=':', alpha=0.7, label='Small')
    ax4.axhline(y=0.3, color='orange', linestyle=':', alpha=0.7, label='Medium')
    ax4.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, label='Large')
    ax4.legend(title='Effect Size')

    plt.tight_layout()
    plt.show()

    return {
        'effect_sizes': effect_sizes,
        'sample_sizes': sample_sizes,
        'required_sample_sizes': dict(zip([0.1, 0.2, 0.3, 0.4, 0.5], required_ns)),
        'power_function': calculate_power,
        'sample_size_function': calculate_required_n
    }



def practical_correlation_reporting_guide(correlation_analysis):
    """
    Provide practical guide for reporting correlation results.

    """
    print(f"\n" + "=" * 70)
    print("PRACTICAL GUIDE FOR REPORTING CORRELATIONS")
    print("=" * 70)

    detailed_results = correlation_analysis['detailed_results']

    print(f"Essential elements for reporting correlations:")
    print("-" * 45)
    print(f"1. Correlation coefficient type (Pearson/Spearman)")
    print(f"2. Sample size and degrees of freedom")
    print(f"3. Confidence intervals")
    print(f"4. Statistical significance")
    print(f"5. Effect size interpretation")
    print(f"6. Multiple comparisons correction (if applicable)")

    print(f"\nExample reporting formats:")
    print("-" * 27)

    # Select a few examples for demonstration
    example_results = detailed_results[:3]  # First 3 correlations

    for i, result in enumerate(example_results, 1):
        var1 = result['var1'].replace('_', ' ')
        var2 = result['var2'].replace('_', ' ')
        r = result['correlation']
        p = result['p_value']
        ci_lower = result['ci_lower']
        ci_upper = result['ci_upper']
        n = result['sample_size']

        # Determine significance stars
        if p < 0.001:
            sig_text = "p < .001"
        elif p < 0.01:
            sig_text = f"p = {p:.3f}"
        elif p < 0.05:
            sig_text = f"p = {p:.3f}"
        else:
            sig_text = f"p = {p:.3f}"

        # Effect size interpretation
        if abs(r) >= 0.5:
            effect_desc = "large"
        elif abs(r) >= 0.3:
            effect_desc = "medium"
        elif abs(r) >= 0.1:
            effect_desc = "small"
        else:
            effect_desc = "negligible"

        direction = "positive" if r > 0 else "negative"

        print(f"\nExample {i}:")
        print(f"Standard format:")
        print(f"  'There was a {effect_desc} {direction} correlation between")
        print(f"   {var1} and {var2}, r({n-2}) = {r:.3f},")
        print(f"   95% CI [{ci_lower:.3f}, {ci_upper:.3f}], {sig_text}.'")

        print(f"APA format:")
        print(f"  '{var1.title()} and {var2} were significantly correlated,")
        print(f"   r = {r:.2f}, {sig_text}, 95% CI [{ci_lower:.2f}, {ci_upper:.2f}].'")

    # Create summary table for all results
    print(f"\n" + "=" * 50)
    print("SUMMARY TABLE FOR ALL CORRELATIONS")
    print("=" * 50)

    summary_data = []
    for result in detailed_results:
        summary_data.append({
            'Variables': f"{result['var1']} ↔ {result['var2']}",
            'r': f"{result['correlation']:.3f}",
            '95% CI': f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]",
            'p-value': f"{result['p_value']:.4f}",
            'n': result['sample_size'],
            'Significant': 'Yes' if result['p_value'] < 0.05 else 'No'
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Multiple comparisons impact
    print(f"\nMultiple comparisons correction impact:")
    print("-" * 40)

    uncorrected_sig = sum(1 for result in detailed_results if result['p_value'] < 0.05)
    bonferroni_sig = sum(1 for result in detailed_results if result.get('significant_bonferroni', False))
    fdr_sig = sum(1 for result in detailed_results if result.get('significant_fdr_bh', False))

    print(f"Uncorrected significant: {uncorrected_sig}/{len(detailed_results)}")
    print(f"Bonferroni significant:  {bonferroni_sig}/{len(detailed_results)}")
    print(f"FDR (BH) significant:    {fdr_sig}/{len(detailed_results)}")

    return summary_df



##########################################
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.stats.correlation_tools import corr_nearest, corr_clipped
import warnings
from itertools import combinations
import requests
from io import StringIO

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import math

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
import warnings
from itertools import combinations
import requests
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization style consistent with previous tutorials
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
##########################################


def check_normality(data, column):
    """Check if data follows a normal distribution using QQ-plot and Shapiro-Wilk test."""
    values = data[column]

    # Create a figure with QQ-plot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    stats.probplot(values, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot for {column}')

    plt.subplot(1, 2, 2)
    sns.histplot(values, kde=True)
    plt.title(f'Distribution of {column}')

    plt.tight_layout()
    plt.show()

    # Perform Shapiro-Wilk test for normality
    statistic, p_value = stats.shapiro(values)

    print(f"Normality check for {column}:")
    print(f"Shapiro-Wilk test: statistic = {statistic:.4f}, p-value = {p_value:.8f}")

    if p_value < 0.05:
        print("The data significantly deviates from a normal distribution (p < 0.05).")
    else:
        print("The data appears to follow a normal distribution (p >= 0.05).")

    # Calculate skewness and kurtosis
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)

    print(f"Skewness: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("  The distribution is approximately symmetric.")
    elif skewness > 0.5:
        print("  The distribution is right-skewed (positively skewed).")
    else:  # skewness < -0.5
        print("  The distribution is left-skewed (negatively skewed).")

    print(f"Kurtosis: {kurtosis:.4f}")
    if abs(kurtosis) < 0.5:
        print("  The distribution has a similar tail weight as the normal distribution.")
    elif kurtosis > 0.5:
        print("  The distribution is leptokurtic (heavier tails, more outliers than normal).")
    else:  # kurtosis < -0.5
        print("  The distribution is platykurtic (lighter tails, fewer outliers than normal).")
    print("\n")
    print("==" * 50)


def detect_outliers(data, column):
    """Detect and visualize outliers using the IQR method."""
    values = data[column]

    # Calculate Q1, Q3, and IQR
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    # Define outlier boundaries
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Find outliers
    outliers = values[(values < lower_bound) | (values > upper_bound)]

    print(f"Outlier detection for {column}:")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Percentage of outliers: {(len(outliers) / len(values)) * 100:.2f}%")
    print(f"Outlier boundaries: Lower = {lower_bound:.4f}, Upper = {upper_bound:.4f}")

    if len(outliers) > 0:
        print("Outlier values:")
        print(outliers.values)

    # Visualize with box plot
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=values)
    plt.title(f'Box Plot of {column} Showing Outliers')
    plt.ylabel('Value')

    plt.subplot(1, 2, 2)
    sns.histplot(values, kde=True)
    plt.axvline(lower_bound, color='r', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
    plt.axvline(upper_bound, color='r', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
    plt.title(f'Distribution of {column} with Outlier Boundaries')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("==" * 50)

    # Return the outliers
    return outliers



# Function to check normality
def check_normality_advanced(data, column):
    """Check if data follows a normal distribution with visualizations and statistical tests."""
    values = data[column]

    # Create a figure for plots
    fig, axes = plt.subplots(2, 2, figsize=(20, 7))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Plot 1: Histogram with normal curve
    ax1 = axes[0, 0]
    sns.histplot(values, kde=True, ax=ax1)

    # Fit a normal distribution to the data and overlay it
    mu, std = stats.norm.fit(values)
    x = np.linspace(min(values), max(values), 100)
    p = stats.norm.pdf(x, mu, std)
    ax1.plot(x, p * len(values) * (max(values) - min(values)) / 100,
            'r-', linewidth=2, label=f'Normal: μ={mu:.2f}, σ={std:.2f}')

    ax1.axvline(mu, color='r', linestyle='--', alpha=0.8, label='Mean')
    ax1.axvline(mu + std, color='g', linestyle='-.', alpha=0.8, label='μ ± 1σ')
    ax1.axvline(mu - std, color='g', linestyle='-.', alpha=0.8)

    ax1.set_title(f'Histogram of {column} with Normal Curve')
    ax1.set_xlabel(column)
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Plot 2: QQ plot
    ax2 = axes[0, 1]
    stats.probplot(values, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot for {column}')

    # Plot 3: Box plot
    ax3 = axes[1, 0]
    sns.boxplot(y=values, ax=ax3)
    ax3.set_title(f'Box Plot of {column}')
    ax3.set_ylabel(column)

    # Plot 4: Skewness and Kurtosis visualization
    ax4 = axes[1, 1]

    # Calculate skewness and kurtosis
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)

    # Create a labeled bar chart
    metrics = ['Skewness', 'Kurtosis']
    values_metrics = [skewness, kurtosis]

    sns.barplot(x=metrics, y=values_metrics, palette='viridis', ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title(f'Skewness and Kurtosis for {column}')

    # Add text annotations on bars
    for i, v in enumerate(values_metrics):
        ax4.text(i, v + 0.1 if v >= 0 else v - 0.2, f'{v:.3f}',
                ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.suptitle(f'Normality Analysis for {column}', fontsize=16, y=1.02)
    plt.show()

    # Perform statistical tests for normality
    print(f"\nNormality Tests for {column}:")

    # Shapiro-Wilk test - best for sample sizes < 2000
    stat_shapiro, p_shapiro = stats.shapiro(values)
    print(f"Shapiro-Wilk Test: statistic={stat_shapiro:.4f}, p-value={p_shapiro:.6f}")
    if p_shapiro < 0.05:
        result_shapiro = "Reject H0: Data is NOT normally distributed"
    else:
        result_shapiro = "Fail to reject H0: Data looks normally distributed"
    print(f"Interpretation (p << 0.05): {result_shapiro}")
    print("Note that.. H₀: The data follows a normal distribution")
    print(" ")

    # Anderson-Darling test
    result_ad = stats.anderson(values, dist='norm')
    print(f"Anderson-Darling Test: statistic={result_ad.statistic:.4f}")
    print("Critical values at significance levels:")
    for i, significance in enumerate(result_ad.critical_values):
        sig_level = [15, 10, 5, 2.5, 1][i]
        if result_ad.statistic > significance:
            result_ad_i = "Reject H0: Data is NOT normally distributed"
        else:
            result_ad_i = "Fail to reject H0: Data looks normally distributed"
        print(f"  {sig_level}%: {significance:.4f} - {result_ad_i}")

    print("Note that for Anderson-Darling test - If test statistic > critical value: Reject H₀ (data is NOT normal)")
    print(" ")

    # Calculate and print skewness and kurtosis
    print(f"\nSkewness: {skewness:.4f}")
    if abs(skewness) < 0.5:
        print("  The distribution is approximately symmetric")
    elif skewness < -0.5:
        print("  The distribution is negatively skewed (left-tailed)")
    else:  # skewness > 0.5
        print("  The distribution is positively skewed (right-tailed)")
    print("Note that ...")
    print("Skewness = 0: Perfectly symmetric distribution \n Skewness > 0: Right-skewed (positive skew) - the right tail is longer \n Skewness < 0: Left-skewed (negative skew) - the left tail is longer")
    print(" In our code, we set a limit of 0.5. So if the skewness < 0.5, we say that the distribution is approximately symmetric")
    print(" ")
    print(f"Kurtosis: {kurtosis:.4f}")
    if abs(kurtosis) < 0.5:
        print("  The distribution has kurtosis similar to normal distribution (mesokurtic)")
    elif kurtosis < -0.5:
        print("  The distribution has lighter tails than normal (platykurtic)")
    else:  # kurtosis > 0.5
        print("  The distribution has heavier tails than normal (leptokurtic)")
    print("Note that ...")
    print("Kurtosis = 0: Same tailedness as a normal distribution (mesokurtic) \n Kurtosis > 0: Heavier tails than normal (leptokurtic) - more outliers \n Kurtosis < 0: Lighter tails than normal (platykurtic) - fewer outliers")
    print(" In our code, we set a limit of 0.5. So if kurtosis < 0.5, we say that he distribution has kurtosis similar to normal distribution (mesokurtic)")
    print(" ")

    print("==" * 50)


###############################
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.power import TTestIndPower, TTestPower
import warnings
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu, wilcoxon, ranksums
from scipy.stats import fisher_exact
from scipy.stats import median_test
from scipy.stats import binomtest  # For sign test

# Suppress warnings
warnings.filterwarnings('ignore')
###############################


def perform_mannwhitney_test(data, group_var, test_var, display_name=None):
    '''
    we can use Mann-Whitney U test which is a non-parametric test that compares the distributions of two independent groups.
     It does not assume normality and is based on the ranks of the data rather than the raw values.
    '''

    if display_name is None:
        display_name = test_var

    group_values = data[group_var].unique()
    if len(group_values) != 2:
        print(f"Error: {group_var} must have exactly 2 unique values for Mann-Whitney test.")
        return

    group1 = data[data[group_var] == group_values[0]][test_var].dropna()  
    group2 = data[data[group_var] == group_values[1]][test_var].dropna()  

    # Calculate descriptive statistics for each group
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    median1, median2 = group1.median(), group2.median()
    std1, std2 = group1.std(), group2.std()
    iqr1 = group1.quantile(0.75) - group1.quantile(0.25)
    iqr2 = group2.quantile(0.75) - group2.quantile(0.25)

    print(f"\nMann-Whitney U Test: Comparing {display_name} between {group_var} is {group_values[0]} and {group_var} is {group_values[1]} groups")
    print(f"\nDescriptive Statistics:")
    print(f"  {group_values[0]}: n = {n1}, Mean = {mean1:.2f}, Median = {median1:.2f}, SD = {std1:.2f}, IQR = {iqr1:.2f}")
    print(f"  {group_values[1]}: n = {n2}, Mean = {mean2:.2f}, Median = {median2:.2f}, SD = {std2:.2f}, IQR = {iqr2:.2f}")

    # Perform Mann-Whitney U test
    '''
    Note: 'alternative' parameter set to 'two-sided' for a two-tailed test
    '''
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    '''
    Calculate effect size (r = Z / sqrt(N))
    Convert U to Z score for the effect size calculation
    '''
    n_total = n1 + n2
    mean_u = (n1 * n2) / 2
    std_u = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    z_score = (u_stat - mean_u) / std_u
    effect_size_r = abs(z_score) / np.sqrt(n_total)

    # Interpret effect size
    if effect_size_r < 0.1:
        effect_interpretation = "negligible"
    elif effect_size_r < 0.3:
        effect_interpretation = "small"
    elif effect_size_r < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    # Print Mann-Whitney U test results
    print("\nMann-Whitney U Test Results:")
    print(f"  U statistic = {u_stat:.2f}")
    print(f"  p-value = {p_value:.6f}")
    print(f"  Effect size r = {effect_size_r:.4f} ({effect_interpretation} effect)")

    # Print interpretation
    if p_value < 0.05:
        print(f"  Result: Statistically significant difference in {display_name} between groups")
        print(f"  The distributions of {display_name} for {group_values[0]} and {group_values[1]} are significantly different")
    else:
        print(f"  Result: No statistically significant difference in {display_name} between groups")
        print(f"  No evidence that the distributions of {display_name} differ between {group_values[0]} and {group_values[1]}")
  

    # Create visualizations
    plt.figure(figsize=(20, 7))

    # 1. Box plots
    plt.subplot(2, 2, 1)
    sns.boxplot(x=group_var, y=test_var, data=data)
    plt.title(f'Box Plot of {display_name} by {group_var}')
    plt.ylabel(display_name)

    # 2. Violin plots
    plt.subplot(2, 2, 2)
    sns.violinplot(x=group_var, y=test_var, data=data, inner='quartile')
    plt.title(f'Violin Plot of {display_name} by {group_var}')
    plt.ylabel(display_name)

    # 3. Overlapping histograms
    plt.subplot(2, 2, 3)
    sns.histplot(group1, color='blue', alpha=0.5, label=group_values[0], kde=True)
    sns.histplot(group2, color='orange', alpha=0.5, label=group_values[1], kde=True)
    plt.title(f'Distribution of {display_name} by {group_var}')
    plt.xlabel(display_name)
    plt.legend()

    # 4. Emperical Cumulative distribution function (ECDF) plot - useful for Mann-Whitney U test
    plt.subplot(2, 2, 4)

    # Calculate ECDF for each group
    def ecdf(x):
        # Count the proportion of values less than or equal to each value
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x1, y1 = ecdf(group1)
    x2, y2 = ecdf(group2)

    plt.step(x1, y1, label=group_values[0], where='post')
    plt.step(x2, y2, label=group_values[1], where='post')

    plt.title(f'Empirical Cumulative Distribution of {display_name}')
    plt.xlabel(display_name)
    plt.ylabel('Cumulative Probability')
    plt.legend()

    # Add p-value annotation to the plot
    sig_text = f"Mann-Whitney U Test\np = {p_value:.4f}"
    if p_value < 0.05:
        sig_text += "\n(Significant)"
    else:
        sig_text += "\n(Not Significant)"

    plt.figtext(0.5, 0.01, sig_text, ha='center', fontsize=12,
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.suptitle(f'Mann-Whitney U Test: {display_name} by {group_var}', fontsize=16, y=1.05)
    plt.show()

    # Return results for further analysis if needed
    return {
        'u_statistic': u_stat,
        'p_value': p_value,
        'effect_size_r': effect_size_r,
        'n1': n1,
        'n2': n2
    }



def explore_categorical_data(data):
    """
    Comprehensive exploration of the categorical_data dataset with focus on price differences
    between p_color (our main ANOVA example).

    Parameters:
    -----------
    data : pandas.DataFrame
        The p_color dataset
    """
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS: price by p_color")
    print("=" * 60)

    dependent_var = 'price'
    group_var = 'p_color'

    # Calculate descriptive statistics by species
    print("\nDescriptive Statistics by p_color:")
    print("-" * 45)

    species_stats = data.groupby(group_var)[dependent_var].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)

    print(species_stats)

    # Calculate overall statistics
    overall_mean = data[dependent_var].mean()
    overall_std = data[dependent_var].std()
    overall_n = len(data)

    print(f"\nOverall Statistics:")
    print(f"Mean body mass: {overall_mean:.2f} g")
    print(f"Standard deviation: {overall_std:.2f} g")
    print(f"Total sample size: {overall_n}")

    # Calculate some preliminary ANOVA components for illustration
    print(f"\nPreliminary ANOVA Concepts:")
    print("-" * 35)

    # Between-group variation (simplified illustration)
    species_means = data.groupby(group_var)[dependent_var].mean()
    species_n = data.groupby(group_var)[dependent_var].count()

    print("p_color means:")
    for species, mean in species_means.items():
        n = species_n[species]
        print(f"  {species}: {mean:.2f} g (n = {n})")

    # Calculate range of means as a simple measure of between-group variation
    mean_range = species_means.max() - species_means.min()
    print(f"\nRange of p_color means: {mean_range:.2f} g")
    print(f"This represents the between-group variation we'll analyze with ANOVA")

    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Exploratory Data Analysis: price by p_color', fontsize=16)

    # 1. Box plot
    sns.boxplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot: price by p_color')
    axes[0, 0].set_ylabel('price')

    # 2. Violin plot
    sns.violinplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 1])
    axes[0, 1].set_title('Violin Plot: price by p_color')
    axes[0, 1].set_ylabel('price')

    # 3. Strip plot with jitter
    sns.stripplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 2],
                  alpha=0.6, size=4)
    axes[0, 2].set_title('Strip Plot: Individual Data Points')
    axes[0, 2].set_ylabel('price')

    # 4. Histograms by p_color  
    species_list = data[group_var].unique()
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    for i, species in enumerate(species_list):
        species_data = data[data[group_var] == species][dependent_var]
        axes[1, 0].hist(species_data, alpha=0.7, label=species, color=colors[i % len(colors)],
                        bins=15, density=True)

    axes[1, 0].set_title('Overlapping Histograms')
    axes[1, 0].set_xlabel('price')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()

    # 5. Mean plot with error bars (standard error)
    means = species_stats['mean']
    stds = species_stats['std']
    ns = species_stats['count']
    ses = stds / np.sqrt(ns)  # Standard errors

    x_pos = range(len(means))
    axes[1, 1].bar(x_pos, means, yerr=ses, capsize=5, alpha=0.7,
                   color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Mean price with Standard Error')
    axes[1, 1].set_xlabel('p_color')
    axes[1, 1].set_ylabel('price')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(means.index)

    # Add value labels on bars
    for i, (mean, se) in enumerate(zip(means, ses)):
        axes[1, 1].text(i, mean + se + 50, f'{mean:.0f}', ha='center', fontweight='bold')

    # 6. Q-Q plots for normality check (combined)
    from scipy.stats import probplot

    all_residuals = []
    for species in species_list:
        species_data = data[data[group_var] == species][dependent_var]
        species_mean = species_data.mean()
        residuals = species_data - species_mean
        all_residuals.extend(residuals)

    probplot(all_residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot: Normality Check of Residuals')

    plt.tight_layout()
    plt.show()

    # Additional analysis: Effect size preview
    print(f"\nEffect Size Preview:")
    print("-" * 25)

    # Calculate total sum of squares
    sst = np.sum((data[dependent_var] - overall_mean) ** 2)

    # Calculate between-group sum of squares
    ssb = 0
    for species in species_list:
        species_data = data[data[group_var] == species][dependent_var]
        species_mean = species_data.mean()
        species_n = len(species_data)
        ssb += species_n * (species_mean - overall_mean) ** 2

    # Calculate within-group sum of squares
    ssw = sst - ssb

    # Calculate eta squared (effect size)
    eta_squared = ssb / sst

    print(f"Total Sum of Squares (SST): {sst:,.2f}")
    print(f"Between-group Sum of Squares (SSB): {ssb:,.2f}")
    print(f"Within-group Sum of Squares (SSW): {ssw:,.2f}")
    print(f"Eta squared (η²): {eta_squared:.4f}")

    if eta_squared < 0.01:
        effect_interpretation = "negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "small"
    elif eta_squared < 0.14:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"

    print(f"Effect size interpretation: {effect_interpretation}")

    return species_stats


def check_anova_assumptions(data, dependent_var, group_var, alpha=0.05):
    """
    Comprehensive check of ANOVA assumptions with visualizations and statistical tests.

    """
    print("\n" + "=" * 60)
    print("ANOVA ASSUMPTIONS CHECKING")
    print("=" * 60)

    # Fit a simple model to get residuals
    # We'll use a simple approach: residuals = value - group mean
    residuals = []
    fitted_values = []
    group_labels = []

    for group in data[group_var].unique():
        group_data = data[data[group_var] == group][dependent_var]
        group_mean = group_data.mean()
        group_residuals = group_data - group_mean

        residuals.extend(group_residuals)
        fitted_values.extend([group_mean] * len(group_data))
        group_labels.extend([group] * len(group_data))

    residuals = np.array(residuals)
    fitted_values = np.array(fitted_values)

    # Create a comprehensive figure for assumption checking
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ANOVA Assumptions Checking', fontsize=16)

    # 1. Normality Check: Q-Q Plot of Residuals
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot: Normality of Residuals')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Normality Check: Histogram of Residuals
    axes[0, 1].hist(residuals, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Overlay normal curve
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    axes[0, 1].set_title('Histogram of Residuals with Normal Curve')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Homogeneity of Variance: Residuals vs Fitted Values
    axes[0, 2].scatter(fitted_values, residuals, alpha=0.6)
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_title('Residuals vs Fitted Values')
    axes[0, 2].set_xlabel('Fitted Values')
    axes[0, 2].set_ylabel('Residuals')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Box plot of residuals by group (homogeneity check)
    residuals_df = pd.DataFrame({
        'residuals': residuals,
        'group': group_labels
    })
    sns.boxplot(data=residuals_df, x='group', y='residuals', ax=axes[1, 0])
    axes[1, 0].set_title('Residuals by Group (Homogeneity Check)')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 5. Scale-Location plot (Square root of absolute residuals vs fitted values)
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    axes[1, 1].scatter(fitted_values, sqrt_abs_residuals, alpha=0.6)
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Residuals|')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cook's Distance (outlier detection)
    # Simplified Cook's distance calculation
    n = len(residuals)
    p = len(data[group_var].unique())  # number of parameters
    mse = np.sum(residuals**2) / (n - p)

    # Leverage calculation (simplified)
    leverage = []
    for group in data[group_var].unique():
        group_size = len(data[data[group_var] == group])
        group_leverage = 1 / group_size  # Simplified leverage
        leverage.extend([group_leverage] * group_size)

    leverage = np.array(leverage)
    cooks_d = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)

    axes[1, 2].stem(range(len(cooks_d)), cooks_d, basefmt=" ")
    axes[1, 2].axhline(y=4/n, color='r', linestyle='--', label=f'Threshold: 4/n = {4/n:.3f}')
    axes[1, 2].set_title("Cook's Distance")
    axes[1, 2].set_xlabel('Observation')
    axes[1, 2].set_ylabel("Cook's Distance")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistical tests for assumptions
    print("\nStatistical Tests for Assumptions:")
    print("-" * 40)

    # 1. Normality of residuals: Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"1. Normality of Residuals (Shapiro-Wilk Test):")
    print(f"   Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.6f}")
    if shapiro_p > alpha:
        print(f"   ✓ Residuals appear normally distributed (p > {alpha})")
    else:
        print(f"   ⚠ Residuals deviate from normality (p ≤ {alpha})")
        print(f"   Consider: transformation, robust methods, or non-parametric alternatives")

    # 2. Homogeneity of variance: Levene's test
    groups = [data[data[group_var] == group][dependent_var] for group in data[group_var].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    print(f"\n2. Homogeneity of Variance (Levene's Test):")
    print(f"   Statistic = {levene_stat:.4f}, p-value = {levene_p:.6f}")
    if levene_p > alpha:
        print(f"   ✓ Variances appear equal across groups (p > {alpha})")
    else:
        print(f"   ⚠ Variances appear unequal (p ≤ {alpha})")
        print(f"   Consider: Welch's ANOVA, transformation, or robust methods")

    # 3. Outlier detection: Cook's distance
    outlier_threshold = 4 / n
    outliers = np.where(cooks_d > outlier_threshold)[0]
    print(f"\n3. Outlier Detection (Cook's Distance > {outlier_threshold:.3f}):")
    if len(outliers) == 0:
        print(f"   ✓ No influential outliers detected")
    else:
        print(f"   ⚠ {len(outliers)} potential outliers detected at positions: {outliers}")
        print(f"   Consider: investigating these observations or using robust methods")


    # Summary and recommendations
    print(f"\n" + "=" * 50)
    print("ASSUMPTION CHECK SUMMARY")
    print("=" * 50)

    assumptions_met = 0

    if shapiro_p > alpha:
        print("✓ Normality: SATISFIED")
        assumptions_met += 1
    else:
        print("⚠ Normality: VIOLATED")

    if levene_p > alpha:
        print("✓ Homogeneity of variance: SATISFIED")
        assumptions_met += 1
    else:
        print("⚠ Homogeneity of variance: VIOLATED")

    if len(outliers) == 0:
        print("✓ No influential outliers: SATISFIED")
        assumptions_met += 1
    else:
        print("⚠ Outliers present: ATTENTION NEEDED")

    print("✓ Independence: ASSUMED (design-based)")

    print(f"\nAssumptions satisfied: {assumptions_met}/3 testable assumptions")

    if assumptions_met == 3:
        print("All assumptions satisfied - ANOVA is appropriate!")
    elif assumptions_met >= 2:
        print("  Most assumptions satisfied - ANOVA likely robust")
        print("   Consider reporting both parametric and non-parametric results")
    else:
        print(" Multiple assumptions violated - Consider alternatives:")
        print("   • Data transformation")
        print("   • Welch's ANOVA (for unequal variances)")
        print("   • Kruskal-Wallis test (non-parametric)")
        print("   • Robust ANOVA methods")

    return {
        'shapiro_p': shapiro_p,
        'levene_p': levene_p,
        'outliers': outliers,
        'residuals': residuals,
        'assumptions_met': assumptions_met
    }


def perform_manual_anova(data, dependent_var, group_var):
    """
    Perform ANOVA with manual calculations to illustrate the underlying mathematics.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    """
    print("\n" + "=" * 60)
    print("MANUAL ANOVA CALCULATIONS")
    print("=" * 60)

    # Basic information
    groups = data[group_var].unique()
    k = len(groups)  # number of groups
    N = len(data)    # total sample size

    print(f"Number of groups (k): {k}")
    print(f"Total sample size (N): {N}")
    print(f"Groups: {list(groups)}")

    # Calculate overall mean
    overall_mean = data[dependent_var].mean()
    print(f"\nOverall mean: {overall_mean:.2f}")

    # Calculate group statistics
    print(f"\nGroup Statistics:")
    print("-" * 20)

    group_stats = {}
    for group in groups:
        group_data = data[data[group_var] == group][dependent_var]
        group_mean = group_data.mean()
        group_n = len(group_data)
        group_var_calc = group_data.var(ddof=1)  # Sample variance

        group_stats[group] = {
            'mean': group_mean,
            'n': group_n,
            'variance': group_var_calc,
            'data': group_data
        }

        print(f"{group}: n={group_n}, mean={group_mean:.2f}, variance={group_var_calc:.2f}")

    # Step 1: Calculate Total Sum of Squares (SST)
    sst = np.sum((data[dependent_var] - overall_mean) ** 2)
    print(f"\nStep 1: Total Sum of Squares (SST)")
    print(f"SST = Σ(x_i - x̄)² = {sst:.2f}")

    # Step 2: Calculate Between-groups Sum of Squares (SSB)
    ssb = 0
    print(f"\nStep 2: Between-groups Sum of Squares (SSB)")
    print("SSB = Σn_j(x̄_j - x̄)²")

    for group in groups:
        group_mean = group_stats[group]['mean']
        group_n = group_stats[group]['n']
        contribution = group_n * (group_mean - overall_mean) ** 2
        ssb += contribution
        print(f"  {group}: {group_n} × ({group_mean:.2f} - {overall_mean:.2f})² = {contribution:.2f}")

    print(f"SSB = {ssb:.2f}")

    # Step 3: Calculate Within-groups Sum of Squares (SSW)
    ssw = sst - ssb  # Alternative: ssw = sum of individual group variances
    print(f"\nStep 3: Within-groups Sum of Squares (SSW)")
    print(f"SSW = SST - SSB = {sst:.2f} - {ssb:.2f} = {ssw:.2f}")

    # Verify with alternative calculation
    ssw_alt = 0
    for group in groups:
        group_data = group_stats[group]['data']
        group_mean = group_stats[group]['mean']
        ssw_alt += np.sum((group_data - group_mean) ** 2)

    print(f"Verification: SSW = Σ(x_ij - x̄_j)² = {ssw_alt:.2f} ✓")

    # Step 4: Calculate degrees of freedom
    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    print(f"\nStep 4: Degrees of Freedom")
    print(f"df_between = k - 1 = {k} - 1 = {df_between}")
    print(f"df_within = N - k = {N} - {k} = {df_within}")
    print(f"df_total = N - 1 = {N} - 1 = {df_total}")

    # Step 5: Calculate Mean Squares
    msb = ssb / df_between
    msw = ssw / df_within

    print(f"\nStep 5: Mean Squares")
    print(f"MSB = SSB / df_between = {ssb:.2f} / {df_between} = {msb:.2f}")
    print(f"MSW = SSW / df_within = {ssw:.2f} / {df_within} = {msw:.2f}")

    # Step 6: Calculate F-statistic
    f_statistic = msb / msw
    print(f"\nStep 6: F-statistic")
    print(f"F = MSB / MSW = {msb:.2f} / {msw:.2f} = {f_statistic:.4f}")

    # Step 7: Calculate p-value
    p_value = 1 - stats.f.cdf(f_statistic, df_between, df_within)

    print(f"\nStep 7: P-value")
    print(f"F({df_between}, {df_within}) = {f_statistic:.4f}")
    print(f"p-value = {p_value:.6f}")

    # Step 8: Calculate effect sizes
    eta_squared = ssb / sst
    omega_squared = (ssb - df_between * msw) / (sst + msw)

    print(f"\nStep 8: Effect Sizes")
    print(f"Eta squared (η²) = SSB / SST = {ssb:.2f} / {sst:.2f} = {eta_squared:.4f}")
    print(f"Omega squared (ω²) = (SSB - df_between × MSW) / (SST + MSW)")
    print(f"                  = ({ssb:.2f} - {df_between} × {msw:.2f}) / ({sst:.2f} + {msw:.2f})")
    print(f"                  = {omega_squared:.4f}")

    # Interpret effect sizes
    def interpret_effect_size(eta_sq):
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"

    eta_interpretation = interpret_effect_size(eta_squared)
    omega_interpretation = interpret_effect_size(omega_squared)

    print(f"\nEffect Size Interpretations:")
    print(f"η² = {eta_squared:.4f} ({eta_interpretation} effect)")
    print(f"ω² = {omega_squared:.4f} ({omega_interpretation} effect)")

    # Create ANOVA table
    print(f"\n" + "=" * 70)
    print("ANOVA TABLE")
    print("=" * 70)
    print(f"{'Source':<15} {'SS':<12} {'df':<6} {'MS':<12} {'F':<10} {'p-value':<10}")
    print("-" * 70)
    print(f"{'Between Groups':<15} {ssb:<12.2f} {df_between:<6} {msb:<12.2f} {f_statistic:<10.4f} {p_value:<10.6f}")
    print(f"{'Within Groups':<15} {ssw:<12.2f} {df_within:<6} {msw:<12.2f}")
    print(f"{'Total':<15} {sst:<12.2f} {df_total:<6}")
    print("-" * 70)

    # Conclusion
    alpha = 0.05
    print(f"\nConclusion (α = {alpha}):")
    if p_value < alpha:
        print(f"✓ Reject H₀: There is a statistically significant difference between groups")
        print(f"  F({df_between}, {df_within}) = {f_statistic:.4f}, p = {p_value:.6f}")
        print(f"  Effect size: η² = {eta_squared:.4f} ({eta_interpretation})")
    else:
        print(f"✗ Fail to reject H₀: No statistically significant difference between groups")
        print(f"  F({df_between}, {df_within}) = {f_statistic:.4f}, p = {p_value:.6f}")

    return {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'omega_squared': omega_squared,
        'ssb': ssb,
        'ssw': ssw,
        'sst': sst,
        'msb': msb,
        'msw': msw,
        'df_between': df_between,
        'df_within': df_within
    }

def perform_scipy_anova(data, dependent_var, group_var):
    """
    Perform ANOVA using scipy.stats for comparison with manual calculations.

    """
    print("\n" + "=" * 60)
    print("SCIPY ANOVA (f_oneway)")
    print("=" * 60)

    # Prepare data for scipy
    groups = data[group_var].unique()
    group_data = [data[data[group_var] == group][dependent_var].values for group in groups]

    # Perform ANOVA
    f_stat, p_val = stats.f_oneway(*group_data)

    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {p_val:.6f}")

    alpha = 0.05
    if p_val < alpha:
        print(f"Result: Statistically significant (p < {alpha})")
    else:
        print(f"Result: Not statistically significant (p ≥ {alpha})")

    return f_stat, p_val

def perform_statsmodels_anova(data, dependent_var, group_var):
    """
    Perform ANOVA using statsmodels for more detailed output and model diagnostics.

    """
    print("\n" + "=" * 60)
    print("STATSMODELS ANOVA (OLS + ANOVA)")
    print("=" * 60)

    # Create formula for OLS regression
    formula = f"{dependent_var} ~ C({group_var})"

    # Fit OLS model
    model = ols(formula, data=data).fit()

    # Perform ANOVA
    anova_table = anova_lm(model, typ=2)

    print("ANOVA Table from statsmodels:")
    print(anova_table)

    # Model summary
    print(f"\nModel Summary:")
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"p-value: {model.f_pvalue:.6f}")

    return model, anova_table



def visualize_anova_results(data, dependent_var, group_var, anova_results):
    """
    Create comprehensive visualizations of ANOVA results.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    anova_results : dict
        Results from manual ANOVA calculation
    """
    print("\n" + "=" * 60)
    print("ANOVA RESULTS VISUALIZATION")
    print("=" * 60)

    # Calculate group means and other statistics
    group_stats = data.groupby(group_var)[dependent_var].agg([
        'count', 'mean', 'std', 'sem'
    ]).round(2)

    groups = data[group_var].unique()

    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'ANOVA Results: {dependent_var} by {group_var}\n' +
                f'F({anova_results["df_between"]}, {anova_results["df_within"]}) = ' +
                f'{anova_results["f_statistic"]:.2f}, p = {anova_results["p_value"]:.4f}, ' +
                f'η² = {anova_results["eta_squared"]:.3f}', fontsize=14)

    # 1. Box plot with individual points
    sns.boxplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 0])
    sns.stripplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 0],
                  alpha=0.6, size=3, color='black')
    axes[0, 0].set_title('Box Plot with Individual Points')
    axes[0, 0].set_ylabel(f'{dependent_var}')

    # 2. Mean plot with error bars (95% CI)
    means = group_stats['mean']
    sems = group_stats['sem']
    ns = group_stats['count']

    # Calculate 95% confidence intervals
    t_crit = stats.t.ppf(0.975, ns - 1)  # 95% CI
    ci_95 = t_crit * sems

    x_pos = range(len(means))
    bars = axes[0, 1].bar(x_pos, means, yerr=ci_95, capsize=5, alpha=0.7,
                         color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Group Means with 95% Confidence Intervals')
    axes[0, 1].set_xlabel(group_var)
    axes[0, 1].set_ylabel(f'Mean {dependent_var}')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(means.index)

    # Add value labels on bars
    for i, (mean, ci) in enumerate(zip(means, ci_95)):
        axes[0, 1].text(i, mean + ci + (means.max() * 0.02), f'{mean:.0f}',
                        ha='center', fontweight='bold')

    # 3. Violin plot with quartiles
    sns.violinplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 2], inner='quartile')
    axes[0, 2].set_title('Violin Plot (Distribution Shape)')
    axes[0, 2].set_ylabel(f'{dependent_var}')

    # 4. Residuals vs Fitted plot
    # Calculate fitted values and residuals
    fitted_values = []
    residuals = []

    for group in groups:
        group_data = data[data[group_var] == group][dependent_var]
        group_mean = group_data.mean()
        group_residuals = group_data - group_mean

        fitted_values.extend([group_mean] * len(group_data))
        residuals.extend(group_residuals)

    axes[1, 0].scatter(fitted_values, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residuals vs Fitted Values')
    axes[1, 0].set_xlabel('Fitted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Q-Q plot of residuals
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Effect size visualization
    # Create a pie chart showing explained vs unexplained variance
    explained_var = anova_results['eta_squared']
    unexplained_var = 1 - explained_var

    labels = ['Explained by Groups', 'Unexplained (Error)']
    sizes = [explained_var, unexplained_var]
    colors = ['lightblue', 'lightgray']
    explode = (0.1, 0)  # explode the explained variance slice

    axes[1, 2].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
    axes[1, 2].set_title(f'Variance Explained\n(η² = {explained_var:.3f})')

    plt.tight_layout()
    plt.show()

    # Print detailed group comparisons
    print(f"\nDetailed Group Statistics:")
    print("-" * 35)
    print(group_stats)

    # Calculate pairwise differences between means
    print(f"\nPairwise Mean Differences:")
    print("-" * 30)

    means_dict = group_stats['mean'].to_dict()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Only show each pair once
                diff = means_dict[group1] - means_dict[group2]
                print(f"{group1} - {group2}: {diff:.2f}")

    # Variance decomposition summary
    print(f"\nVariance Decomposition:")
    print("-" * 25)
    print(f"Total Variance (SST): {anova_results['sst']:.2f}")
    print(f"Between-group Variance (SSB): {anova_results['ssb']:.2f} ({anova_results['eta_squared']*100:.1f}%)")
    print(f"Within-group Variance (SSW): {anova_results['ssw']:.2f} ({(1-anova_results['eta_squared'])*100:.1f}%)")

    return group_stats


def perform_anova_power_analysis(data, dependent_var, group_var, anova_results):
    """
    Perform power analysis for the ANOVA results.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    anova_results : dict
        Results from ANOVA analysis
    """
    print("\n" + "=" * 60)
    print("POWER ANALYSIS FOR ANOVA")
    print("=" * 60)

    # Calculate effect size (Cohen's f)
    eta_squared = anova_results['eta_squared']
    cohens_f = np.sqrt(eta_squared / (1 - eta_squared))

    # Get sample sizes
    group_sizes = data.groupby(group_var).size()
    k = len(group_sizes)  # number of groups
    total_n = len(data)

    print(f"Effect Size Calculation:")
    print(f"η² = {eta_squared:.4f}")
    print(f"Cohen's f = √(η² / (1 - η²)) = √({eta_squared:.4f} / {1-eta_squared:.4f}) = {cohens_f:.4f}")

    # Interpret Cohen's f
    if cohens_f < 0.10:
        f_interpretation = "small"
    elif cohens_f < 0.25:
        f_interpretation = "small to medium"
    elif cohens_f < 0.40:
        f_interpretation = "medium to large"
    else:
        f_interpretation = "large"

    print(f"Effect size interpretation: {f_interpretation}")

    # Calculate achieved power
    power_analysis = FTestAnovaPower()
    achieved_power = power_analysis.power(effect_size=cohens_f, nobs=total_n, alpha=0.05, k_groups=k)

    print(f"\nAchieved Power:")
    print(f"Power = {achieved_power:.4f} ({achieved_power*100:.1f}%)")

    if achieved_power >= 0.8:
        print("✓ Adequate power (≥80%) to detect this effect size")
    else:
        print("⚠ Low power (<80%) - may miss true effects of this size")

    # Calculate required sample size for 80% power
    required_n = power_analysis.solve_power(effect_size=cohens_f, power=0.8, alpha=0.05, k_groups=k)
    print(f"Required total sample size for 80% power: {int(np.ceil(required_n))}")
    print(f"Required sample size per group: {int(np.ceil(required_n/k))}")

    # Power curve visualization
    effect_sizes = np.arange(0.1, 1.0, 0.05)
    sample_sizes = np.arange(30, 300, 10)

    # Create power curves for different effect sizes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Power vs Sample Size for different effect sizes
    for f in [0.1, 0.25, 0.4, cohens_f]:
        powers = [power_analysis.power(effect_size=f, nobs=n, alpha=0.05, k_groups=k)
                 for n in sample_sizes]
        label = f'f = {f:.2f}'
        if f == cohens_f:
            label += ' (Our study)'
        ax1.plot(sample_sizes, powers, label=label, linewidth=2)

    ax1.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
    ax1.axvline(x=total_n, color='gray', linestyle=':', label=f'Our sample size (n={total_n})')
    ax1.set_xlabel('Total Sample Size')
    ax1.set_ylabel('Statistical Power')
    ax1.set_title('Power vs Sample Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Power vs Effect Size for different sample sizes
    for n in [60, 120, 240, total_n]:
        powers = [power_analysis.power(effect_size=f, nobs=n, alpha=0.05, k_groups=k)
                 for f in effect_sizes]
        label = f'n = {n}'
        if n == total_n:
            label += ' (Our study)'
        ax2.plot(effect_sizes, powers, label=label, linewidth=2)

    ax2.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
    ax2.axvline(x=cohens_f, color='gray', linestyle=':', label=f'Our effect size (f={cohens_f:.2f})')
    ax2.set_xlabel("Cohen's f (Effect Size)")
    ax2.set_ylabel('Statistical Power')
    ax2.set_title('Power vs Effect Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle('Power Analysis for ANOVA', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Sample size recommendations
    print(f"\nSample Size Recommendations:")
    print("-" * 35)

    effect_scenarios = [
        ("Small effect (f=0.1)", 0.1),
        ("Medium effect (f=0.25)", 0.25),
        ("Large effect (f=0.4)", 0.4),
        ("Observed effect", cohens_f)
    ]

    for scenario, f_val in effect_scenarios:
        req_n = power_analysis.solve_power(effect_size=f_val, power=0.8, alpha=0.05, k_groups=k)
        req_n_per_group = int(np.ceil(req_n / k))
        print(f"{scenario:<20}: {int(np.ceil(req_n)):>3} total ({req_n_per_group:>2} per group)")

    return {
        'cohens_f': cohens_f,
        'achieved_power': achieved_power,
        'required_n': required_n
    }



def perform_tukey_hsd(data, dependent_var, group_var, alpha=0.05):
    """
    Perform Tukey's Honestly Significant Difference test for post-hoc comparisons.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    alpha : float
        Significance level
    """
    print("\n" + "=" * 60)
    print("TUKEY'S HSD POST-HOC TEST")
    print("=" * 60)

    # Perform Tukey's HSD test
    tukey_results = pairwise_tukeyhsd(data[dependent_var], data[group_var], alpha=alpha)

    # Print results
    print("Tukey's HSD Results:")
    print(tukey_results)

    # Create a more detailed summary
    print(f"\nDetailed Pairwise Comparisons (α = {alpha}):")
    print("-" * 50)

    # Extract results for detailed reporting - Updated for newer statsmodels versions
    groups = tukey_results.groupsunique

    # Create results summary - Updated approach
    comparison_results = []

    # Access the summary data differently based on statsmodels version
    try:
        # Try newer format first
        summary_data = tukey_results.summary().data[1:]  # Skip header row

        for row in summary_data:
            group1, group2, meandiff, p_adj, lower, upper, reject = row
            comparison_results.append({
                'Group 1': group1,
                'Group 2': group2,
                'Mean Diff': float(meandiff),
                'p-adj': float(p_adj),
                'Lower CI': float(lower),
                'Upper CI': float(upper),
                'Significant': bool(reject)
            })

            significance = "Yes" if reject else "No"
            print(f"{group1} vs {group2}:")
            print(f"  Mean difference: {float(meandiff):.2f}")
            print(f"  95% CI: [{float(lower):.2f}, {float(upper):.2f}]")
            print(f"  Adjusted p-value: {float(p_adj):.4f}")
            print(f"  Significant: {significance}")
            print()

    except (AttributeError, IndexError, TypeError):
        # Fallback for older versions or different structure
        print("Using alternative method to extract Tukey results...")

        # Manual pairwise comparisons as fallback
        from scipy.stats import ttest_ind
        from statsmodels.stats.multitest import multipletests

        groups_list = list(groups)
        group_pairs = [(groups_list[i], groups_list[j])
                      for i in range(len(groups_list))
                      for j in range(i+1, len(groups_list))]

        p_values = []
        mean_diffs = []

        for group1, group2 in group_pairs:
            data1 = data[data[group_var] == group1][dependent_var]
            data2 = data[data[group_var] == group2][dependent_var]

            _, p_val = ttest_ind(data1, data2)
            mean_diff = data1.mean() - data2.mean()

            p_values.append(p_val)
            mean_diffs.append(mean_diff)

        # Apply Tukey correction (approximation)
        rejected, p_adjusted, _, _ = multipletests(p_values, method='holm')

        for i, (group1, group2) in enumerate(group_pairs):
            data1 = data[data[group_var] == group1][dependent_var]
            data2 = data[data[group_var] == group2][dependent_var]

            # Calculate confidence interval (approximation)
            mean_diff = mean_diffs[i]
            pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
            se = pooled_std * np.sqrt(1/len(data1) + 1/len(data2))
            t_crit = stats.t.ppf(0.975, len(data1) + len(data2) - 2)

            lower_ci = mean_diff - t_crit * se
            upper_ci = mean_diff + t_crit * se

            comparison_results.append({
                'Group 1': group1,
                'Group 2': group2,
                'Mean Diff': mean_diff,
                'p-adj': p_adjusted[i],
                'Lower CI': lower_ci,
                'Upper CI': upper_ci,
                'Significant': rejected[i]
            })

            significance = "Yes" if rejected[i] else "No"
            print(f"{group1} vs {group2}:")
            print(f"  Mean difference: {mean_diff:.2f}")
            print(f"  95% CI: [{lower_ci:.2f}, {upper_ci:.2f}]")
            print(f"  Adjusted p-value: {p_adjusted[i]:.4f}")
            print(f"  Significant: {significance}")
            print()

    # Create visualization of Tukey results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Plot 1: Tukey HSD plot
    try:
        tukey_results.plot_simultaneous(ax=ax1)
        ax1.set_title("Tukey's HSD: Simultaneous Confidence Intervals")
    except:
        # Fallback visualization if plot_simultaneous doesn't work
        group_means = data.groupby(group_var)[dependent_var].mean()
        group_sems = data.groupby(group_var)[dependent_var].sem()

        x_pos = range(len(group_means))
        ax1.bar(x_pos, group_means, yerr=group_sems*1.96, capsize=5, alpha=0.7)
        ax1.set_title("Group Means with 95% CI")
        ax1.set_ylabel(dependent_var)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(group_means.index)

    # Plot 2: Pairwise comparison matrix
    groups_list = list(groups)
    n_groups = len(groups_list)
    comparison_matrix = np.zeros((n_groups, n_groups))

    for result in comparison_results:
        i = groups_list.index(result['Group 1'])
        j = groups_list.index(result['Group 2'])
        comparison_matrix[i, j] = 1 if result['Significant'] else 0
        comparison_matrix[j, i] = 1 if result['Significant'] else 0

    # Create heatmap
    mask = np.triu(np.ones_like(comparison_matrix, dtype=bool), k=1)
    # Convert to integers to avoid float formatting issues
    comparison_matrix = comparison_matrix.astype(int)
    sns.heatmap(comparison_matrix, mask=mask, annot=True, fmt='d',
                xticklabels=groups_list, yticklabels=groups_list,
                cmap='RdBu_r', center=0.5, ax=ax2, cbar_kws={'label': 'Significant Difference'})
    ax2.set_title('Pairwise Significance Matrix')

    plt.tight_layout()
    plt.show()

    return tukey_results, comparison_results

def perform_multiple_comparisons(data, dependent_var, group_var):
    """
    Compare different post-hoc test methods.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    """
    print("\n" + "=" * 60)
    print("COMPARISON OF POST-HOC METHODS")
    print("=" * 60)

    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    # Get all unique groups
    groups = data[group_var].unique()
    group_pairs = list(combinations(groups, 2))

    # Perform pairwise t-tests
    t_stats = []
    p_values = []
    mean_diffs = []

    print("Pairwise t-tests (unadjusted):")
    print("-" * 35)

    for group1, group2 in group_pairs:
        data1 = data[data[group_var] == group1][dependent_var]
        data2 = data[data[group_var] == group2][dependent_var]

        t_stat, p_val = ttest_ind(data1, data2)
        mean_diff = data1.mean() - data2.mean()

        t_stats.append(t_stat)
        p_values.append(p_val)
        mean_diffs.append(mean_diff)

        print(f"{group1} vs {group2}: t = {t_stat:.3f}, p = {p_val:.4f}")

    # Apply different multiple comparison corrections
    corrections = {
        'Bonferroni': 'bonferroni',
        'Holm-Sidak': 'holm',
        'FDR (Benjamini-Hochberg)': 'fdr_bh'
    }

    print(f"\nMultiple Comparisons Corrections:")
    print("-" * 40)

    comparison_summary = pd.DataFrame({
        'Comparison': [f"{pair[0]} vs {pair[1]}" for pair in group_pairs],
        'Mean Diff': mean_diffs,
        'Unadjusted p': p_values
    })

    for method_name, method_code in corrections.items():
        rejected, p_adjusted, _, _ = multipletests(p_values, method=method_code)
        comparison_summary[f'{method_name} p'] = p_adjusted
        comparison_summary[f'{method_name} Sig'] = rejected

    print(comparison_summary.round(4))

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(20, 8))

    x_pos = np.arange(len(group_pairs))
    width = 0.15

    # Plot unadjusted p-values
    ax.bar(x_pos - width*1.5, p_values, width, label='Unadjusted', alpha=0.8)

    # Plot adjusted p-values
    colors = ['orange', 'green', 'red']
    for i, (method_name, method_code) in enumerate(corrections.items()):
        _, p_adjusted, _, _ = multipletests(p_values, method=method_code)
        ax.bar(x_pos - width/2 + i*width, p_adjusted, width,
               label=method_name, alpha=0.8, color=colors[i])

    ax.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    ax.set_xlabel('Pairwise Comparisons')
    ax.set_ylabel('p-value')
    ax.set_title('Comparison of Multiple Testing Corrections')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{pair[0]}\nvs\n{pair[1]}" for pair in group_pairs])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return comparison_summary
    

def perform_kruskal_wallis(data, dependent_var, group_var, manual_results):
    """
    Perform Kruskal-Wallis test as non-parametric alternative to ANOVA.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    """
    print("\n" + "=" * 60)
    print("KRUSKAL-WALLIS TEST (Non-parametric ANOVA)")
    print("=" * 60)

    # Prepare data
    groups = data[group_var].unique()
    group_data = [data[data[group_var] == group][dependent_var].values for group in groups]

    # Perform Kruskal-Wallis test
    h_statistic, p_value = stats.kruskal(*group_data)

    # Calculate group statistics (medians, ranks)
    print("Group Statistics:")
    print("-" * 20)

    all_data = data[dependent_var].values
    all_ranks = stats.rankdata(all_data)

    group_stats = {}
    start_idx = 0

    for i, group in enumerate(groups):
        group_size = len(group_data[i])
        group_ranks = all_ranks[start_idx:start_idx + group_size]

        group_stats[group] = {
            'n': group_size,
            'median': np.median(group_data[i]),
            'mean_rank': np.mean(group_ranks),
            'sum_ranks': np.sum(group_ranks)
        }

        print(f"{group}: n={group_size}, median={np.median(group_data[i]):.1f}, mean rank={np.mean(group_ranks):.1f}")
        start_idx += group_size

    # Calculate effect size (eta squared for Kruskal-Wallis)
    n = len(all_data)
    k = len(groups)
    eta_squared_kw = (h_statistic - k + 1) / (n - k)

    print(f"\nKruskal-Wallis Test Results:")
    print(f"H-statistic: {h_statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Effect size (η²): {eta_squared_kw:.4f}")

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"✓ Significant difference between groups (p < {alpha})")
        print("At least one group has a different distribution")
    else:
        print(f"✗ No significant difference between groups (p ≥ {alpha})")

    # Compare with parametric ANOVA
    print(f"\nComparison with Parametric ANOVA:")
    print("-" * 35)
    print(f"{'Test':<20} {'Statistic':<12} {'p-value':<12} {'Effect Size':<12}")
    print("-" * 60)
    print(f"{'ANOVA (F-test)':<20} {manual_results['f_statistic']:<12.4f} {manual_results['p_value']:<12.6f} {manual_results['eta_squared']:<12.4f}")
    print(f"{'Kruskal-Wallis':<20} {h_statistic:<12.4f} {p_value:<12.6f} {eta_squared_kw:<12.4f}")

    # Determine agreement
    both_significant = (manual_results['p_value'] < alpha) and (p_value < alpha)
    both_nonsignificant = (manual_results['p_value'] >= alpha) and (p_value >= alpha)

    if both_significant or both_nonsignificant:
        print("✓ Both tests reach the same conclusion")
    else:
        print("⚠ Tests disagree - consider assumptions and data characteristics")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Kruskal-Wallis Test Results', fontsize=14)

    # 1. Box plot (same data, different interpretation)
    sns.boxplot(data=data, x=group_var, y=dependent_var, ax=axes[0, 0])
    axes[0, 0].set_title('Box Plot: Focus on Medians')
    axes[0, 0].set_ylabel(dependent_var)

    # Add median values as text
    for i, group in enumerate(groups):
        median_val = group_stats[group]['median']
        axes[0, 0].text(i, median_val, f'{median_val:.0f}',
                       ha='center', va='bottom', fontweight='bold', color='red')

    # 2. Rank sums by group
    group_names = list(group_stats.keys())
    rank_sums = [group_stats[group]['sum_ranks'] for group in group_names]

    axes[0, 1].bar(group_names, rank_sums, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Sum of Ranks by Group')
    axes[0, 1].set_ylabel('Sum of Ranks')

    # Add value labels
    for i, sum_rank in enumerate(rank_sums):
        axes[0, 1].text(i, sum_rank + max(rank_sums)*0.01, f'{sum_rank:.0f}',
                       ha='center', fontweight='bold')

    # 3. Mean ranks comparison
    mean_ranks = [group_stats[group]['mean_rank'] for group in group_names]

    axes[1, 0].bar(group_names, mean_ranks, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Mean Ranks by Group')
    axes[1, 0].set_ylabel('Mean Rank')

    # Add value labels
    for i, mean_rank in enumerate(mean_ranks):
        axes[1, 0].text(i, mean_rank + max(mean_ranks)*0.01, f'{mean_rank:.1f}',
                       ha='center', fontweight='bold')

    # 4. Comparison of test statistics
    test_names = ['ANOVA\n(F-statistic)', 'Kruskal-Wallis\n(H-statistic)']

    # Normalize statistics for comparison (divide by their critical values)
    f_critical = stats.f.ppf(0.95, manual_results['df_between'], manual_results['df_within'])
    h_critical = stats.chi2.ppf(0.95, k-1)  # Kruskal-Wallis uses chi-square distribution

    normalized_stats = [
        manual_results['f_statistic'] / f_critical,
        h_statistic / h_critical
    ]

    colors = ['blue', 'orange']
    bars = axes[1, 1].bar(test_names, normalized_stats, color=colors, alpha=0.7)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Critical threshold')
    axes[1, 1].set_title('Normalized Test Statistics')
    axes[1, 1].set_ylabel('Statistic / Critical Value')
    axes[1, 1].legend()

    # Add value labels
    for i, (bar, stat) in enumerate(zip(bars, normalized_stats)):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, stat + 0.05,
                       f'{stat:.2f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return h_statistic, p_value, eta_squared_kw, group_stats



def create_comprehensive_summary(data, dependent_var, group_var, manual_results, kw_results,assumption_results):
    """
    Create a comprehensive summary of all analyses performed.

    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    dependent_var : str
        Name of the dependent variable
    group_var : str
        Name of the grouping variable
    manual_results : dict
        ANOVA results
    kw_results : tuple
        Kruskal-Wallis results (h, p, eta_squared, group_stats)
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)

    h_stat, kw_p, kw_eta, kw_group_stats = kw_results

    # Dataset summary
    groups = data[group_var].unique()
    group_sizes = data.groupby(group_var).size()

    print(f"Dataset: {len(data)} observations across {len(groups)} groups")
    print(f"Groups: {', '.join(groups)}")
    print(f"Group sizes: {dict(group_sizes)}")

    # Research question
    print(f"\nResearch Question:")
    print(f"Do the {len(groups)} groups differ significantly in {dependent_var}?")

    # Hypotheses
    print(f"\nHypotheses:")
    print(f"H₀: μ₁ = μ₂ = μ₃ (all group means are equal)")
    print(f"H₁: At least one group mean differs from the others")

    # Assumption check summary
    print(f"\nAssumption Check Summary:")
    print(f"✓ Independence: Satisfied (study design)")
    print(f"✓ Normality: {'Satisfied' if assumption_results['shapiro_p'] > 0.05 else 'Questionable'} (Shapiro-Wilk p = {assumption_results['shapiro_p']:.4f})")
    print(f"✓ Homogeneity: {'Satisfied' if assumption_results['levene_p'] > 0.05 else 'Questionable'} (Levene p = {assumption_results['levene_p']:.4f})")
    outlier_count = len(assumption_results['outliers'])
    outlier_text = 'None detected' if outlier_count == 0 else f'{outlier_count} detected'
    print(f"✓ Outliers: {outlier_text}")

    # Main results
    print(f"\nMain Results:")
    print("-" * 15)

    # ANOVA results
    print(f"One-way ANOVA:")
    print(f"  F({manual_results['df_between']}, {manual_results['df_within']}) = {manual_results['f_statistic']:.3f}")
    print(f"  p-value = {manual_results['p_value']:.6f}")
    print(f"  Effect size (η²) = {manual_results['eta_squared']:.3f}")

    # Kruskal-Wallis results
    print(f"\nKruskal-Wallis Test:")
    print(f"  H({len(groups)-1}) = {h_stat:.3f}")
    print(f"  p-value = {kw_p:.6f}")
    print(f"  Effect size (η²) = {kw_eta:.3f}")

    # Conclusion
    alpha = 0.05
    anova_significant = manual_results['p_value'] < alpha
    kw_significant = kw_p < alpha

    print(f"\nConclusion:")
    if anova_significant and kw_significant:
        print(f"✓ Both parametric and non-parametric tests indicate significant differences")
        print(f"  Strong evidence that groups differ in {dependent_var}")
    elif anova_significant or kw_significant:
        print(f"⚡ Tests disagree - one significant, one not")
        print(f"  Consider assumption violations and effect size")
    else:
        print(f"✗ Neither test found significant differences")
        print(f"  No evidence that groups differ in {dependent_var}")

    # Effect size interpretation
    eta_interpretation = "negligible" if manual_results['eta_squared'] < 0.01 else \
                        "small" if manual_results['eta_squared'] < 0.06 else \
                        "medium" if manual_results['eta_squared'] < 0.14 else "large"

    print(f"\nEffect Size Interpretation:")
    print(f"η² = {manual_results['eta_squared']:.3f} represents a {eta_interpretation} effect")
    print(f"Approximately {manual_results['eta_squared']*100:.1f}% of variance in {dependent_var} is explained by group membership")

    # Power analysis summary
    if 'power_results' in globals():
        print(f"\nPower Analysis:")
        print(f"Achieved power: {power_results['achieved_power']:.3f} ({power_results['achieved_power']*100:.1f}%)")
        print(f"Cohen's f: {power_results['cohens_f']:.3f}")
        if power_results['achieved_power'] >= 0.8:
            print(f"✓ Adequate power to detect this effect size")
        else:
            print(f"⚠ Low power - consider larger sample size")

    # Post-hoc summary (if applicable)
    if anova_significant and 'tukey_comparisons' in globals():
        print(f"\nPost-hoc Test Summary (Tukey's HSD):")
        sig_comparisons = [comp for comp in tukey_comparisons if comp['Significant']]

        if sig_comparisons:
            print(f"Significant pairwise differences:")
            for comp in sig_comparisons:
                print(f"  {comp['Group 1']} vs {comp['Group 2']}: diff = {comp['Mean Diff']:.2f}, p = {comp['p-adj']:.4f}")
        else:
            print(f"  No significant pairwise differences found")

    # Create final summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Final Summary: {dependent_var} by {group_var}', fontsize=16)

    # 1. Group means with error bars and significance
    group_means = data.groupby(group_var)[dependent_var].mean()
    group_sems = data.groupby(group_var)[dependent_var].sem()

    bars = axes[0, 0].bar(range(len(group_means)), group_means,
                         yerr=group_sems*1.96, capsize=5, alpha=0.7,
                         color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title(f'Group Means ± 95% CI\nANOVA: F = {manual_results["f_statistic"]:.2f}, p = {manual_results["p_value"]:.4f}')
    axes[0, 0].set_ylabel(f'Mean {dependent_var}')
    axes[0, 0].set_xticks(range(len(group_means)))
    axes[0, 0].set_xticklabels(group_means.index)

    # Add significance indicator
    if anova_significant:
        axes[0, 0].text(0.5, 0.95, '***' if manual_results['p_value'] < 0.001 else
                       '**' if manual_results['p_value'] < 0.01 else '*',
                       transform=axes[0, 0].transAxes, ha='center', fontsize=20)

    # 2. Effect size comparison
    effect_sizes = [manual_results['eta_squared'], kw_eta]
    test_names = ['ANOVA\n(η²)', 'Kruskal-Wallis\n(η²)']

    axes[0, 1].bar(test_names, effect_sizes, color=['blue', 'orange'], alpha=0.7)
    axes[0, 1].set_title('Effect Sizes Comparison')
    axes[0, 1].set_ylabel('Effect Size (η²)')

    # Add interpretation lines
    axes[0, 1].axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='Small (0.01)')
    axes[0, 1].axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, label='Medium (0.06)')
    axes[0, 1].axhline(y=0.14, color='red', linestyle='--', alpha=0.5, label='Large (0.14)')
    axes[0, 1].legend(loc='upper right')

    # 3. Box plot with statistical annotation
    sns.boxplot(data=data, x=group_var, y=dependent_var, ax=axes[1, 0])
    axes[1, 0].set_title('Distribution by Group')

    # 4. Summary statistics table as text
    axes[1, 1].axis('off')

    # Create summary table
    summary_stats = data.groupby(group_var)[dependent_var].agg(['count', 'mean', 'std']).round(2)

    table_text = "Summary Statistics\n" + "="*20 + "\n"
    table_text += f"{'Group':<12} {'n':<5} {'Mean':<8} {'SD':<8}\n"
    table_text += "-"*35 + "\n"

    for group, stats in summary_stats.iterrows():
        table_text += f"{group:<12} {stats['count']:<5.0f} {stats['mean']:<8.1f} {stats['std']:<8.1f}\n"

    table_text += "\nTest Results\n" + "="*15 + "\n"
    table_text += f"ANOVA: F = {manual_results['f_statistic']:.3f}, p = {manual_results['p_value']:.4f}\n"
    table_text += f"Kruskal-Wallis: H = {h_stat:.3f}, p = {kw_p:.4f}\n"
    table_text += f"Effect size: η² = {manual_results['eta_squared']:.3f}\n"

    axes[1, 1].text(0.1, 0.9, table_text, transform=axes[1, 1].transAxes,
                   fontfamily='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()



def generate_report_text(data, dependent_var, group_var, manual_results, assumption_results):
    """
    Generate a properly formatted results section for academic reporting.
    """
    print("\n" + "=" * 80)
    print("SAMPLE RESULTS SECTION FOR ACADEMIC REPORTING")
    print("=" * 80)

    groups = data[group_var].unique()
    group_sizes = data.groupby(group_var).size()
    group_means = data.groupby(group_var)[dependent_var].agg(['mean', 'std'])

    # Sample sizes text
    n_text = ", ".join([f"n = {size}" for size in group_sizes])

    # Format group means and SDs
    means_text = []
    for group in groups:
        mean_val = group_means.loc[group, 'mean']
        std_val = group_means.loc[group, 'std']
        means_text.append(f"{group} (M = {mean_val:.2f}, SD = {std_val:.2f})")

    # Assumption check results
    normality_text = "met" if assumption_results['shapiro_p'] > 0.05 else "violated"
    variance_text = "met" if assumption_results['levene_p'] > 0.05 else "violated"

    # Significance result
    if manual_results['p_value'] < 0.001:
        sig_text = "statistically significant"
        p_text = "p < .001"
    elif manual_results['p_value'] < 0.05:
        sig_text = "statistically significant"
        p_text = f"p = {manual_results['p_value']:.3f}"
    else:
        sig_text = "non-significant"
        p_text = f"p = {manual_results['p_value']:.3f}"

    # Effect size interpretation
    eta_sq = manual_results['eta_squared']
    if eta_sq < 0.01:
        effect_text = "negligible"
    elif eta_sq < 0.06:
        effect_text = "small"
    elif eta_sq < 0.14:
        effect_text = "medium"
    else:
        effect_text = "large"

    # Generate the report - using regular string formatting to avoid f-string issues
    dependent_var_clean = dependent_var.replace('_', ' ')
    groups_text = ', '.join(groups[:-1]) + f", and {groups[-1]}"

    report = f"""
METHOD

A one-way analysis of variance (ANOVA) was conducted to compare {dependent_var_clean}
across three penguin species: {groups_text} ({n_text} respectively).
Prior to analysis, assumptions of ANOVA were evaluated. The assumption of normality was {normality_text}
based on Shapiro-Wilk tests of residuals (p = {assumption_results['shapiro_p']:.3f}). Levene's test
indicated that the assumption of homogeneity of variance was {variance_text},
F({len(groups)-1}, {len(data)-len(groups)}) = {assumption_results.get('levene_stat', 0):.3f},
p = {assumption_results['levene_p']:.3f}.

RESULTS

There was a {sig_text} difference in {dependent_var_clean} between the penguin species,
F({manual_results['df_between']}, {manual_results['df_within']}) = {manual_results['f_statistic']:.3f},
{p_text}, η² = {eta_sq:.3f}, indicating a {effect_text} effect.

Descriptive statistics revealed the following group means: {'; '.join(means_text)}.
"""

    # Add post-hoc results if applicable
    if manual_results['p_value'] < 0.05 and 'tukey_comparisons' in globals():
        report += "\nPost-hoc comparisons using Tukey's HSD test revealed the following significant differences:\n"

        sig_comparisons = [comp for comp in tukey_comparisons if comp['Significant']]
        if sig_comparisons:
            for comp in sig_comparisons:
                report += f"• {comp['Group 1']} vs {comp['Group 2']}: Mean difference = {comp['Mean Diff']:.2f}, "
                report += f"95% CI [{comp['Lower CI']:.2f}, {comp['Upper CI']:.2f}], p = {comp['p-adj']:.3f}\n"
        else:
            report += "No pairwise comparisons reached statistical significance.\n"

    print(report)

    return report



###########################################
from scipy.stats import pearsonr, spearmanr, kendalltau
import statsmodels.api as sm
from statsmodels.stats.correlation_tools import corr_nearest, corr_clipped
import warnings
from itertools import combinations
import requests
from io import StringIO

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

from scipy.stats import shapiro
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
from sklearn.utils import resample
import math

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import FTestAnovaPower
import warnings
from itertools import combinations
import requests
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Set visualization style consistent with previous tutorials
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.power import TTestIndPower, TTestPower
import warnings
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu, wilcoxon, ranksums
from scipy.stats import fisher_exact
from scipy.stats import median_test
from scipy.stats import binomtest  # For sign test

# Suppress warnings
warnings.filterwarnings('ignore')
###########################################

def analyze_dimensionality(data, feature_names, target):
    import numpy as np
    import pandas as pd

    df_temp = pd.DataFrame(data, columns=feature_names)

    n_samples, n_features = df_temp.shape

    print(f"\nDimensionality Analysis:")
    print(f"Features: {n_features}, Samples: {n_samples}")
    print(f"Sample-to-feature ratio: {n_samples/n_features:.2f}")

    # Feature scale analysis
    feature_scales = df_temp.max() - df_temp.min()
    min_scale = feature_scales.min()
    scale_ratio = feature_scales.max() / min_scale if min_scale != 0 else np.inf

    print(f"\nFeature Scale Analysis:")
    print(f"Scale ratio (max/min): {scale_ratio:.2f}")
    if scale_ratio > 100:
        print("Large scale differences → standardization recommended")

    # Correlation analysis
    corr_matrix = df_temp.corr()
    upper_triangle = np.triu(corr_matrix, k=1)
    high_corr_count = np.sum(np.abs(upper_triangle) > 0.8)
    total_pairs = (n_features * (n_features - 1)) // 2

    print(f"\nCorrelation Analysis:")
    print(f"High correlations (|r| > 0.8): {high_corr_count} ({high_corr_count/total_pairs*100:.1f}%)")

    if high_corr_count > total_pairs * 0.1:
        print("High redundancy detected → good candidate for PCA")

    # 🔥 Target relationship (NEW)
    target_corr = df_temp.corrwith(target)

    print(f"\nTop features correlated with target:")
    print(target_corr.abs().sort_values(ascending=False).head(5))

    return corr_matrix, scale_ratio, high_corr_count, target_corr



def visualize_data_structure(df, feature_names):
    """Visualize basic data structure and relationships."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Breast Cancer Dataset: Data Structure', fontsize=14)

    # Feature scales (first 10 features)
    features_subset = list(feature_names[:10])
    means = [df[feat].mean() for feat in features_subset]
    stds = [df[feat].std() for feat in features_subset]

    x_pos = np.arange(len(features_subset))
    axes[0, 0].bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels([name.replace('mean ', '')[:8] for name in features_subset],
                               rotation=45, ha='right')
    axes[0, 0].set_title('Feature Scales (First 10 Features)')
    axes[0, 0].set_ylabel('Value')

    # Correlation heatmap
    corr_subset = df[list(feature_names[:15])].corr()
    sns.heatmap(corr_subset, annot=True, annot_kws={"size": 8}, cmap='RdBu_r', center=0,
                square=True, ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Correlation Matrix (First 15 Features)')

    # Distribution by diagnosis
    axes[1, 0].hist(df[df['p_color'] == 'red']['price'],
                    alpha=0.5, label='red', bins=20, color='red')
    axes[1, 0].hist(df[df['p_color'] == 'green']['price'],
                    alpha=0.7, label='green', bins=20, color='lightgreen')
    axes[1, 0].hist(df[df['p_color'] == 'yellow']['price'],
                    alpha=0.7, label='yellow', bins=20, color='lightyellow')
    axes[1, 0].set_xlabel('price')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Feature Distribution by p_color')
    axes[1, 0].legend()

    # 2D feature relationship
    sns.scatterplot(data=df, x='total_volume', y='price',
                    hue='p_color', alpha=0.7, ax=axes[1, 1])
    axes[1, 1].set_title('2D Feature Relationship')

    plt.tight_layout()
    plt.show()


def find_top_correlations(df, feature_names, n_top=10):
    """Find and display top correlations."""
    corr_matrix = df[list(feature_names)].corr()

    # Extract correlation pairs
    correlation_pairs = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            corr_val = corr_matrix.iloc[i, j]
            correlation_pairs.append({
                'feature1': feature_names[i],
                'feature2': feature_names[j],
                'correlation': corr_val
            })

    # Sort by absolute correlation
    correlation_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    print(f"\nTop {n_top} Correlations:")
    for i, pair in enumerate(correlation_pairs[:n_top], 1):
        feat1 = pair['feature1'].replace('mean ', '').replace('worst ', '')[:12]
        feat2 = pair['feature2'].replace('mean ', '').replace('worst ', '')[:12]
        print(f"{i:2d}. {feat1} ↔ {feat2}: r = {pair['correlation']:6.3f}")

    return correlation_pairs


def standardize_data(X, feature_names):
    """Standardize features for PCA analysis."""

    # Show scale differences before standardization
    print(f"\nBefore standardization:")
    for i in range(len(feature_names)):
        mean_val = np.mean(X[:, i])
        std_val = np.std(X[:, i])
        range_val = np.ptp(X[:, i])
        print(f"{feature_names[i][:20]:<20}: mean={mean_val:>8.2f}, std={std_val:>8.2f}, range={range_val:>8.2f}")

    # Standardize
    scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance.
    X_scaled = scaler.fit_transform(X)

    print(f"\nAfter standardization:")
    for i in range(len(feature_names)):
        mean_val = np.mean(X_scaled[:, i])
        std_val = np.std(X_scaled[:, i])
        print(f"{feature_names[i][:20]:<20}: mean={mean_val:>8.3f}, std={std_val:>8.3f}")

    return X_scaled, scaler


def implement_pca_step_by_step(X_scaled, feature_names):
    """Implement PCA step-by-step to understand the mathematics."""

    print(f"\nPCA Implementation Steps:")
    n_samples, n_features = X_scaled.shape
    print(f"Data dimensions: {n_samples} samples × {n_features} features")

    # Step 1: Data is already centered (standardization sets mean=0)
    print(f"\nStep 1: Data Centering")
    print("Data already centered (standardization sets mean=0)")

    # Step 2: Calculate covariance matrix
    print(f"\nStep 2: Covariance Matrix")
    cov_matrix = np.cov(X_scaled.T)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Matrix is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}")

    # Step 3: Eigendecomposition
    print(f"\nStep 3: Eigendecomposition")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    print(f"Eigenvalues sum: {np.sum(eigenvalues):.2f}")
    print(f"Expected sum (number of features): {n_features}")

    # Step 4: Calculate explained variance
    print(f"\nStep 4: Variance Explained")
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    print("Top 10 Principal Components:")
    print("PC    Eigenvalue   Var Explained   Cumulative")
    print("-" * 45)
    for i in range(min(10, len(eigenvalues))):
        print(f"{i+1:2d}    {eigenvalues[i]:8.3f}      {explained_variance_ratio[i]:6.3f}      {cumulative_variance[i]:6.3f}")

    # Components needed for variance thresholds
    thresholds = [0.8, 0.9, 0.95]
    print(f"\nComponents needed for variance thresholds:")
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        print(f"{threshold*100:2.0f}% variance: {n_components:2d} components")

    # Step 5: Transform data
    print(f"\nStep 5: Data Transformation")
    X_pca = X_scaled @ eigenvectors
    print(f"Original shape: {X_scaled.shape}")
    print(f"Transformed shape: {X_pca.shape}")

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'X_pca': X_pca,
        'cov_matrix': cov_matrix
    }


def analyze_components(pca_results, feature_names, n_components=3):
    """Analyze and interpret principal components."""
    print(f"\nPrincipal Component Analysis:")

    eigenvectors = pca_results['eigenvectors']
    eigenvalues = pca_results['eigenvalues']
    explained_variance_ratio = pca_results['explained_variance_ratio']

    for pc in range(n_components):
        print(f"\nPRINCIPAL COMPONENT {pc + 1}:")
        print(f"Eigenvalue: {eigenvalues[pc]:.3f}")
        print(f"Variance explained: {explained_variance_ratio[pc]:.3f} ({explained_variance_ratio[pc]*100:.1f}%)")

        # Get loadings (feature contributions)
        loadings = eigenvectors[:, pc]   # Loadings are the weights that show how much each original feature contributes to a principal component.

        # Sort features by absolute loading
        loading_pairs = [(feature_names[i], loadings[i]) for i in range(len(feature_names))]
        loading_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"Top contributing features:")
        print("Feature                 Loading")
        print("-" * 35)
        for i, (feature, loading) in enumerate(loading_pairs[:6]):
            direction = "+" if loading > 0 else "-"
            print(f"{feature[:20]:<20} {direction}{abs(loading):6.3f}")


def visualize_pca_results(pca_results, feature_names, y, target_names):
    """Create visualizations of PCA results."""
    print(f"\nCreating PCA visualizations...")

    eigenvalues = pca_results['eigenvalues']
    explained_variance_ratio = pca_results['explained_variance_ratio']
    cumulative_variance = pca_results['cumulative_variance']
    X_pca = pca_results['X_pca']
    eigenvectors = pca_results['eigenvectors']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PCA Analysis Results', fontsize=14)

    # Plot 1: Scree plot
    n_components = min(15, len(eigenvalues))
    axes[0, 0].plot(range(1, n_components + 1), eigenvalues[:n_components], 'bo-', linewidth=2)
    axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser criterion (λ=1)')
    axes[0, 0].set_xlabel('Principal Component')
    axes[0, 0].set_ylabel('Eigenvalue')
    axes[0, 0].set_title('Scree Plot')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot 2: Cumulative explained variance
    axes[0, 1].plot(range(1, n_components + 1), cumulative_variance[:n_components] * 100, 'go-', linewidth=2)
    axes[0, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    axes[0, 1].axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% threshold')
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Variance Explained (%)')
    axes[0, 1].set_title('Cumulative Explained Variance')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot 3: 2D PCA scatter plot
    colors = ['red' if label == 0 else 'green' if label == 1 else 'yellow' for label in y]
    scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6, s=50)
    axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
    axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
    axes[1, 0].set_title('2D PCA Projection')
    axes[1, 0].grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.6, label='red'),
                      Patch(facecolor='green', alpha=0.6, label='green'),
                      Patch(facecolor='lightyellow', alpha=0.6, label='yellow')]
    axes[1, 0].legend(handles=legend_elements)

    # Plot 4: Component loadings heatmap
    n_pcs_heatmap = 5
    loadings_matrix = eigenvectors[:, :n_pcs_heatmap]

    im = axes[1, 1].imshow(loadings_matrix.T, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_yticks(range(n_pcs_heatmap))
    axes[1, 1].set_yticklabels([f'PC{i+1}' for i in range(n_pcs_heatmap)])
    axes[1, 1].set_xticks(range(0, len(feature_names), 5))
    axes[1, 1].set_xticklabels([feature_names[i][:8] for i in range(0, len(feature_names), 5)], rotation=45)
    axes[1, 1].set_title('Component Loadings Heatmap')

    plt.colorbar(im, ax=axes[1, 1], shrink=0.8, label='Loading Value')
    plt.tight_layout()
    plt.show()

    # Print key insights
    pc_80 = np.argmax(cumulative_variance >= 0.8) + 1
    pc_90 = np.argmax(cumulative_variance >= 0.9) + 1
    print(f"\nKey Insights:")
    print(f"- {pc_80} components explain 80% of variance")
    print(f"- {pc_90} components explain 90% of variance")
    print(f"- First 2 components explain {cumulative_variance[1]*100:.1f}% of variance")
    print(f"- Good class separation visible in 2D projection")

    return pc_80, pc_90


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("PCA Tutorial - Part 3: Component Selection and Quality Assessment")
print("-" * 65)

def select_optimal_components(pca_results, variance_thresholds=[0.8, 0.9, 0.95]):
    """Determine optimal number of components using multiple criteria."""
    print("Component Selection Methods:")

    eigenvalues = pca_results['eigenvalues']
    explained_variance_ratio = pca_results['explained_variance_ratio']
    cumulative_variance = pca_results['cumulative_variance']

    recommendations = {}

    # Method 1: Variance threshold
    print("\n1. Variance Threshold Method:")
    for threshold in variance_thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        recommendations[f'{threshold*100:.0f}% variance'] = n_components
        print(f"   {threshold*100:3.0f}% variance: {n_components:2d} components")

    # Method 2: Kaiser criterion (eigenvalues > 1)
    print("\n2. Kaiser Criterion (eigenvalue > 1):")
    kaiser_components = np.sum(eigenvalues > 1)
    recommendations['Kaiser criterion'] = kaiser_components
    print(f"   Components with eigenvalue > 1: {kaiser_components}")

    # Method 3: Scree plot elbow
    print("\n3. Scree Plot Elbow:")
    # Simple elbow detection
    second_differences = np.diff(np.diff(eigenvalues))
    elbow_idx = np.argmax(second_differences) + 2
    recommendations['Scree plot'] = elbow_idx
    print(f"   Elbow point suggests: {elbow_idx} components")

    # Method 4: Interpretability (subjective)
    interpretable_components = 5  # Based on clear loading patterns
    recommendations['Interpretability'] = interpretable_components
    print(f"\n4. Interpretability Assessment: {interpretable_components} components")

    # Summary
    print(f"\nMethod Summary:")
    print("Method               Components   Variance Retained")
    print("-" * 50)
    for method, n_comp in recommendations.items():
        variance = cumulative_variance[n_comp-1] * 100
        print(f"{method:<20} {n_comp:6d}        {variance:6.1f}%")

    # Final recommendation
    recommended = int(np.median(list(recommendations.values())))
    print(f"\nRecommended: {recommended} components")
    print(f"Balances variance retention ({cumulative_variance[recommended-1]*100:.1f}%) with interpretability")

    return {
        'recommendations': recommendations,
        'final_recommendation': recommended,
        'variance_at_recommendation': cumulative_variance[recommended-1]
    }


def create_3d_visualization(pca_results, y, target_names):
    """Create 3D visualization of first three principal components."""
    print("\nCreating 3D PCA visualization...")

    X_pca = pca_results['X_pca']
    explained_variance_ratio = pca_results['explained_variance_ratio']

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Color points by class
    colors = ['red', 'green', 'yellow']
    for i, target in enumerate(np.unique(y)):
        mask = y == target
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                  c=colors[i], label=target_names[target], alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
    ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}% variance)')
    ax.set_title(f'3D PCA Visualization\nFirst 3 Components: {sum(explained_variance_ratio[:3])*100:.1f}% Total Variance')
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"3D Insights:")
    print(f"- First 3 components explain {sum(explained_variance_ratio[:3])*100:.1f}% of variance")
    print(f"- Clear class separation visible in 3D space")
    print(f"- PC1 provides primary separation axis")

    return sum(explained_variance_ratio[:3])


def assess_reconstruction_quality(X_scaled, pca_results, n_components_list=[2, 5, 10]):
    """Assess PCA quality using reconstruction error."""
    print("\nReconstruction Quality Assessment:")

    eigenvectors = pca_results['eigenvectors']

    print("Components  Reconstruction Error  Variance Retained  Info Loss")
    print("-" * 60)

    quality_metrics = {}

    for n_comp in n_components_list:
        # Reconstruct data using n components
        eigenvectors_subset = eigenvectors[:, :n_comp]
        X_projected = X_scaled @ eigenvectors_subset
        X_reconstructed = X_projected @ eigenvectors_subset.T

        # Calculate reconstruction error
        reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
        variance_retained = np.sum(pca_results['explained_variance_ratio'][:n_comp])
        info_loss = 1 - variance_retained

        quality_metrics[n_comp] = {
            'reconstruction_error': reconstruction_error,
            'variance_retained': variance_retained,
            'info_loss': info_loss
        }

        print(f"{n_comp:6d}         {reconstruction_error:15.4f}         {variance_retained:10.3f}     {info_loss:8.3f}")

    # Additional quality metrics
    print(f"\nAdditional Quality Metrics:")
    eigenvalues = pca_results['eigenvalues']

    # Condition number
    condition_number = np.max(eigenvalues) / np.min(eigenvalues[eigenvalues > 1e-10])
    print(f"Condition number: {condition_number:.2e}")

    # Effective dimensionality
    normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)
    effective_dim = np.exp(-np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues + 1e-12)))
    print(f"Effective dimensionality: {effective_dim:.2f}")
    print(f"Reduction ratio: {len(eigenvalues)/effective_dim:.1f}x")

    return quality_metrics


def visualize_quality_metrics(quality_metrics, pca_results):
    """Visualize reconstruction quality and eigenvalue spectrum."""
    print("\nCreating quality assessment plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reconstruction error vs components
    components = list(quality_metrics.keys())
    errors = [quality_metrics[k]['reconstruction_error'] for k in components]
    variances = [quality_metrics[k]['variance_retained'] for k in components]

    ax1.plot(components, errors, 'ro-', linewidth=2, markersize=8, label='Reconstruction Error')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Reconstruction Error', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis for variance
    ax1_twin = ax1.twinx()
    ax1_twin.plot(components, [v*100 for v in variances], 'go--', alpha=0.7, label='Variance Retained (%)')
    ax1_twin.set_ylabel('Variance Retained (%)', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')

    ax1.set_title('Reconstruction Quality vs Components')

    # Plot 2: Eigenvalue spectrum
    eigenvalues = pca_results['eigenvalues']
    ax2.semilogy(range(1, len(eigenvalues[:15]) + 1), eigenvalues[:15], 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser criterion')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Eigenvalue (log scale)')
    ax2.set_title('Eigenvalue Spectrum')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Quality plots show:")
    print(f"- Trade-off between reconstruction error and dimensionality")
    print(f"- Eigenvalue spectrum decay pattern")
    print(f"- Kaiser criterion threshold visualization")



def compare_reconstruction_samples(X_scaled, pca_results, feature_names, sample_indices=[0, 100, 200]):
    """Compare original vs reconstructed data for specific samples."""
    print(f"\nReconstruction Quality for Individual Samples:")

    eigenvectors = pca_results['eigenvectors']
    n_components_test = [2, 5, 10]

    fig, axes = plt.subplots(len(sample_indices), len(n_components_test), figsize=(15, 10))
    fig.suptitle('Sample Reconstruction Quality by Number of Components', fontsize=14)

    reconstruction_errors = {}

    for i, sample_idx in enumerate(sample_indices):
        original_sample = X_scaled[sample_idx]
        reconstruction_errors[sample_idx] = {}

        for j, n_comp in enumerate(n_components_test):
            # Reconstruct sample
            eigenvectors_subset = eigenvectors[:, :n_comp]
            sample_projected = original_sample @ eigenvectors_subset
            sample_reconstructed = sample_projected @ eigenvectors_subset.T

            reconstruction_error = np.mean((original_sample - sample_reconstructed) ** 2)
            reconstruction_errors[sample_idx][n_comp] = reconstruction_error

            if len(sample_indices) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            x_pos = np.arange(len(feature_names))
            ax.plot(x_pos, original_sample, 'b-', alpha=0.7, label='Original', linewidth=2)
            ax.plot(x_pos, sample_reconstructed, 'r--', alpha=0.7, label='Reconstructed', linewidth=2)

            ax.set_title(f'Sample {sample_idx}: {n_comp} Components\nError: {reconstruction_error:.4f}')
            ax.set_ylabel('Standardized Value')
            if i == len(sample_indices) - 1:
                ax.set_xlabel('Feature Index')
            if j == 0:
                ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return reconstruction_errors
