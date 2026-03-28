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



