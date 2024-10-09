import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file (adjust to match your file path if needed)
data = np.genfromtxt('Data.csv', delimiter=',', skip_header=1)  # Assuming comma-separated and has a header

# Extract relevant columns from the dataset
inner_rings = data[:, 0]  # Inner Rings (mm)
inner_rings_uncertainty = data[:, 1]  # Uncertainty (mm)
outer_rings = data[:, 2]  # Outer Ring (mm)
outer_rings_uncertainty = data[:, 3]  # Uncertainty (mm)
lambda_ = data[:, 4]  # Lambda (pm)
lambda_uncertainty = data[:, 5]  # Lambda uncertainty (pm)

# Number of data points
N = len(lambda_)

# Calculate sums for slope and intercept for inner rings
sum_x = np.sum(lambda_)
sum_y_inner = np.sum(inner_rings)
sum_xy_inner = np.sum(lambda_ * inner_rings)
sum_x_squared = np.sum(lambda_ ** 2)

# Slope for inner rings
m_inner = (N * sum_xy_inner - sum_x * sum_y_inner) / (N * sum_x_squared - sum_x ** 2)

# Intercept for inner rings
b_inner = (sum_y_inner - m_inner * sum_x) / N

# Residuals for inner rings
residuals_inner = inner_rings - (m_inner * lambda_ + b_inner)

# Variance of y for inner rings
variance_inner = np.sum(residuals_inner ** 2) / (N - 2)

# Calculate Delta for uncertainties in inner rings
Delta_inner = N * sum_x_squared - sum_x ** 2

# Uncertainty in the slope for inner rings
slope_uncertainty_inner = np.sqrt(variance_inner / Delta_inner)

# Uncertainty in the intercept for inner rings
intercept_uncertainty_inner = np.sqrt(variance_inner * sum_x_squared / ((N - 2) * Delta_inner))

# Perform the same calculation for outer rings
sum_y_outer = np.sum(outer_rings)
sum_xy_outer = np.sum(lambda_ * outer_rings)

# Slope for outer rings
m_outer = (N * sum_xy_outer - sum_x * sum_y_outer) / (N * sum_x_squared - sum_x ** 2)

# Intercept for outer rings
b_outer = (sum_y_outer - m_outer * sum_x) / N

# Residuals for outer rings
residuals_outer = outer_rings - (m_outer * lambda_ + b_outer)

# Variance of y for outer rings
variance_outer = np.sum(residuals_outer ** 2) / (N - 2)

# Calculate Delta for uncertainties in outer rings
Delta_outer = N * sum_x_squared - sum_x ** 2

# Uncertainty in the slope for outer rings
slope_uncertainty_outer = np.sqrt(variance_outer / Delta_outer)

# Uncertainty in the intercept for outer rings
intercept_uncertainty_outer = np.sqrt(variance_outer * sum_x_squared / ((N - 2) * Delta_outer))

# -- R^2 CALCULATIONS --

# R^2 for inner rings
ss_res_inner = np.sum(residuals_inner ** 2)
ss_tot_inner = np.sum((inner_rings - np.mean(inner_rings)) ** 2)
r_squared_inner = 1 - (ss_res_inner / ss_tot_inner)

# R^2 for outer rings
ss_res_outer = np.sum(residuals_outer ** 2)
ss_tot_outer = np.sum((outer_rings - np.mean(outer_rings)) ** 2)
r_squared_outer = 1 - (ss_res_outer / ss_tot_outer)

# -- CHI-SQUARED CALCULATIONS --

# Chi-squared for inner rings
chi_squared_inner = np.sum(((inner_rings - (m_inner * lambda_ + b_inner)) / inner_rings_uncertainty) ** 2)
reduced_chi_squared_inner = chi_squared_inner / (N - 2)  # 2 parameters: slope and intercept

# Chi-squared for outer rings
chi_squared_outer = np.sum(((outer_rings - (m_outer * lambda_ + b_outer)) / outer_rings_uncertainty) ** 2)
reduced_chi_squared_outer = chi_squared_outer / (N - 2)

# Plot the data with error bars and the fitted lines
plt.figure(figsize=(8, 6))

# Plot for inner rings
plt.errorbar(
    lambda_, inner_rings,
    xerr=lambda_uncertainty,
    yerr=inner_rings_uncertainty,
    fmt='o',
    ecolor='black',
    capsize=3,
    label='Inner rings with error bars'
)
plt.plot(lambda_, m_inner * lambda_ + b_inner, 'r-', label=f'Inner fit: y = ({m_inner:.2f} ± {slope_uncertainty_inner:.2f})x + ({b_inner:.2f} ± {intercept_uncertainty_inner:.2f})')

# Plot for outer rings
plt.errorbar(
    lambda_, outer_rings,
    xerr=lambda_uncertainty,
    yerr=outer_rings_uncertainty,
    fmt='o',
    ecolor='blue',
    capsize=3,
    label='Outer rings with error bars'
)
plt.plot(lambda_, m_outer * lambda_ + b_outer, 'b-', label=f'Outer fit: y = ({m_outer:.2f} ± {slope_uncertainty_outer:.2f})x + ({b_outer:.2f} ± {intercept_uncertainty_outer:.2f})')

# Labels and legend
plt.xlabel('Lambda (pm)')
plt.ylabel('r (mm)')
plt.title('Linear Fit for Inner and Outer Rings as a Function of Lambda')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()

# Print the fit results for inner rings
print(f"Inner Rings Fit:")
print(f"Slope: {m_inner:.4f} ± {slope_uncertainty_inner:.4f} mm/pm")
print(f"Intercept: {b_inner:.4f} ± {intercept_uncertainty_inner:.4f} mm")
print(f"R^2: {r_squared_inner:.4f}")
print(f"Chi-Squared: {chi_squared_inner:.4f}")
print(f"Reduced Chi-Squared: {reduced_chi_squared_inner:.4f}")

# Print the fit results for outer rings
print(f"Outer Rings Fit:")
print(f"Slope: {m_outer:.4f} ± {slope_uncertainty_outer:.4f} mm/pm")
print(f"Intercept: {b_outer:.4f} ± {intercept_uncertainty_outer:.4f} mm")
print(f"R^2: {r_squared_outer:.4f}")
print(f"Chi-Squared: {chi_squared_outer:.4f}")
print(f"Reduced Chi-Squared: {reduced_chi_squared_outer:.4f}")
