
from dtuimldmtools.statistics.statistics import correlated_ttest
import re

with open("./RegressionPartB\data", "r") as f:
    data = f.read()

    # Find all numbers located after Etest_* in the data file
    etests = re.findall(r'Etest_\d+\s+(\[?[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?\]?)', data)

    # Remove brackets if they exist around numbers
    etests = [float(value.strip('[]')) for value in etests]

ANN,linreg,baseline,error_diff_baseline_linreg ,error_diff_baseline_ann,error_diff_linreg_ann = [],[],[],[],[],[] # instantiate arrays

# split all found numbers into seperate arrays for the model they represent
for index, value in enumerate(etests):
    if index%3 == 0:
        ANN.append(value)
    if index%3 == 1:
        linreg.append(value)
    if index%3 == 2:
        baseline.append(value)

# print(ANN,linreg,baseline)

# Alpha and rho values as described in setup II in the book
alpha = 0.05
rho = 1 / len(linreg)

#Calculate error differences for each pair comparison
for i in range(len(linreg)):
    error_diff_baseline_linreg.append((baseline[i]-linreg[i]))
    error_diff_baseline_ann.append((baseline[i]-ANN[i]))
    error_diff_linreg_ann.append((linreg[i]-ANN[i]))

# Perform correlated t-test for each pairwise comparison
p_baseline_linreg, CI_baseline_linreg = correlated_ttest(error_diff_baseline_linreg, rho, alpha=alpha)
p_baseline_ann, CI_baseline_ann = correlated_ttest(error_diff_baseline_ann, rho, alpha=alpha)
p_linreg_ann, CI_linreg_ann = correlated_ttest(error_diff_linreg_ann, rho, alpha=alpha)

# Print the results for each comparison
print("\nCorrelated t-test results:")
print(f"Baseline vs Linear Regression: p-value = {p_baseline_linreg:.4f}, CI = {CI_baseline_linreg}")
print(f"Baseline vs ANN: p-value = {p_baseline_ann:.4f}, CI = {CI_baseline_ann}")
print(f"Linear Regression vs ANN: p-value = {p_linreg_ann:.4f}, CI = {CI_linreg_ann}")


