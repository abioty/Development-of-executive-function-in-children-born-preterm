library(MASS) # For robust regression

# Seed for reproducibility
set.seed(123)

# Set the working directory to where the data file is located
setwd("~/Desktop/fc-sc-data/NMF-Y")

# Load Data
data <- read.csv('all-sc-features_confo.csv')

# Check for any non-positive values in 'y'
if(any(data$y <= 0)) {
  warning("Non-positive values detected in 'y', adding a constant before transformation.")
  data$y <- data$y + abs(min(data$y[data$y <= 0], na.rm = TRUE)) + 0.01
}

# Preprocessing
continuous_confounders <- scale(data[, 2:4])
categorical_confounders <- data[, 5:8]
final_confounders <- cbind(continuous_confounders, categorical_confounders)

selected_vars <- scale(data[, c("Comp_sc_asp7", "Comp_sc_asp10", "Comp_sc_asp23", "Comp_sc_bc20", "Comp_sc_cc16", "Comp_sc_cc22", "Comp_sc_le6",
                                "Comp_sc_cc23", "Comp_sc_co6", "Comp_sc_co28", "Comp_sc_rd3", "Comp_sc_rd17")])

final_vars <- cbind(selected_vars, final_confounders)
final_data <- cbind(y = data$y, final_vars)

model_formula <- reformulate(termlabels = names(final_vars), response = "y")
model_robust <- rlm(model_formula, data = final_data, method="MM")

summary_model_robust <- summary(model_robust)
coefs <- summary_model_robust$coefficients

conf_intervals <- cbind(coefs[, "Value"] - qnorm(0.975) * coefs[, "Std. Error"],
                        coefs[, "Value"] + qnorm(0.975) * coefs[, "Std. Error"])
colnames(conf_intervals) <- c("Lower 95% CI", "Upper 95% CI")

# Calculate t-values and p-values
t_values <- coefs[, "Value"] / coefs[, "Std. Error"]
p_values <- 2 * pt(abs(t_values), df = summary_model_robust$df[2], lower.tail = FALSE)

coefs <- cbind(coefs, conf_intervals, "t-value" = t_values, "p-value" = p_values)

print(coefs)

summary_model_robust$coefficients <- coefs
print(summary_model_robust)

#Export
coefs_df <- data.frame(Variable = rownames(coefs), coefs, row.names = NULL)
write.csv(coefs_df, file = "~/Desktop/fc-sc-data/NMF-Y/model_robust_summary(re_sc).csv", row.names = FALSE)

predicted_values <- predict(model_robust, final_data)
residuals <- final_data$y - predicted_values

plot(predicted_values, residuals, xlab = "Predicted Values", ylab = "Residuals",
     main = "Residuals vs Predicted Values", pch = 20)
abline(h = 0, col = "red", lty = 2)

title(main = "Residu vs Predi Values in Robust RM",
      sub = "Data from 'all-sc-features_confo.csv'")
