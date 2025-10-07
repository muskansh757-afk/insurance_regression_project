# -------------------------------------------------------------
# Project: Insurance Charges Prediction using Linear Regression
# Author: Muskan Sharma
# -------------------------------------------------------------
# Goal:
# Understand how factors like age, BMI, smoker status, and region
# influence medical insurance charges using regression analysis.
# -------------------------------------------------------------

# 1️⃣ Load the dataset
# Choose your file or use the path directly
df <- read.csv(file.choose(), stringsAsFactors = FALSE)

# Quick look at data
head(df)
summary(df)

# 2️⃣ Convert categorical variables to factors
df$sex <- as.factor(df$sex)
df$smoker <- as.factor(df$smoker)
df$region <- as.factor(df$region)

# 3️⃣ Basic Exploratory Plots (to understand relationships)
# Save plots to 'outputs' folder
# Create folder if not existing
if (!dir.exists("outputs")) dir.create("outputs")

# Plot: Age vs Charges
png("outputs/scatter_age_charges.png", width=700, height=500)
plot(df$age, df$charges, pch=16, col="blue",
     xlab="Age", ylab="Charges (USD)",
     main="Age vs Insurance Charges")
grid()
dev.off()
getwd()
# Plot: Charges by Smoker Status
png("outputs/box_smoker_charges.png", width=700, height=500)
boxplot(df$charges ~ df$smoker, data=df, main="Charges by Smoker Status",
        ylab="Charges (USD)", col=c("lightgreen","pink"))
dev.off()

# 4️⃣ Log-transform the charges (because charges are skewed)
df$log_charges <- log(df$charges)

png("outputs/hist_log_charges.png", width=700, height=500)
hist(df$log_charges, breaks=25, col="skyblue",
     main="Distribution of log(Charges)",
     xlab="log(Charges)")
dev.off()

# 5️⃣ Split data into training (70%) and testing (30%)
set.seed(123)  # reproducibility
n <- nrow(df)
train_idx <- sample(seq_len(n), size = floor(0.7*n))
train <- df[train_idx, ]
test <- df[-train_idx, ]

# 6️⃣ Build regression models
# Model 1: Charges directly
model1 <- lm(charges ~ age + sex + bmi + children + smoker + region, data=train)
summary(model1)

# Model 2: On log(charges)
model2 <- lm(log_charges ~ age + sex + bmi + children + smoker + region, data=train)
summary(model2)

# 7️⃣ Diagnostic plots for model2 (to check assumptions)
png("outputs/diagnostic_plots_model2.png", width=900, height=700)
par(mfrow=c(2,2))
plot(model2)
par(mfrow=c(1,1))
dev.off()

# 8️⃣ Predict on test data
test$pred_charges <- predict(model1, newdata=test)
test$pred_log <- predict(model2, newdata=test)
test$pred_charges_from_log <- exp(test$pred_log)  # back-transform to original scale

# 9️⃣ Evaluate model performance using RMSE and R²
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))
rsq <- function(actual, pred) 1 - sum((actual - pred)^2)/sum((actual - mean(actual))^2)

rmse_orig <- rmse(test$charges, test$pred_charges)
rmse_log <- rmse(test$charges, test$pred_charges_from_log)
r2_orig <- rsq(test$charges, test$pred_charges)
r2_log <- rsq(test$charges, test$pred_charges_from_log)

cat("Model Performance:\n")
cat("------------------\n")
cat("RMSE (Original):", round(rmse_orig,2), "\n")
cat("RMSE (Log model back-transformed):", round(rmse_log,2), "\n")
cat("R² (Original):", round(r2_orig,3), "\n")
cat("R² (Log model back-transformed):", round(r2_log,3), "\n")

# 10️⃣ Visualize predicted vs actual charges
png("outputs/pred_vs_actual.png", width=700, height=500)
plot(test$charges, test$pred_charges_from_log, pch=16, col="darkgreen",
     xlab="Actual Charges", ylab="Predicted Charges (log model)",
     main="Actual vs Predicted Charges")
abline(a=0,b=1,col="red",lwd=2)
grid()
dev.off()

# 11️⃣ Save summary output to text file (optional)
sink("outputs/model_summary.txt")
summary(model2)
sink()

# -------------------------------------------------------------
# End of script
# -------------------------------------------------------------

