# Loan Default Risk Analysis
# Dataset: Mortgage Loan Default Dataset (Kaggle - M Yasser H)
# 148,670 loan applications from 2019
# Goal: predict which borrowers will default using 4 ML models
# Models: Logistic Regression, Decision Tree, Random Forest, XGBoost

#libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(pROC)
library(randomForest)
library(xgboost)
library(PRROC)
library(smotefamily)



# load data
loan_df <- read.csv("Loan_Default.csv", stringsAsFactors = FALSE)
# quick look at the data
nrow(loan_df)
ncol(loan_df)
head(loan_df)
str(loan_df)

# check class distribution - how many defaulted vs repaid
table(df$Status)
prop.table(table(df$Status))

colSums(is.na(df))


#Data Cleaning
#unique values check
sapply(df, function(x) length(unique(x)))

#drop useless columns
# ID - just a row number, no predictive value
# year - all 2019, zero variance
# rate_of_interest, Interest_rate_spread - missing for almost all defaulters, data leakage
# Upfront_charges - too many missing values
df <- df %>% select(-ID, -year, -rate_of_interest, -Interest_rate_spread, -Upfront_charges)

ncol(df)


#region inconsistency
#north/North treated as different categories
table(df$Region)

df$Region <- tolower(df$Region)
table(df$Region)

#handle missing values
#numerical with median
df$property_value <- ifelse(is.na(df$property_value), median(df$property_value, na.rm=TRUE), df$property_value)
df$LTV <- ifelse(is.na(df$LTV), median(df$LTV, na.rm=TRUE), df$LTV)
df$dtir1 <- ifelse(is.na(df$dtir1), median(df$dtir1, na.rm=TRUE), df$dtir1)
df$income <- ifelse(is.na(df$income), median(df$income, na.rm=TRUE), df$income)

#catagorical with mode
df$term <- ifelse(is.na(df$term), as.numeric(names(sort(table(df$term), decreasing=TRUE)[1])), df$term)
df$submission_of_application <- ifelse(is.na(df$submission_of_application), names(sort(table(df$submission_of_application), decreasing=TRUE)[1], df$submission_of_application)

#cheak any missing value
colSums(is.na(df))

sapply(df, function(x) sum(x == "" | is.na(x)))

table(df$loan_limit)
table(df$approv_in_adv)
table(df$Neg_ammortization)
table(df$loan_purpose)
table(df$age)
colSums(df == "" | is.na(df), na.rm = TRUE)


#input mode for catagorical columns
# function to get mode
get_mode <- function(v) {
  uniqv <- unique(v[!is.na(v) & v != ""])
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

df$loan_limit <- ifelse(is.na(df$loan_limit) | df$loan_limit == "", get_mode(df$loan_limit), df$loan_limit)
df$approv_in_adv <- ifelse(is.na(df$approv_in_adv) | df$approv_in_adv == "", get_mode(df$approv_in_adv), df$approv_in_adv)
df$loan_purpose <- ifelse(is.na(df$loan_purpose) | df$loan_purpose == "", get_mode(df$loan_purpose), df$loan_purpose)
df$Neg_ammortization <- ifelse(is.na(df$Neg_ammortization) | df$Neg_ammortization == "", get_mode(df$Neg_ammortization), df$Neg_ammortization)
df$age <- ifelse(is.na(df$age) | df$age == "", get_mode(df$age), df$age)
df$submission_of_application <- ifelse(is.na(df$submission_of_application) | df$submission_of_application == "", get_mode(df$submission_of_application), df$submission_of_application)
#check agian 
colSums(df == "" | is.na(df), na.rm = TRUE)


#final verification
dim(df)
colSums(is.na(df))

#EDA
#skewness check
# if mean >> median, column is right skewed and needs log transformation
numeric_cols <- df %>% select(where(is.numeric)) %>% names()
print(numeric_cols)
numeric_cols_check <- c("loan_amount", "term", "property_value", "income", "Credit_Score", "LTV", "dtir1")

for(col in numeric_cols_check) {
  cat(col, "-- mean:", round(mean(df[[col]]), 2), 
      "| median:", round(median(df[[col]]), 2), "\n")
}


#Plots
#1. Class Distribution
class_data <- df %>%
  group_by(Status) %>%
  summarise(Count = n()) %>%
  mutate(
    Percentage = round(Count / sum(Count) * 100, 2),
    Label = paste0(Count, "\n(", Percentage, "%)")
  )

ggplot(class_data, aes(x = factor(Status), y = Count, fill = factor(Status))) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Label), vjust = -0.3, size = 4.5) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("Repaid", "Defaulted")) +
  scale_x_discrete(labels = c("0" = "Repaid", "1" = "Defaulted")) +
  labs(title = "Loan Status Distribution",
       x = "Status", y = "Count", fill = "Status") +
  theme_minimal()

#credit score to default rate
region_data <- df %>%
  group_by(Region) %>%
  summarise(
    Total = n(),
    Defaults = sum(Status),
    Default_Rate = round(mean(Status) * 100, 2)
  )

print(region_data)

ggplot(region_data, aes(x = reorder(Region, -Default_Rate), y = Default_Rate, fill = Region)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Default_Rate, "%")), vjust = -0.4, size = 4) +
  labs(title = "Default Rate by Region",
       x = "Region", y = "Default Rate (%)") +
  theme_minimal()

#by age
age_data <- df %>%
  group_by(age) %>%
  summarise(
    Total = n(),
    Defaults = sum(Status),
    Default_Rate = round(mean(Status) * 100, 2)
  )

print(age_data)

ggplot(age_data, aes(x = reorder(age, Default_Rate), y = Default_Rate, fill = Default_Rate)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Default_Rate, "%")), hjust = -0.2, size = 4) +
  coord_flip() +
  scale_fill_gradient(low = "#2ecc71", high = "#e74c3c") +
  labs(title = "Default Rate by Age Group",
       x = "Age Group", y = "Default Rate (%)") +
  theme_minimal()

#by loan type
loan_type_data <- df %>%
  group_by(loan_type) %>%
  summarise(
    Total = n(),
    Defaults = sum(Status),
    Default_Rate = round(mean(Status) * 100, 2)
  )

print(loan_type_data)

ggplot(loan_type_data, aes(x = loan_type, y = Default_Rate, fill = loan_type)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = paste0(Default_Rate, "%")), vjust = -0.4, size = 4) +
  labs(title = "Default Rate by Loan Type",
       x = "Loan Type", y = "Default Rate (%)") +
  theme_minimal()
#by credit score
ggplot(df, aes(x = factor(Status), y = Credit_Score, fill = factor(Status))) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("Repaid", "Defaulted")) +
  scale_x_discrete(labels = c("0" = "Repaid", "1" = "Defaulted")) +
  labs(title = "Credit Score by Loan Status",
       subtitle = "Surprisingly similar distribution across both classes",
       x = "Loan Status", y = "Credit Score", fill = "Status") +
  theme_minimal(base_size = 13)

#by amount distribution
ggplot(df, aes(x = loan_amount, fill = factor(Status))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("Repaid", "Defaulted")) +
  labs(title = "Loan Amount Distribution by Status",
       subtitle = "After log transformation",
       x = "Loan Amount (log scale)", y = "Density", fill = "Status") +
  theme_minimal(base_size = 13)

#by DTI Ratio
ggplot(df, aes(x = factor(Status), y = dtir1, fill = factor(Status))) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "#2ecc71", "1" = "#e74c3c"),
                    labels = c("Repaid", "Defaulted")) +
  scale_x_discrete(labels = c("0" = "Repaid", "1" = "Defaulted")) +
  labs(title = "Debt-to-Income Ratio by Loan Status",
       x = "Loan Status", y = "DTI Ratio", fill = "Status") +
  theme_minimal(base_size = 13)



#feature engineering
#log transformation
# loan_amount, income, property_value all had mean >> median
# log1p used instead of log because log(0) = -inf, log1p(0) = 0
df$loan_amount <- log1p(df$loan_amount)
df$income <- log1p(df$income)
df$property_value <- log1p(df$property_value)

#verification
cat("loan_amount -- mean:", round(mean(df$loan_amount), 2), "| median:", round(median(df$loan_amount), 2), "\n")
cat("income -- mean:", round(mean(df$income), 2), "| median:", round(median(df$income), 2), "\n")
cat("property_value -- mean:", round(mean(df$property_value), 2), "| median:", round(median(df$property_value), 2), "\n")



#one hot encoding
# models cant handle text, need to convert to 0/1 binary columns
cat_cols <- df %>% select(where(is.character)) %>% names()
print(cat_cols)


df$Status <- as.factor(df$Status)

df_encoded <- df %>%
  mutate(across(all_of(cat_cols), as.factor))

dummy_model <- dummyVars(Status ~ ., data = df_encoded, fullRank = TRUE)

df_final <- predict(dummy_model, newdata = df_encoded)
df_final <- as.data.frame(df_final)
df_final$Status <- df$Status

cat("Columns after encoding:", ncol(df_final), "\n")
cat("Rows:", nrow(df_final), "\n")

head(df_final[, 1:10])

#Train/Test split
# 80% train, 20% test
# createDataPartition does stratified split so it maintains 75/25 class ratio in both sets
set.seed(42)
split_index <- createDataPartition(df_final$Status, p = 0.80, list = FALSE)

train <- df_final[split_index, ]
test  <- df_final[-split_index, ]

cat("Train rows:", nrow(train), "\n")
cat("Test rows:", nrow(test), "\n")

#final class distribution check
prop.table(table(train$Status)) * 100
prop.table(table(test$Status)) * 100

# fix column names for randomForest compatibility
names(train) <- make.names(names(train))
names(test) <- make.names(names(test))



#ML Models
#LR Model
train$Status <- as.factor(train$Status)
test$Status <- as.factor(test$Status)

model_lr <- glm(Status ~ ., data = train, 
                family = binomial(),
                control = glm.control(maxit = 500))
lr_probs <- predict(model_lr, newdata = test, type = "response")
lr_preds <- factor(ifelse(lr_probs >= 0.5, "1", "0"), levels = c("0", "1"))

lr_cm <- confusionMatrix(lr_preds, test$Status, positive = "1")
print(lr_cm)

lr_accuracy    <- round(lr_cm$overall["Accuracy"] * 100, 2)
lr_recall      <- round(lr_cm$byClass["Sensitivity"] * 100, 2)
lr_precision   <- round(lr_cm$byClass["Pos Pred Value"] * 100, 2)
lr_f1          <- round(lr_cm$byClass["F1"] * 100, 2)
lr_specificity <- round(lr_cm$byClass["Specificity"] * 100, 2)
lr_roc <- roc(as.numeric(as.character(test$Status)), lr_probs, quiet = TRUE)
lr_auc <- round(auc(lr_roc), 4)

cat("LR -- Accuracy:", lr_accuracy, "| Recall:", lr_recall, 
    "| Precision:", lr_precision, "| F1:", lr_f1, 
    "| Specificity:", lr_specificity, "| AUC:", lr_auc, "\n")

#DT Model
model_dt <- rpart(Status ~ ., data = train, method = "class")
dt_probs <- predict(model_dt, newdata = test, type = "prob")[, "1"]
dt_preds <- predict(model_dt, newdata = test, type = "class")

dt_cm <- confusionMatrix(dt_preds, test$Status, positive = "1")
print(dt_cm)


dt_roc <- roc(as.numeric(as.character(test$Status)), dt_probs, quiet = TRUE)
dt_auc <- round(auc(dt_roc), 4)
cat("DT ROC-AUC:", dt_auc, "\n")

dt_accuracy    <- round(dt_cm$overall["Accuracy"] * 100, 2)
dt_recall      <- round(dt_cm$byClass["Sensitivity"] * 100, 2)
dt_precision   <- round(dt_cm$byClass["Pos Pred Value"] * 100, 2)
dt_f1          <- round(dt_cm$byClass["F1"] * 100, 2)
dt_specificity <- round(dt_cm$byClass["Specificity"] * 100, 2)

cat("DT -- Accuracy:", dt_accuracy, "| Recall:", dt_recall, 
    "| Precision:", dt_precision, "| F1:", dt_f1, 
    "| Specificity:", dt_specificity, "| AUC:", dt_auc, "\n")

#RF Model
names(train) <- make.names(names(train))
names(test) <- make.names(names(test))
set.seed(42)
model_rf <- randomForest(Status ~ ., data = train, ntree = 100, importance = TRUE)

rf_probs <- predict(model_rf, newdata = test, type = "prob")[, "1"]
rf_preds <- predict(model_rf, newdata = test, type = "response")

rf_cm <- confusionMatrix(rf_preds, test$Status, positive = "1")
print(rf_cm)

rf_roc <- roc(as.numeric(as.character(test$Status)), rf_probs, quiet = TRUE)
rf_auc <- round(auc(rf_roc), 4)
cat("RF ROC-AUC:", rf_auc, "\n")

rf_accuracy    <- round(rf_cm$overall["Accuracy"] * 100, 2)
rf_recall      <- round(rf_cm$byClass["Sensitivity"] * 100, 2)
rf_precision   <- round(rf_cm$byClass["Pos Pred Value"] * 100, 2)
rf_f1          <- round(rf_cm$byClass["F1"] * 100, 2)
rf_specificity <- round(rf_cm$byClass["Specificity"] * 100, 2)

cat("RF -- Accuracy:", rf_accuracy, "| Recall:", rf_recall, 
    "| Precision:", rf_precision, "| F1:", rf_f1, 
    "| Specificity:", rf_specificity, "| AUC:", rf_auc, "\n")

#XGBoost
train_matrix <- xgb.DMatrix(
  data  = as.matrix(train %>% select(-Status)),
  label = as.numeric(as.character(train$Status))
)

test_matrix <- xgb.DMatrix(
  data  = as.matrix(test %>% select(-Status)),
  label = as.numeric(as.character(test$Status))
)


set.seed(42)
model_xgb <- xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  ),
  data    = train_matrix,
  nrounds = 200,
  verbose = 0
)
xgb_probs <- predict(model_xgb, test_matrix)
xgb_preds <- factor(ifelse(xgb_probs >= 0.5, "1", "0"), levels = c("0", "1"))

xgb_cm <- confusionMatrix(xgb_preds, test$Status, positive = "1")
print(xgb_cm)

xgb_roc <- roc(as.numeric(as.character(test$Status)), xgb_probs, quiet = TRUE)
xgb_auc <- round(auc(xgb_roc), 4)
cat("XGBoost ROC-AUC:", xgb_auc, "\n")

xgb_accuracy    <- round(xgb_cm$overall["Accuracy"] * 100, 2)
xgb_recall      <- round(xgb_cm$byClass["Sensitivity"] * 100, 2)
xgb_precision   <- round(xgb_cm$byClass["Pos Pred Value"] * 100, 2)
xgb_f1          <- round(xgb_cm$byClass["F1"] * 100, 2)
xgb_specificity <- round(xgb_cm$byClass["Specificity"] * 100, 2)

cat("XGB -- Accuracy:", xgb_accuracy, "| Recall:", xgb_recall, 
    "| Precision:", xgb_precision, "| F1:", xgb_f1, 
    "| Specificity:", xgb_specificity, "| AUC:", xgb_auc, "\n")

#XGBoost Confusion Matrix heatmap
xgb_cm_df <- as.data.frame(xgb_cm$table)
names(xgb_cm_df) <- c("Predicted", "Actual", "Count")

xgb_cm_df$Predicted <- factor(xgb_cm_df$Predicted, 
                              levels = c("1", "0"),
                              labels = c("Default", "No Default"))
xgb_cm_df$Actual <- factor(xgb_cm_df$Actual,
                           levels = c("0", "1"),
                           labels = c("No Default", "Default"))

ggplot(xgb_cm_df, aes(x = Predicted, y = Actual, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = Count), size = 10, fontface = "bold", color = "white") +
  scale_fill_gradientn(colors = c("#440154", "#31688e", "#35b779", "#fde725")) +
  labs(title = "XGBoost Confusion Matrix",
       subtitle = paste0("Recall=", xgb_recall, "% | Precision=", xgb_precision, "% | F1=", xgb_f1, "%"),
       x = "Predicted Label", y = "True Label") +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid = element_blank())


#comparision table
comparison_table <- data.frame(
  Metric = c("Accuracy %", "Recall %", "Precision %", "F1 %", "Specificity %", "AUC"),
  Logistic_Regression = c(lr_accuracy, lr_recall, lr_precision, lr_f1, lr_specificity, lr_auc),
  Decision_Tree = c(dt_accuracy, dt_recall, dt_precision, dt_f1, dt_specificity, dt_auc),
  Random_Forest = c(rf_accuracy, rf_recall, rf_precision, rf_f1, rf_specificity, rf_auc),
  XGBoost = c(xgb_accuracy, xgb_recall, xgb_precision, xgb_f1, xgb_specificity, xgb_auc)
)

print(comparison_table)

#ROC Curve plots
roc_data <- rbind(
  data.frame(FPR = 1 - lr_roc$specificities,  TPR = lr_roc$sensitivities,  Model = paste0("Logistic Regression (AUC=", lr_auc, ")")),
  data.frame(FPR = 1 - dt_roc$specificities,  TPR = dt_roc$sensitivities,  Model = paste0("Decision Tree (AUC=", dt_auc, ")")),
  data.frame(FPR = 1 - rf_roc$specificities,  TPR = rf_roc$sensitivities,  Model = paste0("Random Forest (AUC=", rf_auc, ")")),
  data.frame(FPR = 1 - xgb_roc$specificities, TPR = xgb_roc$sensitivities, Model = paste0("XGBoost (AUC=", xgb_auc, ")"))
)

ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(lwd = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  scale_color_manual(values = c("#2ecc71", "#e74c3c", "#9b59b6", "#f39c12")) +
  labs(title = "ROC Curve Comparison",
       subtitle = "All 4 Models on Test Set",
       x = "False Positive Rate",
       y = "True Positive Rate",
       color = "Model") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())
#PR Curve all models
pr_data <- rbind(
  data.frame(
    Recall    = pr_lr$curve[,1],
    Precision = pr_lr$curve[,2],
    Model     = paste0("Logistic Regression (AUC=", round(pr_lr$auc.integral, 3), ")")
  ),
  data.frame(
    Recall    = pr_dt$curve[,1],
    Precision = pr_dt$curve[,2],
    Model     = paste0("Decision Tree (AUC=", round(pr_dt$auc.integral, 3), ")")
  ),
  data.frame(
    Recall    = pr_rf$curve[,1],
    Precision = pr_rf$curve[,2],
    Model     = paste0("Random Forest (AUC=", round(pr_rf$auc.integral, 3), ")")
  ),
  data.frame(
    Recall    = pr_xgb$curve[,1],
    Precision = pr_xgb$curve[,2],
    Model     = paste0("XGBoost (AUC=", round(pr_xgb$auc.integral, 3), ")")
  )
)

ggplot(pr_data, aes(x = Recall, y = Precision, color = Model)) +
  geom_line(lwd = 1.2) +
  scale_color_manual(values = c("#2ecc71", "#e74c3c", "#9b59b6", "#f39c12")) +
  labs(title = "Precision-Recall Curve Comparison",
       subtitle = "All 4 Models on Test Set",
       x = "Recall", y = "Precision", color = "Model") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

#Threshold Analysis on XGBoost(Best model)
thresholds <- seq(0.1, 0.9, by = 0.05)
thresh_results <- data.frame()

for(t in thresholds) {
  preds_t <- factor(ifelse(xgb_probs >= t, "1", "0"), levels = c("0", "1"))
  cm_t    <- confusionMatrix(preds_t, test$Status, positive = "1")
  
  thresh_results <- rbind(thresh_results, data.frame(
    Threshold   = t,
    Recall      = round(cm_t$byClass["Sensitivity"] * 100, 2),
    Precision   = round(cm_t$byClass["Pos Pred Value"] * 100, 2),
    F1          = round(cm_t$byClass["F1"] * 100, 2)
  ))
}

print(thresh_results)


thresh_long <- thresh_results %>%
  pivot_longer(cols = c(Recall, Precision, F1),
               names_to = "Metric",
               values_to = "Value")

ggplot(thresh_long, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(lwd = 1.2) +
  geom_point(size = 2) +
  geom_vline(xintercept = 0.35, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = c("Recall" = "#e74c3c", 
                                "Precision" = "#2ecc71", 
                                "F1" = "#f39c12")) +
  labs(title = "XGBoost Threshold Analysis",
       subtitle = "Optimal threshold at 0.35 (best F1)",
       x = "Decision Threshold", y = "Score (%)") +
  theme_minimal(base_size = 13)

importance_xgb <- xgb.importance(model = model_xgb)
head(importance_xgb, 15)

#model comparision Bar Chart
comparison_long <- comparison_table %>%
  filter(Metric != "AUC") %>%
  pivot_longer(cols = c(Logistic_Regression, Decision_Tree, Random_Forest, XGBoost),
               names_to = "Model", values_to = "Value")

ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = paste0(Value, "%")),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 2.8) +
  scale_fill_manual(values = c("Logistic_Regression" = "#2ecc71",
                               "Decision_Tree"       = "#e74c3c",
                               "Random_Forest"       = "#9b59b6",
                               "XGBoost"             = "#f39c12"),
                    labels = c("Logistic Regression", "Decision Tree",
                               "Random Forest", "XGBoost")) +
  scale_y_continuous(limits = c(0, 115)) +
  labs(title = "Model Performance Comparison",
       subtitle = "All metrics at threshold 0.5",
       x = "Metric", y = "Value (%)", fill = "Model") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        plot.title = element_text(face = "bold"))




#top 5 feature plot
xgb_imp_df <- as.data.frame(importance_xgb[1:15])

ggplot(xgb_imp_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#f39c12") +
  coord_flip() +
  labs(title = "XGBoost Feature Importance",
       x = "Feature", y = "Gain") +
  theme_minimal(base_size = 12)
