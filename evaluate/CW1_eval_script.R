# Load libraries
library(data.table)

# Set seed
set.seed(123)

# Import training data (run from project root: Rscript src/CW1_eval_script.R)
trn <- fread('data/CW1_train.csv')
tst <- fread('data/CW1_test.csv')  # no true outcomes

# Train your model (using a simple LM here as an example)
f <- lm(outcome ~ ., data = trn)

# Test set predictions
yhat_lm <- predict(f, tst)

# Format submission: single-column CSV with predictions
out <- data.table('yhat' = yhat_lm)
fwrite(out, 'CW1_submission_KNUMBER.csv')  # replace KNUMBER with your k-number

################################################################################
# R² on true outcomes (only when staff provide the file)
if (file.exists('data/CW1_test_with_true_outcome.csv')) {
  tst_true <- fread('data/CW1_test_with_true_outcome.csv')
  r2_fn <- function(yhat) {
    eps <- tst_true$outcome - yhat
    rss <- sum(eps^2)
    tss <- sum((tst_true$outcome - mean(tst_true$outcome))^2)
    1 - (rss / tss)
  }
  cat('R²:', round(r2_fn(yhat_lm), 4), '\n')
}
