# Predicting Future High-Value Customers (E-Commerce, Python)

## 1. Business Problem

For an e-commerce business, not all customers contribute equally to revenue.  
The goal of this project is to **predict which existing customers are likely to become high-value in the future**, so that marketing and retention efforts can be focused on the right people instead of treating everyone the same.

We use historical transaction data (customers and their orders) to estimate the **probability that each customer will become a future high-value customer**, based on their past purchasing behaviour.

---

## 2. Data

The project uses two main tables:

- **Customers table**: one row per customer  
  - `customer_id` and basic profile fields

- **Purchases / Orders table**: one row per transaction  
  - `order_id`, `customer_id`, `order_date`, `price`, `quantity`, `discount`, etc.

From the raw transactions, we engineer **customer-level behavioural features** and a **future high-value label**.

> Note: The data used here is for learning/demo purposes (course / synthetic-style data), not real production data.

---

## 3. Approach

**Tools**

- Python, Jupyter Notebook  
- `pandas`, `numpy`  
- `scikit-learn` (logistic regression, train/test split, metrics)

**Key Steps**

1. **Data Preparation & Aggregation**
   - Aggregate transaction data to the customer level:
     - `past_total_spend`
     - `past_num_orders`
     - `past_avg_order_value`
     - `last_purchase_date`
   - Create **recency**: days since last purchase at a chosen cut-off date.

2. **Past vs Future Split & Label Definition**
   - Choose a **cut-off date** and split transactions into:
     - **Past**: used for features
     - **Future**: used to define the target label
   - Define **`future_total_spend`** for each customer (after the cut-off).
   - Label customers as **future high-value** if their `future_total_spend` is in the **top 25%** (quantile threshold).

3. **Feature Engineering**
   - Use past data to build features:
     - `past_total_spend`
     - `past_num_orders`
     - `past_avg_order_value`
     - `recency_days` (days since last past purchase)

4. **Model Training & Evaluation**
   - Train a **logistic regression** model to predict `high_value_future` (0/1).
   - Split into train/test sets (e.g. 70/30) with stratification.
   - Evaluate performance using:
     - Accuracy
     - Confusion matrix
     - Precision, recall, F1-score

5. **Interpretation & Ranking**
   - Inspect model coefficients to understand which behaviours drive future value.
   - Use `predict_proba` to generate a **predicted probability of being high-value** for each customer.
   - Rank customers by this probability to create a **prioritised list** for marketing or retention campaigns.

---

## 4. Results (Summary)

- The logistic regression model achieves around **79% accuracy** on the test set.
- For non–high-value customers, recall is high (~98%), meaning the model rarely mislabels them as high-value.
- For future high-value customers, the model is **conservative**:
  - Precision is around **78%** (when it flags a customer as high-value, it is often correct),
  - Recall is around **24%** (it captures a subset of all high-value customers, not all of them).

**Key behavioural drivers** observed from the model:

- Higher **past total spend** → higher probability of being future high-value.
- Longer **recency** (has not purchased for a long time) → lower probability.
- After controlling for total spend, customers with many small orders may be less likely to become high-value than those with fewer, larger purchases.

---

## 5. How to Run

1. Clone this repository or download it as a ZIP.
2. Open the main notebook (e.g. `notebooks/01_future_high_value_customers.ipynb`) in Jupyter or VS Code.
3. Ensure required Python packages are installed:
   - `pandas`, `numpy`, `scikit-learn`
4. Run the cells from top to bottom.

*(In this portfolio version, sensitive or large raw data may be replaced with sample data or described at a high level.)*

---

## 6. Possible Next Steps

- Try additional models (e.g. Random Forest, Gradient Boosting) and compare performance.
- Adjust class weights or decision thresholds to improve recall for high-value customers.
- Add more features (RFM-style, product categories, channels).
- Wrap the model into a simple **Streamlit app** for non-technical users to score customers.

## 7. Model Performance & Visual Insights

### Confusion Matrix (Test Set)

![Confusion Matrix](images/pa_confusion_matrix.png)

The confusion matrix shows that the model is very good at recognising customers who will not become high-value (220 correctly classified as non–high-value and only 5 wrongly flagged as high-value), while it correctly identifies a smaller number of true future high-value customers (18) and misses many others (57). In plain language, when the model says “this customer is high-value,” it is usually right, but it is conservative and leaves many potential high-value customers unflagged. Overall accuracy is around 79%, so as a first pass this gives us a reliable, if cautious, signal for selecting a shortlist of high-potential customers.  
**In short, the model is accurate but cautious.**

### Distribution of Predicted Probabilities

![Predicted Probabilities](images/pa_prob_hist.png)

The distribution of predicted probabilities shows that most customers receive relatively low scores, with only a small tail of very high probabilities. This means the model rarely over-confidently labels customers as high-value and tends to keep most people in a low-risk bucket. It also suggests that using the default 0.5 cut-off is strict; lowering the threshold would expand the target pool at the cost of including more customers who will not ultimately become high-value.  
**Overall, probability scores are skewed towards low risk.**

### Predicted Probabilities by True Class

![Probabilities by Class](images/pa_prob_by_class.png)

When we compare predicted probabilities by true class, true high-value customers clearly tend to receive higher scores than non–high-value customers, but there is still overlap in the middle range. This matches the confusion matrix: the model is genuinely learning useful patterns, but a strict threshold leads to many high-value customers being classified as non–high-value. In practice, this means the business can adjust how far down the probability ranking it goes to deliberately trade some precision for better coverage.  
**So recall can be improved by lowering cut-offs.**

### Feature Importance (Logistic Regression Coefficients)

![Coefficients](images/pa_coefficients.png)

The coefficient chart shows that higher past total spend and patterns consistent with fewer, larger orders increase the probability of becoming high-value, while longer recency (more days since last purchase) and patterns of many small orders decrease it. Based on this, an actionable strategy is to rank all customers by predicted probability and define at least two segments—for example, a **Tier A** consisting of the top 20–30% by predicted probability and a **Tier B** consisting of the next band (e.g. 30–50%). Marketing can then run targeted campaigns for Tier A and Tier B and compare their uplift in revenue, repeat purchase rate or average order value against a control group, turning the model into a measurable, testable intervention rather than a purely academic exercise.  
**This segmentation makes the model directly usable operationally.**

![ROC Curve](images/pa_roc_curve.png)

The ROC curve provides an additional view of how well the model distinguishes future high-value customers from non–high-value customers across different probability thresholds. Instead of fixing a single cut-off (for example 0.5), it shows how the true positive rate and false positive rate move together as we vary the threshold.

In this project, the ROC AUC is **0.76**, which is clearly better than random guessing (0.50) but not perfect. This means the model has a **meaningful but imperfect ability to rank customers by their likelihood of becoming high-value**: high-value customers tend, on average, to receive higher predicted probabilities than non–high-value customers, even though there is still overlap.

This supports using the predicted probabilities to create ranked tiers (for example, Tier A for the top 20–30% and Tier B for the next band) and choosing thresholds in a way that fits the business’ tolerance for false positives versus missed opportunities. The ROC curve and AUC of 0.76 confirm that the model is genuinely useful for ranking customers, even if it should be combined with business rules and monitored over time.

