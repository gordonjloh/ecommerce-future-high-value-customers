# Predicting Future High-Value Customers (E-Commerce, Python)

## 1. Business Problem

For an e-commerce business, not all customers contribute equally to revenue.  
The goal of this project is to **predict which existing customers are likely to become high-value in the future**, so that marketing and retention efforts can be focused on the right people instead of treating everyone the same.

We use historical transaction data (customers and their orders) to estimate the **probability that each customer will become a future high-value customer**, based on their past purchasing behaviour.

---

## 2. Data

The project uses two main tables:

- **Customers table** – one row per customer  
  - `customer_id` and basic profile fields

- **Purchases / Orders table** – one row per transaction  
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
