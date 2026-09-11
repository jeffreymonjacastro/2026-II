# Temporal Pattern Exploration

## Objective

Discover how the data varies over time - trends, seasonality, cycles, or irregular patterns.

## Exploration Approach

[Explain what temporal analyses you'll perform]

1. Overall trend (is metric growing/shrinking/stable?)
2. Seasonal patterns (day-of-week, month-of-year, etc.)
3. Period-over-period comparisons (YoY, MoM, etc.)
4. Irregular events or anomalies in time series

---

## Analysis 1: Overall Trend

### Rationale
[Why checking overall trend first]

Example: "Establish baseline understanding of whether sales are growing, declining, or stable over the dataset period"

### Query
```sql
-- Monthly aggregation to see trend
SELECT
  STRFTIME('%Y-%m', transaction_date) as month,
  COUNT(*) as transaction_count,
  SUM(amount) as total_sales,
  ROUND(AVG(amount), 2) as avg_transaction,
  COUNT(DISTINCT customer_id) as unique_customers
FROM sales
GROUP BY month
ORDER BY month;
```

### Results
[Paste actual results]

### Observations
[What do you see? Facts only]

- First month: [stats]
- Last month: [stats]
- Overall direction: [growing/declining/stable/erratic]
- Growth rate: [approximate % change]
- Volatility: [stable or highly variable month-to-month]

### Text Visualization
[Use creating-visualizations skill to make a simple ASCII chart if helpful]

```
Month         Sales ($)
2024-01       ████████████████ 45,890
2024-02       ██████████████ 38,234
2024-03       ████████████████████ 52,145
```

---

## Analysis 2: Day-of-Week Patterns

### Rationale
[Why checking day-of-week]

### Query
```sql
SELECT
  CASE CAST(STRFTIME('%w', transaction_date) AS INTEGER)
    WHEN 0 THEN 'Sunday'
    WHEN 1 THEN 'Monday'
    WHEN 2 THEN 'Tuesday'
    WHEN 3 THEN 'Wednesday'
    WHEN 4 THEN 'Thursday'
    WHEN 5 THEN 'Friday'
    WHEN 6 THEN 'Saturday'
  END as day_name,
  CAST(STRFTIME('%w', transaction_date) AS INTEGER) as day_num,
  COUNT(*) as transaction_count,
  ROUND(AVG(amount), 2) as avg_amount,
  ROUND(SUM(amount), 2) as total_sales
FROM sales
GROUP BY day_num, day_name
ORDER BY day_num;
```

### Results
[Paste results]

### Observations
[Describe pattern - which days are high/low, by how much]

---

## Analysis 3: [Other temporal pattern]

[Repeat structure for month-of-year, hour-of-day if timestamps available, etc.]

---

## Temporal Patterns Summary

### Key Findings
1. **[Pattern 1]:** [Description with magnitude]
2. **[Pattern 2]:** [Description with magnitude]
3. **[Pattern 3]:** [Description with magnitude]

### Interesting Anomalies
- [Any unusual spikes or drops worth investigating]

### Questions Raised
- [What questions does this create for later investigation?]
