# Anomaly Investigation

## Anomalies Identified

[List anomalies found in Phase 2]

1. **[Anomaly description]** - found in [which analysis]
2. **[Anomaly description]** - found in [which analysis]
3. **[Anomaly description]** - found in [which analysis]

---

## Anomaly 1: [Description]

### Where Found
[Reference to Phase 2 analysis where this was spotted]

Example: "In temporal analysis (02-temporal-patterns.md), noticed sales spike of 300% on 2024-02-14"

### Why It's Anomalous
[What makes this unusual or unexpected]

Example: "February 14 sales were $45,890, compared to typical daily average of $15,200 - a 3x spike"

### Investigation Query

```sql
-- Drill into the anomalous period
SELECT
  transaction_date,
  COUNT(*) as transaction_count,
  SUM(amount) as total_sales,
  ROUND(AVG(amount), 2) as avg_transaction,
  MIN(amount) as min_amount,
  MAX(amount) as max_amount,
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(DISTINCT product_id) as unique_products
FROM sales
WHERE transaction_date BETWEEN '2024-02-13' AND '2024-02-15'
GROUP BY transaction_date
ORDER BY transaction_date;
```

### Results
[Paste results]

### Deeper Investigation

[If needed, more queries to understand the anomaly]

```sql
-- Check if specific products drove the spike
SELECT
  p.product_name,
  p.category,
  COUNT(*) as transaction_count,
  SUM(s.amount) as total_sales
FROM sales s
JOIN products p ON s.product_id = p.id
WHERE s.transaction_date = '2024-02-14'
GROUP BY p.product_name, p.category
ORDER BY total_sales DESC
LIMIT 10;
```

### Results
[Paste results]

### Explanation

**Determination:** [Data quality issue / Real phenomenon / Unclear]

**Reasoning:** [Explain what caused this anomaly]

Example: "Real phenomenon - Valentine's Day (Feb 14) drove 3x spike in flower and chocolate sales. Pattern is legitimate seasonal effect, not data error."

**Action:** [Should this be excluded from other analyses? Noted as special case? Investigated further?]

---

## Anomaly 2: [Next anomaly]

[Repeat structure]

---

## Anomalies Summary

### Real Phenomena
[List anomalies that are legitimate patterns worthy of insight]

1. **[Anomaly]:** [Brief explanation]
2. **[Anomaly]:** [Brief explanation]

### Data Quality Issues
[List anomalies that are data errors or artifacts]

1. **[Issue]:** [What's wrong and how it affects analysis]

### Unexplained
[List anomalies you couldn't fully explain]

1. **[Anomaly]:** [What remains unclear]

### Implications for Pattern Discovery

[How do these anomalies affect your Phase 2 findings?]

- [Anomaly] explains [pattern from Phase 2]
- [Data issue] invalidates [finding from Phase 2]
- [Real phenomenon] is interesting enough to become a standalone insight
