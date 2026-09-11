# Segmentation Pattern Exploration

## Objective

Discover how the data varies across natural groups - categories, segments, cohorts, or clusters.

## Exploration Approach

[Explain what segmentation dimensions you'll explore]

1. Categorical dimensions (region, product type, customer tier, etc.)
2. Derived segments (high/medium/low value, new/returning, etc.)
3. Cross-segments (region × product, etc.)

---

## Analysis 1: [Dimension] Distribution

### Rationale
[Why exploring this dimension]

Example: "Understand if sales are concentrated in certain regions or distributed evenly"

### Query
```sql
SELECT
  region,
  COUNT(*) as transaction_count,
  SUM(amount) as total_sales,
  ROUND(AVG(amount), 2) as avg_transaction,
  COUNT(DISTINCT customer_id) as unique_customers,
  -- Calculate percentage of total
  ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM sales), 1) as pct_of_transactions
FROM sales
WHERE region IS NOT NULL
GROUP BY region
ORDER BY total_sales DESC;
```

### Results
[Paste results]

### Observations
- [Top segment]: [stats] ([X]% of total)
- [Bottom segment]: [stats] ([X]% of total)
- Concentration: [How concentrated or distributed?]
- Differences: [Magnitude of differences between segments]

---

## Analysis 2: [Another dimension]

[Repeat structure for products, customer segments, etc.]

---

## Analysis 3: Cross-Segmentation

### Rationale
[Why looking at interaction of two dimensions]

Example: "Check if product preferences vary by region"

### Query
```sql
SELECT
  region,
  product_category,
  COUNT(*) as transaction_count,
  ROUND(SUM(amount), 2) as total_sales
FROM sales
  JOIN products ON sales.product_id = products.id
WHERE region IS NOT NULL AND product_category IS NOT NULL
GROUP BY region, product_category
ORDER BY region, total_sales DESC;
```

### Results
[Paste results]

### Observations
- [Pattern across segments]
- [Interesting variations]

---

## Segmentation Patterns Summary

### Key Findings
1. **[Finding 1]:** [Description]
2. **[Finding 2]:** [Description]

### Notable Differences
- [Biggest differences between segments]

### Questions Raised
- [What questions does this create?]
