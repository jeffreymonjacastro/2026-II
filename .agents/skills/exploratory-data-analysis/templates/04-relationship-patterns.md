# Relationship Pattern Exploration

## Objective

Discover correlations, associations, or dependencies between variables.

## Exploration Approach

[Explain what relationships you'll explore]

1. Metric correlations (does X relate to Y?)
2. Sequential patterns (does event A precede event B?)
3. Cohort behaviors (do groups behave differently over time?)

---

## Analysis 1: [Relationship description]

### Rationale
[Why exploring this relationship]

Example: "Check if transaction size correlates with customer tenure"

### Query
```sql
-- Create tenure buckets and compare average transaction size
SELECT
  CASE
    WHEN tenure_days < 30 THEN '0-30 days'
    WHEN tenure_days < 90 THEN '30-90 days'
    WHEN tenure_days < 180 THEN '90-180 days'
    WHEN tenure_days < 365 THEN '180-365 days'
    ELSE '1+ years'
  END as tenure_bucket,
  COUNT(*) as transaction_count,
  ROUND(AVG(amount), 2) as avg_amount,
  COUNT(DISTINCT customer_id) as unique_customers
FROM (
  SELECT
    s.*,
    JULIANDAY(s.transaction_date) - JULIANDAY(c.signup_date) as tenure_days
  FROM sales s
  JOIN customers c ON s.customer_id = c.id
)
GROUP BY tenure_bucket
ORDER BY MIN(tenure_days);
```

### Results
[Paste results]

### Observations
- [Pattern description]
- [Magnitude of correlation if present]
- [Direction: positive/negative/no relationship]

---

## Analysis 2: [Another relationship]

[Repeat structure]

---

## Relationship Patterns Summary

### Key Findings
1. **[Finding 1]:** [Description]
2. **[Finding 2]:** [Description]

### Correlations Identified
- [List any meaningful correlations with magnitude]

### Questions Raised
- [What questions does this create?]
