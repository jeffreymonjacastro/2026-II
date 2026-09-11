# Data Familiarization

## Exploration Context

**Dataset:** [Name or description]
**Source:** [Where data came from, if known]
**Exploration Goal:** [User's request, e.g., "Find interesting patterns in sales data"]

## Tables Overview

[List all tables with row counts and apparent purposes]

### Table: [table_name]
**Row count:** [count]
**Apparent purpose:** [What this table seems to represent]
**Grain:** [What one row represents, e.g., "one transaction", "one customer", "one daily summary"]

Example:
### Table: sales
**Row count:** 15,847
**Apparent purpose:** Transaction-level sales records
**Grain:** One row per individual sale transaction

### Table: [table_name_2]
...

## Schema Details

[For each table, document key columns]

### Table: sales
```sql
PRAGMA table_info(sales);
```

**Key columns identified:**
- `transaction_date` (TEXT) - when sale occurred
- `amount` (REAL) - sale value in currency
- `product_id` (INTEGER) - product identifier
- `customer_id` (INTEGER) - customer identifier
- `region` (TEXT) - geographic region

**Potential relationships:**
- `product_id` likely joins to products table
- `customer_id` likely joins to customers table

**Temporal coverage:**
- Earliest date: [result from MIN(transaction_date)]
- Latest date: [result from MAX(transaction_date)]
- Coverage: [X months/years of data]

## Data Quality Assessment

### Completeness

```sql
-- Check for NULL values in key columns
SELECT
  COUNT(*) as total_rows,
  COUNT(transaction_date) as non_null_dates,
  COUNT(amount) as non_null_amounts,
  COUNT(product_id) as non_null_products
FROM sales;
```

**Results:** [Paste results]

**Assessment:**
- [Column]: [% complete], [any concerns]
- [Column]: [% complete], [any concerns]

### Value Distributions

```sql
-- Check numeric column ranges
SELECT
  MIN(amount) as min_amount,
  MAX(amount) as max_amount,
  ROUND(AVG(amount), 2) as avg_amount,
  COUNT(DISTINCT customer_id) as unique_customers,
  COUNT(DISTINCT product_id) as unique_products
FROM sales;
```

**Results:** [Paste results]

**Observations:**
- Amount range: $[min] to $[max] (average: $[avg])
- [Number] unique customers
- [Number] unique products
- [Any outliers or surprises]

### Temporal Distribution

```sql
-- Check temporal distribution
SELECT
  STRFTIME('%Y-%m', transaction_date) as month,
  COUNT(*) as transaction_count
FROM sales
GROUP BY month
ORDER BY month;
```

**Results:** [Paste results]

**Observations:**
- [Gaps in data?]
- [Seasonality visible?]
- [Growing or shrinking over time?]
- [Complete weeks/months or partial?]

## Initial Impressions

[Before exploring patterns, note your initial observations]

- Dataset covers [timeframe]
- [Number] transactions across [number] products/customers/regions
- Data quality appears [good/fair/poor] because [reasons]
- Potential areas of interest:
  1. [Something that caught your attention]
  2. [Another interesting aspect]
  3. [Third area to explore]

## Exploration Strategy

[Based on familiarization, what vectors will you explore?]

Will explore:
1. **Time-based patterns:** [What temporal analyses make sense?]
2. **Segmentation patterns:** [What groups/categories to compare?]
3. **Relationship patterns:** [What correlations or associations to check?]
