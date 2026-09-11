# Questions for Further Investigation

## Purpose

Exploratory analysis has identified patterns and insights. This document translates those insights into specific, answerable questions for deeper investigation.

---

## Question 1: [Specific question]

### What We Learned in Exploration

[Reference the insight that led to this question]

Example: "Exploration revealed that weekend sales are 40% lower than weekdays (Insight 1), but we cannot determine if this is due to operational constraints or customer behavior."

### The Question

[State the specific question to investigate]

Example: "Is the weekend sales decline caused by reduced operating hours, or do customers simply shop less on weekends when given equal opportunity?"

### Why It Matters

[Explain business value of answering this question]

Example: "If caused by reduced hours, extending weekend hours could capture ~$50K additional monthly revenue. If caused by customer behavior, we should optimize staffing for weekday peaks and reduce weekend overhead."

### Recommended Process

**Process Skill:** [Which DataPeeker process would answer this?]

- [ ] `hypothesis-testing` - if you have a specific claim to test
- [ ] `guided-investigation` - if you need to systematically explore the question
- [ ] `comparative-analysis` - if you need to compare specific segments

Example: **`comparative-analysis`** - compare weekend vs weekday sales controlling for operating hours

### Data Required

[What data would you need to answer this?]

- Current data: [What you already have]
- Additional data needed: [What you'd need to collect]

Example:
- Current data: Transaction timestamps (if available - check if we have hour-of-day data)
- Additional data needed: Store operating hours by day-of-week

### Investigation Approach

[Brief outline of how to investigate]

Example:
1. Extract hour-of-day from transaction timestamps
2. Calculate sales per operating hour for weekend vs weekday
3. Compare: if sales/hour are equal, it's an hours issue; if sales/hour are lower on weekends, it's a demand issue

### Priority

**Priority:** [High / Medium / Low]

**Reasoning:** [Why this priority level?]

Example:
**Priority:** High

**Reasoning:** Potential $50K/month revenue impact and requires minimal data collection (may already have timestamp data). Quick investigation with high-value outcome.

---

## Question 2: [Next question]

[Repeat structure]

---

## Question 3: [Next question]

[Repeat structure]

---

## Questions Summary

### High Priority Questions
[List high-priority questions in recommended investigation order]

1. **[Question]** - Expected value: [impact], Estimated effort: [time/data required]
2. **[Question]** - Expected value: [impact], Estimated effort: [time/data required]

### Medium Priority Questions
[Questions that would be valuable but less urgent]

### Questions Deferred
[Questions that can't be answered with available data or aren't worth the effort]

**[Question]:** [Why deferred]

### Investigation Roadmap

[If someone wanted to investigate all high-priority questions, what order makes sense?]

Example:
1. Start with Q1 (weekend sales) - quick win, uses existing data
2. Then Q3 (customer segmentation) - builds on Q1 findings
3. Finally Q2 (product mix) - most complex, benefits from context from Q1 and Q3

## Data Collection Priorities

[What new data sources would unlock multiple questions?]

Example: "Adding customer segment identifier to transaction data would enable investigation of Q2, Q4, and Q5. High-value data enrichment."
