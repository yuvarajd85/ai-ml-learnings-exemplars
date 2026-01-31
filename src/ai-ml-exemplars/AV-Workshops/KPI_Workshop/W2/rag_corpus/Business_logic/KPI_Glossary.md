# KPI Glossary

This glossary is intentionally **practical and auditable**: every KPI includes a definition, formula, and notes.

## 1) Sessions
**Definition:** A visit to the site/app (one user session).  
**Primary table:** `sessions`

**Formula:** `COUNT(sessions)` in the time window.

## 2) Orders
**Definition:** A completed checkout transaction (an order record).  
**Primary table:** `orders`

**Notes:**  
- In this dataset, `orders.status` can be `delivered`, `cancelled`, or `returned`.
- For revenue KPIs, exclude `cancelled` and `returned` unless explicitly stated.

## 3) Purchase Conversion Rate
**Definition:** Share of sessions that result in a purchase/order.  
**Formula:** `orders / sessions` for the same time window.  
**Reference:** Adobe definition of ecommerce conversion rate.

## 4) GMV (Gross Merchandise Value)
**Definition:** Total value of merchandise sold (before fees/costs).  
**Formula (line-item):** `SUM(quantity * unit_price_usd)`  
**Reference:** Investopedia definition of GMV.

## 5) Net Revenue (simplified)
**Definition:** Revenue after discounts and excluding cancelled/returned orders.  
**Formula (order-level):** `SUM(total_usd)` over orders where status NOT IN ('cancelled','returned')  
**Note:** Refund handling can vary across companies; here we treat cancelled/returned as zero revenue.

## 6) AOV (Average Order Value)
**Definition:** Average amount spent per order in a period.  
**Formula:** `net_revenue / completed_orders`  
**Reference:** Shopify AOV definition and formula.

## 7) Repeat Purchase Rate
**Definition:** % of customers who purchase more than once in a period (or lifetime, as specified).  
**Formula (lifetime-style):** customers with >=2 completed orders / customers with >=1 completed order  
**Reference:** Wall Street Prep definition of repeat purchase rate.

## 8) 60-day Churn Rate (retail-style, operational definition)
**Definition:** Customers who previously purchased but have not purchased in the last 60 days (as of an anchor date).  
**Formula:** churned_customers / eligible_customers  
**Reference:** Salesforce overview of churn and churn rate (companies operationalize time windows differently).

## 9) Cart Abandonment (session-level)
**Definition:** Sessions where users added to cart but did not purchase.  
**Formula:** sessions_with_add_to_cart_and_no_purchase / sessions_with_add_to_cart
**Primary table:** `events`

## 10) Channel Mix
**Definition:** Distribution of sessions by acquisition channel.  
**Formula:** sessions by `sessions.channel` in the time window.

---
For SQL examples, see `sql/kpi_queries.sql`.
