# Data Dictionary

## customers
- `customer_id` (TEXT, PK): anonymized customer identifier
- `signup_date` (TEXT): ISO date the customer signed up
- `country` (TEXT): country code
- `segment` (TEXT): customer segment (Consumer/SMB/Enterprise)

## products
- `product_id` (TEXT, PK)
- `category` (TEXT)
- `brand` (TEXT)
- `list_price_usd` (REAL): reference list price

## sessions
- `session_id` (TEXT, PK)
- `customer_id` (TEXT, nullable): empty/NULL indicates guest session
- `session_start_ts` (TEXT): ISO timestamp
- `channel` (TEXT): organic, paid_search, social, email, referral, affiliate
- `device` (TEXT): mobile/desktop/tablet
- `country` (TEXT)

## events
- `event_id` (TEXT, PK)
- `session_id` (TEXT, FK sessions)
- `event_ts` (TEXT): ISO timestamp
- `event_type` (TEXT): view_product, add_to_cart, purchase
- `product_id` (TEXT, FK products, nullable): purchase event may not reference a product

## orders
- `order_id` (TEXT, PK)
- `customer_id` (TEXT, FK customers)
- `order_ts` (TEXT): ISO timestamp
- `status` (TEXT): delivered, cancelled, returned
- `subtotal_usd` (REAL): sum of line items (before discounts, shipping)
- `discount_usd` (REAL): total discount
- `shipping_usd` (REAL)
- `tax_usd` (REAL) (kept as 0 for simplicity)
- `refund_usd` (REAL): refund amount if cancelled/returned
- `total_usd` (REAL): subtotal - discount + shipping + tax
- `payment_method` (TEXT)

## order_items
- `order_id` (TEXT, FK orders)
- `product_id` (TEXT, FK products)
- `quantity` (INTEGER)
- `unit_price_usd` (REAL)
- `discount_usd` (REAL): discount for the line
