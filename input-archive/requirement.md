# Requirement: Order Manager for Trading System

## Functional requirements

Build the order manager for a trading system. When a user places a buy or sell order, the order manager must:

1. **Validate the account** — confirm the account is active, not frozen, and has no restrictions in place.
2. **Validate the trade type** — confirm the requested trade type (cash equity, options, margin, etc.) is permitted for the account based on its eligibility.
3. **Collect order facts** — fetch the account's current positions and running balance.
4. **Validate funds / position** — for buys, confirm sufficient buying power; for sells, confirm sufficient holdings.
5. **Submit the order** — once all validations pass, submit the order to the downstream broker / exchange gateway.
6. **Persist as pending** — write the submitted order to a `pending_orders` table.
7. **Settle** — when settlement confirmation arrives, atomically move the order row from `pending_orders` to `transaction_history`.

## Non-functional requirements

The user did not provide explicit NFRs. The skill auto-classifies based on domain (financial / trading) and applies Tier M defaults plus financial-domain heuristics. The assumed NFRs are documented in `tech-stack.md` under `## Assumptions` and should be tuned if reality differs.

## Notes

- This is not a high-frequency-trading (HFT) system. Microsecond latency is not required.
- Settlement timing is governed by the exchange / clearing house, not by this system. The settlement workflow is event-driven from external settlement notifications.
- Audit trail is regulatory (SOX, FINRA-relevant). Every order state transition must be immutable and recoverable.
