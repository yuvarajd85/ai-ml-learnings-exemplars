---
description: >
  Seed the database with N demo users, each with M realistic tickets across statuses,
  priorities, and categories. Each ticket carries a consistent activity history matching
  its final status. Supports --reset to wipe and re-seed. FR-SEED-01..04.
argument-hint: "[users] [tickets] [--reset]   e.g.  5 10  or  3 5 --reset  or  --reset --users 0"
---

Seed the TicketFlow database with demo data by running `db/seed.py`.

## Argument parsing

Parse `$ARGUMENTS` as follows:
- **First positional integer** (if present): number of demo users. Default **5**.
- **Second positional integer** (if present): tickets per user. Default **10**.
- **`--reset`** anywhere in the argument string: wipe existing seed data before seeding.
- **`--users 0`** combined with `--reset`: wipe only, skip seeding.

Examples:
| Input | Command run |
|---|---|
| *(none)* | `python db/seed.py --users 5 --tickets 10` |
| `3 20` | `python db/seed.py --users 3 --tickets 20` |
| `--reset` | `python db/seed.py --reset --users 5 --tickets 10` |
| `--reset --users 0` | `python db/seed.py --reset --users 0` |
| `10 50 --reset` | `python db/seed.py --reset --users 10 --tickets 50` |

## Steps

1. Verify `db/seed.py` exists. If not, tell the user and stop.

2. Construct the command from the parsed arguments and run it from the project root:
   ```
   python db/seed.py [--reset] --users <N> --tickets <M>
   ```

3. Stream or display the full script output so the user can see what was created.

4. On success, print a one-line summary:
   > Created N users and ~T tickets. Login password for all demo accounts: **Demo1234!**

5. On failure (DB connection error, import error):
   - Show the error message.
   - Suggest: "Check that MySQL is running and `.env` contains the correct `MYSQL_*` variables,
     then try again."

## What the seeder does

- **Users** — created with the `@demo.ticketflow.test` email domain (the reset marker).
  Password for every demo account: `Demo1234!`
- **Tickets** — realistic titles and descriptions across all five categories
  (Bug, Feature Request, Access, Hardware, Other), distributed across all four priorities
  and all four statuses.
- **Activity history** — each ticket's activity rows match its final status:
  - OPEN: one `created` row.
  - IN_PROGRESS: `created` → `status_changed OPEN→IN_PROGRESS`.
  - RESOLVED: `created` → `OPEN→IN_PROGRESS` → `IN_PROGRESS→RESOLVED`.
  - CLOSED: `created` → `OPEN→IN_PROGRESS` → `IN_PROGRESS→RESOLVED` → `RESOLVED→CLOSED`.
- **Idempotency** — re-running without `--reset` skips users that already exist (no duplicates).
- **Reset** — `--reset` deletes rows identified by the `@demo.ticketflow.test` marker
  (ticket_activity → tickets → users, respecting foreign-key constraints) then re-seeds.
  Use `--reset --users 0` to wipe without re-seeding.

## Constraints

- Do **not** modify `db/seed.py` before running it unless the user explicitly asks.
- The seeder uses the same application DB functions as the running app (`db_create_ticket`,
  `db_update_ticket_status`) so every ticket write is atomic and produces real activity rows.
- The reset deletes only rows in the `@demo.ticketflow.test` domain — production data is safe.
