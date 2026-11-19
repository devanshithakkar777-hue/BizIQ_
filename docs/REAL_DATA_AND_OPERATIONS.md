# Running BizIQ on real business data

## What happens to a file you upload
1. **Reading.** CSV, TSV, TXT, XLSX/XLS. BizIQ detects the encoding (UTF-8, UTF-16, Windows-1252) and delimiter (comma, semicolon, tab, pipe),
   finds the real heading row under title lines ("Sales Register", company name, period), picks the right sheet in a workbook, and keeps IDs as
   text (so `00123` stays `00123`). Files over `BIZIQ_MAX_UPLOAD_MB` (default 200) are refused with a message.
2. **Column matching.** Headings from Tally, Zoho, Busy, Marg, myBillBook, Shopify and plain Excel are matched by name and by a fuzzy score.
   Confident matches are applied and listed. Weaker ones (score below 0.95) are only **suggested**: you confirm them on the Data page. Nothing weak is applied silently.
3. **Cleaning.** Day-first dates by default (`BIZIQ_DATE_ORDER=dmy`; month-first files are detected from values like 04/25/2026); Excel serial and `YYYYMMDD` dates;
   ambiguous dates (03/04/2026 could be two dates) are flagged, never guessed quietly; Indian number formats (`1,23,456.00`, `(5,000)`, `12,000 Dr`, `Rs.`, `₹`);
   total/subtotal rows removed; placeholders (`-`, `N/A`, `nil`) treated as blank.
4. **Nothing disappears silently.** Every row that is set aside appears in a table with the **row number in your file** and the reason, and can be downloaded as CSV.
   Download files are safe to open in Excel (cells starting `= + - @` are neutralised).
5. **Duplicates.** Line-level data (sales lines, expenses, returns...) keeps identical rows, because two identical lines can be two real sales; BizIQ warns instead of deleting.
   Keyed data (invoices, pipeline, quotes, HR...) keeps the **latest** version of a repeated key.
6. **Saving.** For a dataset that already has data you choose: *Add new rows and update existing ones* (monthly exports: re-uploading the same file changes nothing; an invoice now marked paid is updated) or *Replace everything*.
   **Undo last load** restores the previous version (pressing it again re-applies). Every load is written to the audit log with file name, mode and counts.
7. **Data health** panel: rows, date range, days since the newest record and a plain status (ok / stale / not loaded) per dataset.

## Measured on synthetic data (1,000,000 sales rows, 67 MB CSV, one laptop-class container)
read 2.0 s, clean 2.3 s, save 3.5 s, full analysis of all modules 19 s, peak memory 1.4 GB. No module failed. Real files add messier values than this generator does,
so treat the numbers as a floor. Before this work the same load took about two minutes and 3 GB.

## Robustness
Modules were run against one-row, two-row, empty, all-zero and constant-value tables for every dataset (`pytest -m slow`). The pipeline never raises; a module that
cannot run reports why in plain words ("needs at least 6 months of sales"). Empty tables count as missing. Known remaining limits: absurd values (1e9 days) in date-like fields.

## Operating it
* `docker compose up -d` starts the app, the scheduler and a daily backup. Set `BIZIQ_SECRET_KEY` in `.env` first and keep a copy somewhere safe.
  Put HTTPS in front of it before exposing it beyond your own network.
* Data lives in `BIZIQ_DATA_DIR` (`/data` in Docker; one SQLite file per organisation). Health check: `/_stcore/health`. Logs go to stderr, level `BIZIQ_LOG_LEVEL`.
* **Backups:** `python scripts/backup.py` (online-safe, integrity-checked, keeps 14). Copy `backups/` to another machine. **Restore:** stop the app, then
  `python scripts/backup.py --restore <timestamp> --yes` (the current files are kept under `backups/pre-restore-*`). The backup does not contain `BIZIQ_SECRET_KEY`.
* SQLite handles a handful of simultaneous users. For a real multi-user launch plan the move to Postgres.

## Still not proven
No real customer file has been run through it. The first real export will find things this does not know. Use the rejected-rows table and the suggestions on the Data page as your feedback loop, and add each new heading you meet to `biziq/mapping.py` (ALIASES).
