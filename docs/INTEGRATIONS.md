# Live integrations: setup and honest limits

Everything below was written from the vendors' public documentation and tested against **simulated** responses (`tests/test_live.py`).
None has been run against a real account. Do the first sync on a test store/company, then compare totals with the vendor's own reports.

## Before you start: where secrets go
* Easiest and safest: set environment variables on the server, e.g. `BIZIQ_SHOPIFY_TOKEN`. Pattern: `BIZIQ_<SOURCE>_<FIELD>`
  (`BIZIQ_ZOHO_CLIENT_SECRET`, `BIZIQ_WHATSAPP_TOKEN`). They override anything saved in the app.
* To save tokens in BizIQ (encrypted), set `BIZIQ_SECRET_KEY` first:
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Lose the key and saved tokens are unreadable.
* Without a key, BizIQ never writes a secret to disk.

## Shopify (orders -> sales)
* Needs a custom-app Admin API access token with `read_orders` (and `read_all_orders` if you want more than 60 days, which Shopify must approve).
* Settings: store address `your-store.myshopify.com`, API version (default `2026-07`; use one your store supports), days to read.
* Uses the GraphQL Admin API (REST is legacy). Cancelled orders are skipped; orders with more than 100 lines are cut and noted.
* Shopify's app-creation flow changes often; follow their current docs for obtaining the token.

## Zoho Books (invoices and bills -> invoices)
* Needs: data centre (`in` for India), organization id, client id/secret, and a refresh token with `ZohoBooks.invoices.READ` and `ZohoBooks.bills.READ`
  (create a Self Client in the Zoho API Console and generate the refresh token once).
* Fully paid documents with no payment date are left out (not guessed); part-paid ones load at the amount still outstanding.

## Tally (vouchers -> sales and invoices)
* Tally must be running with a company loaded and HTTP server enabled (default port 9000). BizIQ must be able to reach the address.
* Receivable/payable status comes from Receipt/Payment vouchers that settle bills by reference. Bills without a credit period use your default.
* The Day Book export may omit stock-item lines depending on your Tally configuration; those invoices load as one line at the total.
* Credit/debit notes are ignored. Reconcile against Tally's Outstandings report.

## myBillBook
* **No public API was found**, so there is no live connection. Two routes work today:
  1. Upload its report downloads on the Data page using the `mybillbook_*` presets (column headings are unverified guesses; check the preview).
  2. Request its "Data export to Tally" (Tally-compatible XML, emailed to you, not real-time) and import it on Connections > Import a file.
* For a live sync, ask myBillBook for API access; if they give documentation it can be added as another connector.

## WhatsApp reminders
* Needs a Meta business account, WhatsApp Cloud API phone number id, a permanent access token, and an **approved template** with two body
  variables: `{{1}}` customer name, `{{2}}` amount overdue.
* Flow: ReceivablesIQ -> "Submit reminder drafts for approval" -> approver approves -> Connections > Send approved payment reminders.
* Safeguards: dry run unless `BIZIQ_SEND_WHATSAPP=1`; only opted-in customers with a phone number; each approval+customer is sent once ever;
  every send, skip and failure is audited. You are responsible for consent and WhatsApp's policies.
