# BizIQ build status (honest version)

## Built and tested (about 430 automated tests on SQLite and Postgres, plus a slow robustness sweep; app boots, every page renders)
* **Accounts and personalisation (this round).** Self-service sign-up creates a business with its own sign-in page (`?org=<slug>`) and its own data store; data persists between logins, so nobody re-uploads.
  Email verification (needs SMTP; off by default), hashed single-use expiring links, anti-enumeration responses, rate limits on sign-up and reset, 7-day invites.
  A business profile (type plus capabilities) decides which modules apply; a retail shop that opens RecipeIQ sees "Not applicable for your business model", and an owner can enable it anyway.
* **BillBook (this round)**: counter-style billing for small Indian shops: New bill, Day book, Customers and dues, Items and stock; English and Hindi on these screens. Stock is deducted per bill; dues are always derived (FIFO) from bills and payments, never typed;
  day profit counts only items with a cost price and says how much it could not count. Bills feed the IQ modules automatically (sales, invoices, inventory, costs, expenses). WhatsApp links are `wa.me` links the owner taps: nothing is sent by BizIQ.
* **15 more modules (45 total):** BudgetIQ, PayablesIQ, ReconIQ, UnitEconomicsIQ, ScenarioIQ (with an interactive what-if slider page), InventoryIQ, ComplianceCalendarIQ, ProjectIQ, SubscriptionIQ, AppointmentIQ, RecipeIQ, AssetIQ, DiscountIQ, AnomalyIQ, SurveyIQ,
  with 9 new dataset types, aliases and planted-problem sample data. Thin data now produces a polite "needs ..." skip rather than a red failure.
* **People-data safeguards:** PeopleIQ no longer lists or scores named employees (counts of "strain signals" per department only; groups under 5 people are not reported; performance rating is not an input). AppointmentIQ no longer breaks results down by staff member. Every People page says "Signals, not judgements". There is deliberately **no CV-screening module**: ranking or filtering candidates was considered and rejected.
* **Consultant pack:** page *Consultant pack* produces a PDF (to read) and an Excel file (to recompute) with the findings, a confidence level and reasons for each section, method notes and a "what BizIQ did not check" page. People sections are off by default; tables that list individuals are always left out; optional best-effort hiding of customer, vendor, supplier and staff names; needs the `export` permission and is audit-logged.
* **Trust features:** a "How sure are we about this page?" panel (High/Medium/Low from rows, coverage, freshness and blank cells, with a source table), and a one-click export of everything the business has put in (CSV zip with a formula-injection guard).
* Original 11 analytics modules, decision layer with number guard, automation (dry-run email, drafts, scheduler), PDF board report.
* **30 registered modules** on the module contract: 14 finance/commercial/sales-ops/support modules, and 16 added in this round:
  ChurnIQ, PersonalisationIQ, WhatsAppIQ, MarketplaceIQ, StoreIQ, DistributorIQ (customers and channels); LogisticsIQ, ProductionIQ, MaintenanceIQ, TenderIQ, ESGIQ
  (operations); VendorRiskIQ, FraudIQ (risk); HiringIQ, SalaryIQ, PerformanceReviewIQ (people, gated by `view_people`). 10 new dataset types, each with column aliases
  for messy exports and synthetic sample data with planted problems. Every module has tests on hand-built cases, not only on the sample data.
* **PostgreSQL** backend (schema per organisation, COPY bulk load, append-only audit triggers, concurrent hash-chain writers), SQLite kept as the default for local use; SQLite-to-Postgres migration script; pg_dump backup/restore. 1M rows tested on Postgres.
* **Security:** AES-256-GCM envelope encryption (datasets, integration secrets, 2FA secrets), key rotation script, TOTP two-factor with recovery codes and org-wide enforcement, OIDC SSO mapping, session timeouts, stronger password policy (see docs/SECURITY.md).
* **Interface:** dark design system (`biziq/theme.py`), command-centre home with health gauge, module map and filterable findings feed, KPI cards, interactive chart explorer on every module page (chart type, axes, row count), consistent Plotly theme.
* Accounts, five roles, lockout, per-organisation data, audit log with tamper detection, approvals, connector presets.
* Live connectors, encrypted credential store, daily sync job, WhatsApp dispatch with approval gate, dry-run default.
* Real-data ingestion, tested on a public Superstore sales file (GitHub-hosted copy; Kaggle itself needs a login) as well as messy synthetic files.
* Operations: Dockerfile + compose (app, scheduler, daily backup, Postgres), health check, backup/restore, logging, upload limit.

## Not built or not proven
* **SalaryIQ no longer names anyone**: far-from-median salaries are reported as a count per department (groups of 5 or more only), with no employee IDs or individual salaries.
* **The consultant pack has not been read by a real accountant or consultant.** Layout and wording are untested with its intended readers. Name hiding is a text match, not a guarantee: a person should read the file before it is sent.
* **Nothing from this round has met a real shop or real company data.** BillBook, the 15 modules and the evidence levels were tested on synthetic data and hand-built cases only. Evidence thresholds are my judgement, not calibrated against anything.
* **BillBook screens are tested through Streamlit's harness, not on a phone or at a real counter.** One known gap: the harness cannot carry a customer name typed into the box (not already saved) across reruns, so that path (new customer on a credit bill) is covered at the logic layer only. Try it by hand first.
* **Hindi covers the billing screens only**, and the wording has not been checked by a native reader. Analysis pages are English.
* **BillBook has no GST invoice, no barcode scanning, no printer support, no multi-user conflict handling beyond duplicate-tap protection, and no offline mode.** It is not a replacement for a GST-compliant billing app yet.
* **Email verification and reset emails have only run against a fake SMTP server and dry-run.**
* **Heuristic modules:** CAC/payback, forecast, scenario, anomaly and compliance-calendar figures are rule-based estimates; ComplianceCalendarIQ dates must be checked by a CA before anyone relies on them.
* **Live integrations are untested against real accounts.** Shopify, Zoho Books, Tally, WhatsApp and the Tally-XML (myBillBook) import exist and pass
  tests on simulated responses only. Expect first-contact surprises (field names, permissions, API versions). No GST portal, marketplace or myBillBook live API.
* **Never run on a real customer's data.** The ingestion layer is built for real-world mess and tested on deliberately messy synthetic files, but no real export has gone through it. Expect new headings and formats to surface on day one.
* **Docker files are written but not built or run here** (no Docker in the build environment). The backup script and app are tested; the container wiring is not.
* **Domain rules need expert review.** GST due dates, return-recovery rates, target margins, deal-stage probabilities and SLA hours are
  labelled assumptions, not advice. Have a CA and a domain expert check them before anyone relies on them.
* **LLM and email paths** are only tested with mocks and dry-run. Real-key testing is still to do.
* **Security has had no third-party review.** The code is tested, but no penetration test or audit has been done, SSO has never been run against a real identity provider, the master key sits in an environment variable (no KMS/HSM), and files in `output/` are not encrypted. See docs/SECURITY.md for the full list. The DPDP Act applies to personal data, especially the people modules.
* **Postgres and encryption never run under Docker or at real concurrency beyond the tests** (6-thread audit test, 1M-row load).
* **New modules are rule-based and lightly validated.** Their thresholds (churn gaps, opt-out rates, fee terms, risk-score weights, tender price gaps, grid emission factor) are labelled assumptions. They were checked on synthetic data and one public sales file, never on a real business's data. ProductionIQ is "OEE-style" (plan-based, no ideal cycle times), ESGIQ's CO2e is an estimate and not a BRSR figure, and FraudIQ output is a list of indicators, not findings.
* **UI has been looked at in a headless browser and tested through Streamlit's test harness, but not user-tested** on phones, other browsers or with real users. There is no mobile app.
* **Deliberately not built as separate modules:** ReviewIQ (BrandIQ already analyses reviews), AccountIQ (ClientIQ covers account value), CompetitorIQ (PricingIQ uses competitor prices). Building them would duplicate working modules.
* **Not built:** deep manufacturing/industrial modules beyond ProductionIQ and MaintenanceIQ (they need cycle-time, BOM and quality data nobody has given us), and the maturity phase: mobile apps, white-labelling and billing/subscriptions.
