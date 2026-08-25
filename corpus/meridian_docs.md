# Meridian Platform Documentation

A fictional cloud platform. The corpus is fictional on purpose: every fact here is
verifiable *only* from this text, so a model cannot answer from pretraining and
the golden set measures retrieval and grounding rather than memorisation.

---

## doc:billing-overview

Meridian bills in one-second increments with a 60-second minimum per workload.
Invoices are issued on the first business day of each month and are due within 30
days. Accounts more than 45 days past due enter read-only mode: existing
workloads keep running, but no new workloads can be created. Meridian does not
offer refunds for unused reserved capacity, though reserved capacity can be
transferred between projects in the same organisation at no cost.

## doc:billing-tiers

There are three billing tiers. **Starter** costs $0 per month with a hard cap of
50 vCPU-hours; exceeding the cap suspends workloads rather than charging
overage. **Team** costs $400 per month and includes 2,000 vCPU-hours, with
overage billed at $0.045 per vCPU-hour. **Scale** costs $2,500 per month,
includes 20,000 vCPU-hours, and bills overage at $0.031 per vCPU-hour. Tier
changes take effect at the start of the next billing period; downgrades are
blocked if current usage exceeds the target tier's included capacity.

## doc:auth-tokens

Meridian issues two credential types. **Session tokens** are short-lived, expire
after 12 hours, and cannot be renewed - clients must re-authenticate. **Service
keys** are long-lived, do not expire, and are the only credential accepted by the
batch API. Service keys are shown exactly once at creation time and cannot be
retrieved afterwards. A service key can be scoped to a single project or to an
entire organisation; organisation-scoped keys require two approvers.

## doc:auth-rotation

Service keys should be rotated every 90 days. Rotation is a two-step process: a
new key is created while the old key stays valid, and the old key is revoked
after the caller confirms the new key is in use. Meridian sends a rotation
reminder at 75 days and marks a key `stale` in the console at 120 days. Stale
keys continue to work - Meridian never auto-revokes a credential, because a
silent revocation is worse than a stale key.

## doc:rate-limits

The control-plane API allows 600 requests per minute per organisation. The data
plane allows 10,000 requests per second per project. Exceeding either returns
HTTP 429 with a `Retry-After` header in seconds. Rate limits are enforced with a
token bucket that refills continuously; a burst of up to 1,200 control-plane
requests is tolerated if the bucket has accumulated capacity. Limits are per
organisation, not per key, so adding keys does not increase throughput.

## doc:regions

Meridian operates in five regions: us-east-2, us-west-1, eu-central-1,
ap-south-1, and sa-east-1. Only us-east-2 and eu-central-1 support GPU workloads.
Data does not move between regions automatically. Cross-region reads are billed
at $0.02 per GB; cross-region writes are not supported at all and must be
performed as an explicit export/import.

## doc:storage-classes

Three storage classes exist. **Hot** offers single-digit-millisecond reads and
costs $0.023 per GB-month. **Warm** costs $0.011 per GB-month with a 12-hour
minimum retention and first-byte latency under 500 ms. **Cold** costs $0.004 per
GB-month, has a 90-day minimum retention, and requires a restore request that
completes in 3 to 5 hours. Deleting a Cold object before 90 days still bills the
remaining days.

## doc:backup-policy

Automated backups run nightly at 02:00 in the region's local time and are
retained for 14 days on the Team tier and 35 days on the Scale tier. The Starter
tier has no automated backups. Backups are stored in the same region as the
source data. Point-in-time restore is available on the Scale tier only, with a
granularity of five minutes and a maximum lookback of seven days.

## doc:restore-procedure

Restoring from a backup creates a new workload; it never overwrites the source.
The restore target must be in the same region as the backup. A restore of under
100 GB typically completes in 20 minutes; larger restores scale roughly linearly.
During a restore the source workload continues serving traffic. There is no way
to restore a single table or object from a workload-level backup - the granularity
is the whole workload.

## doc:networking-private

Private networking attaches a workload to a customer-owned subnet. A workload can
belong to exactly one subnet and the subnet cannot be changed after creation;
moving a workload requires a restore into a new workload. Private workloads have
no public endpoint and are reachable only through a peering connection or the
Meridian tunnel agent. Peering connections take up to 15 minutes to become active.

## doc:tunnel-agent

The tunnel agent is a small daemon that establishes an outbound TLS connection to
Meridian, so no inbound firewall rule is required. It supports up to 200
concurrent streams per agent instance and reconnects with exponential backoff
starting at 1 second and capping at 60 seconds. Running two agent instances with
the same identity is supported and load-balances streams between them.

## doc:deploy-strategies

Meridian supports three deployment strategies. **Rolling** replaces instances in
batches of 25% by default. **Blue-green** provisions a full parallel environment
and switches traffic atomically, requiring double capacity for the duration.
**Canary** routes a configurable percentage of traffic to the new version, with
automatic promotion after a soak period. Canary is available on the Scale tier
only.

## doc:rollback

A rollback re-points traffic at the previous known-good revision and completes in
under 90 seconds regardless of workload size, because the previous revision is
kept warm for 24 hours after a deploy. After 24 hours the previous revision is
released and a rollback becomes a full redeploy, which takes as long as the
original deploy. Meridian keeps the last 10 revision manifests indefinitely even
after the warm copies are released.

## doc:observability-metrics

Metrics are scraped every 15 seconds and retained at full resolution for 7 days,
then downsampled to 5-minute resolution for 13 months. Custom metrics are limited
to 500 distinct series per workload; exceeding the limit drops the newest series
and raises a `series_limit_exceeded` event. Metric queries are limited to a
31-day window in a single request.

## doc:observability-logs

Logs are retained for 30 days on all paid tiers and 3 days on Starter. Log lines
longer than 256 KB are truncated with a `truncated: true` field. Log search
supports substring and field-equality predicates but not regular expressions.
Log export to customer-owned storage is available on Scale and runs every 5
minutes.

## doc:incident-severity

Incidents are classified S1 through S4. **S1** means a complete outage of a
production workload and carries a 15-minute response target. **S2** is degraded
production with a 1-hour target. **S3** is a non-production issue with a
next-business-day target. **S4** is a question or request with no response
target. Only S1 and S2 can be opened by phone; S3 and S4 must be filed in the
console.

## doc:sla

The Scale tier carries a 99.95% monthly uptime SLA; Team carries 99.9%; Starter
carries none. SLA credits are 10% of the monthly bill for uptime below the
target, rising to 25% below 99.0% and 50% below 95.0%. Credits must be claimed
within 30 days of the incident and are applied to a future invoice; they are
never paid in cash. Scheduled maintenance is excluded from the uptime
calculation.

## doc:maintenance-windows

Each region has a weekly four-hour maintenance window. Customers on Team and
Scale can shift the window by up to 12 hours but cannot disable it. Meridian
gives 7 days' notice for disruptive maintenance and 24 hours for non-disruptive
maintenance. Emergency security maintenance may occur with no notice and is still
excluded from SLA calculations.

## doc:quotas

Default quotas per project: 100 workloads, 50 subnets, 20 peering connections,
and 5,000 service keys. Quota increases are requested in the console and are
usually granted within one business day; GPU quota increases require a capacity
check and can take up to five business days. Quotas are per project and cannot be
pooled across projects.

## doc:gpu-workloads

GPU workloads are available in us-east-2 and eu-central-1 only. Three instance
shapes exist: `g1` (1 GPU, 16 GB), `g4` (4 GPUs, 64 GB), and `g8` (8 GPUs,
128 GB). GPU workloads cannot use the Warm or Cold storage classes for their
primary volume. Preemptible GPU capacity is 60% cheaper and can be reclaimed with
a 30-second notice; preemptible workloads are not eligible for the SLA.

## doc:data-residency

Data residency guarantees are available in eu-central-1 and ap-south-1. When
residency is enabled, backups, logs, and metrics all stay in-region, and the
support team cannot access workload contents without a per-incident approval
recorded in the audit log. Enabling residency disables cross-region reads
entirely and cannot be turned off once enabled for a project.

## doc:audit-log

The audit log records control-plane actions only; data-plane reads and writes are
never recorded. Entries are immutable and retained for 400 days. The audit log
can be streamed to a customer-owned endpoint with at-least-once delivery, so
consumers must deduplicate on the `event_id` field. Audit log delivery lag is
typically under 60 seconds and is not covered by any SLA.

## doc:support-tiers

Support is included at all tiers but with different scope. Starter gets community
support only. Team gets business-hours support in English. Scale gets 24/7
support in English and Japanese, plus a named technical account manager once
annual spend exceeds $250,000. Support does not cover application code debugging
at any tier.

## doc:migration-limits

The bulk import API accepts files up to 5 TB per job and up to 10 concurrent jobs
per project. Imports are idempotent on the `object_key` field. An import that
fails partway leaves already-imported objects in place; re-running the job skips
them. There is no rollback for a completed import - undoing one means deleting
the objects explicitly.
