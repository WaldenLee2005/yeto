# Performance checks for arXiv 2604.21428

This note maps Decoupled DiLoCo's performance claims to things Yeto can
test. It is intentionally operational: use it before claiming that a change
improves or preserves performance.

Paper axes worth checking:

- **Quality parity**: asynchronous Decoupled DiLoCo should stay close to a
  synchronous baseline at the same token budget.
- **Goodput under stragglers**: learners should keep taking inner steps while
  sync rounds gather quorum, wait through grace, merge, and broadcast.
- **Adaptive grace**: the grace window should follow Eq. 3,
  `gamma * (tau * step_time - quorum_time - sync_time)`, capped by
  `--grace-ms`.
- **Quorum behavior**: a round should merge after at least `K` responders and
  should not wait for dead or slow learners beyond the grace budget.
- **Token-weighted merge**: learner contribution should follow
  `c_tokens * (c_tokens / c_steps)`, so faster/high-token learners contribute
  more without giving stale, many-step deltas too much weight.
- **Pipelining**: default `--pipeline 2` should overlap sync work with learner
  compute; `--pipeline 1` is the serial control arm.
- **Transport/compression**: q4 learner pushes should reduce network pressure
  without a large eval-loss regression.

## Local quality benchmark

Use `scripts/compare_diloco.py` for paper-style ML-quality checks. Start with a
dry run:

```bash
python scripts/compare_diloco.py --data smoke_chat.jsonl --settings all --dry-run
```

Then run a small real comparison. On Apple Silicon:

```bash
python scripts/compare_diloco.py \
  --model lfm25-230m \
  --data smoke_chat.jsonl \
  --token-budget 50000 \
  --settings m2,m2h24,serial,alpha0,q4,noheloco,strided,avg \
  --device mps
```

On an NVIDIA box, use `--device cuda`. For a tighter signal, increase
`--token-budget` and use a real chat dataset instead of `smoke_chat.jsonl`.

How to read the report:

- `baseline` is the synchronous reference.
- `m2h24` is the main async check: it throttles syncs to the paper's
  design-point interval, about H=24 inner steps per fragment.
- `m2` is intentionally unthrottled in the comparison harness; on localhost it
  can sync too often and over-drive the outer optimizer.
- `serial` isolates the cost of disabling pipelined rounds.
- `q4` isolates wire compression.
- `alpha0`, `noheloco`, `strided`, and `avg` are ablation/debug arms.

Expected small-run result: the exact loss numbers will be noisy, but `m2h24`
should be much closer to `baseline` than an unthrottled localhost `m2` run if
the bottleneck is sync frequency rather than the async algorithm itself.

### Example smoke result

The following command is a fast harness check, not a statistically meaningful
quality result:

```bash
HF_HOME=/private/tmp/yeto-hf-home \
HF_DATASETS_CACHE=/private/tmp/yeto-hf-datasets \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python scripts/compare_diloco.py \
  --model lfm25-230m \
  --data /private/tmp/yeto-perf-chat.jsonl \
  --token-budget 512 \
  --settings m2,m2h24,serial,q4 \
  --seq-len 32 \
  --micro-batch-size 1 \
  --eval-rows 8 \
  --max-rows 40 \
  --device cpu
```

One local CPU run completed all selected arms and emitted the paper-style
system metrics from each syncer tape:

| arm | M | wall (s) | eval loss/token | delta vs baseline |
|---|---:|---:|---:|---:|
| base (untrained) | - | 0 | 6.2290 | - |
| baseline (sync) | 1 | 11 | 1.6705 | - |
| m2 | 2 | 9 | 3.7415 | +123.98% |
| m2h24 | 2 | 9 | 5.1104 | +205.93% |
| serial | 2 | 9 | 3.7678 | +125.55% |
| q4 | 2 | 9 | 3.5732 | +113.90% |

| arm | rounds | full quorum | missed grace | participation | avg responders | avg round ms | p95 round ms | tokens/s | steps/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m2 | 14 | 13 | 0 | 96.4% | 1.9 | 706.9 | 1622.0 | 107.5 | 3.4 |
| m2h24 | 5 | 5 | 0 | 100.0% | 2.0 | 849.6 | 1335.0 | 52.2 | 1.6 |
| serial | 7 | 6 | 0 | 92.9% | 1.9 | 654.1 | 1284.0 | 130.3 | 4.1 |
| q4 | 14 | 13 | 0 | 96.4% | 1.9 | 665.6 | 1423.0 | 112.9 | 3.5 |

| arm | node | responses | tokens | steps | contribution |
|---|---:|---:|---:|---:|---:|
| m2 | 0 | 13 | 480 | 15 | 48.4% |
| m2 | 1 | 14 | 512 | 16 | 51.6% |
| m2h24 | 0 | 5 | 224 | 7 | 50.0% |
| m2h24 | 1 | 5 | 224 | 7 | 50.0% |
| serial | 0 | 6 | 512 | 16 | 45.7% |
| serial | 1 | 7 | 608 | 19 | 54.3% |
| q4 | 0 | 13 | 480 | 15 | 48.4% |
| q4 | 1 | 14 | 512 | 16 | 51.6% |

This validates that the benchmark harness runs the synchronous baseline,
pipelined async, H-targeted async, serial, and q4 wire-format paths end to end,
and that it reports participation, responder balance, contribution, latency,
and token/step throughput. It should not be used as evidence of model quality
because the dataset is repeated smoke rows and the budget is only 512 tokens
per arm.

The report also has columns for average quorum, grace, and sync milliseconds.
Those are populated when the richer event-tape schema from the rendezvous
status telemetry change is present; older tapes leave them as `n/a`.

## Rendezvous and goodput smoke

Run a short heterogeneous or multi-process syncer smoke with an event tape:

```bash
syncer/target/release/yeto-syncer \
  --port 29400 \
  --learners 2 \
  --quorum 1 \
  --grace-ms 200 \
  --total-steps 8 \
  --fragments 4 \
  --event-tape /tmp/yeto-events.jsonl
```

Start two learners against that syncer. The earlier Mac MLX + Windows CUDA
smoke is a good version of this test because it exercises heterogeneous
learner speeds.

Check the tape for:

- every record has at least `K` responders;
- slow/missing learners do not prevent global steps from advancing;
- responder weights match `c_tokens^2 / c_steps`;
- contribution share roughly follows cumulative weights;
- round latency stays inside the expected quorum/grace/sync budget;
- both learners save outputs at the final global step.

With the rendezvous-status branch applied, this becomes:

```bash
yeto status --tape /tmp/yeto-events.jsonl
```

Without that branch, inspect the JSONL directly:

```bash
tail -n 8 /tmp/yeto-events.jsonl
```

## Large-scale checks not covered locally

The paper's biggest goodput claims come from many simulated chips, failures,
and elastic recovery. A Mac + one NVIDIA Windows machine can validate the
protocol, weighting, and heterogeneity path, but it does not validate:

- AWS/SkyPilot provisioning behavior;
- large-fleet interruption rates;
- multi-region spot churn;
- long-running checkpoint/recovery economics.

If those matter for a report, run the same tape/status checks on a cloud fleet
and include the cost approval explicitly.
