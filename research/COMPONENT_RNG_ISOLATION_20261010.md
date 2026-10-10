# Component prototype runtime isolation

Base: `08a59846d81c249f3560235382d5e8c69065e339`, the component-audit draft.
This amendment applies to that exploratory toolkit, not to the separately
delivered full FREM package or its retained results.

## Defect and change

The optional `latent_features` routine previously called `torch.manual_seed`,
which resets CPU and accelerator generators, and unconditionally changed the
process's intra-op thread count to two. Neither was restored after fitting or a
failure. Adding an optional diagnostic could therefore change later random draws
or other model initialization in the same process.

The routine now seeds only the CPU generator inside `torch.random.fork_rng`.
A CPU device context also makes model and tensor construction independent of a
caller's default device. Both contexts restore their prior state on return or
failure. Thread configuration belongs to the caller and is no longer changed.
The fitting equations, seed argument, dimensions, objective, optimizer, epochs,
EMA update and diagnostics are unchanged. This is sequential CPU isolation,
not a guarantee for simultaneous callers using the global random generator.

## Executed verification

- The final local repository suite passed **24 tests** on Python 3.12.14 and
  PyTorch 2.14.1+cpu, including the optional prototype tests. No tests skipped.
- On the unmodified base, the three defect regressions failed as expected:
  caller random-state preservation, exceptional-exit restoration, and avoiding
  accelerator reseeding. The seed-control comparison passed on that base.
- A separate old/new comparison at identical seed and two CPU threads retained
  exactly equal training features, evaluation features and diagnostic values.
- The new default-device test uses a `meta` context to verify CPU routing and
  restoration without an accelerator. The exception test fails after an
  optimizer callback has consumed CPU randomness.

Independent review identified the missing CPU-device enforcement and the old
workflow's lack of optional-dependency coverage. Both were corrected. The normal
workflow now collects the whole test directory. A separate CPU prototype job
installs explicit CPU PyTorch and verifies its import before collecting the
optional tests, so missing PyTorch cannot silently count as hosted coverage.
The hosted job outcome must be read on the published head independently of this
local receipt.

## Evidence boundary

These are deterministic correctness fixtures and failure injection. They do not
provide licensed market observations, an investment backtest, a confirmatory
comparison, a trained canonical FREM model, or evidence of alpha. Existing
scientific hypotheses, protected-data gates and retained results are unchanged.
