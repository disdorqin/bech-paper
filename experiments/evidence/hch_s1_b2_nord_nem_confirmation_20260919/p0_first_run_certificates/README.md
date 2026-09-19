# First P0 Host re-derivation — preserved, superseded

These are byte copies of the certificates written by `p0_hosts.py` on the first P0
pass (NEM_SA1 at 13:54, NORD_DK1 at 14:00 on 2026-09-19):

* `P0_HOSTS.NEM_SA1.json`, `P0_HOSTS.NORD_DK1.json` — `producer_sha256 = 3371BA2DC4B334FC…`,
  `all_exact = true`, `protected_final_windows_read = 0` for both markets.

They were moved aside, not deleted, because `p0_hosts.py` refuses to overwrite an
existing certificate (`[skip] … already exists`: one writer per path, repo
`AGENTS.md` section 8) and the final P0 chain has to be contemporaneous with the
**final** executor tree — the executor was edited after these certificates were
written (see `../aborted_attempt_1/README.md`), so P0 was re-run from scratch.

Nothing here is a result. Nothing here was produced by reading a `PROTECTED_FINAL`
value: the re-derivation materialises open-role windows only and asserts the
protected read count is zero before writing.

## The blob copies are not duplicated here

The eight `host_rederivation/<market>/<host>.pt` files were copied here as well
when the certificates were moved, then compared byte-for-byte against the second
run's blobs and **removed**: all eight were identical, so keeping a second 60 MB
copy would add no evidence. `BLOB_SHA256.txt` records the sha256 values the two
runs agreed on. The live blobs are in `../host_rederivation/`, and
`verification/verify_nn.py` (V3) re-checks each of them against its frozen
`FREEZE_MANIFEST.json` digest and against the sha recorded in the live certificate
— never against this directory, and never by timestamp.
