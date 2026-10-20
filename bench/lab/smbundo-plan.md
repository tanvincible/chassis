# The undo file on an SMB share

Plan written 2026-10-20, before the run. Workflow: `.github/workflows/smbundo.yml`.

The filesystems job (#133) failed once in two runs on an SMB share:
`test_backups_from_a_reader_while_the_writer_adds_and_flushes_hold_one_flush_each` panicked with
"Can't create ...index.chassis.undo ... Access is denied". #135 (849fe83) renames the undo file
away before removing it.

**Run:** on one `windows-latest` runner, with the temporary directory on an SMB share at
`\\localhost\chassis`, that test 30 times on `main` (a620e65) and 30 times on #135 (849fe83).

**Predictions:** `main` fails at least 3 of 30, each time with "Access is denied" on the undo
file; #135 fails none.

**Decision:** if #135 fails any, or `main` fails none (the test doesn't show the bug), #135 isn't
shown to fix it, and the cause is looked at again.

## Results

Not yet run.
