# SPMF

The sequential-pattern and rule generators call SPMF through `src/icu/v2/baselines/spmf.py`, which pins and validates the
parameter orders of **SPMF 2.64b** (they differ between versions and from the online documentation). Download
`spmf.jar` version 2.64b from https://www.philippe-fournier-viger.com/spmf/ and place it here as `tools/spmf/spmf.jar`.
It needs Java 21 (SPMF 2.67 needs Java 25). SPMF is distributed under the GPL v3 and is not redistributed in this
repository.
