The production ABI builds only `solver_bridge.cpp` and `proposed_dp.cpp`.
`common.hpp` and `proposed_dp.cpp` were copied byte for byte from the validated
standalone solver during the package migration; the benchmark keeps its own
independent files. Production has no import or build dependency on experiments.

Original and migrated SHA256 values:

- `common.hpp`: `22587378b57ede5bcd300975b716d8b51b6fb4d35d88e33fab1787c3364ccdbe`
- `proposed_dp.cpp`: `09ee4f8ae11d25d67ba719942e7a4e6ba05ee673a0b228e90c6ab8a7343f64ce`

The original recurrence, conservative quantization, tie order, backtracking and
status handling are preserved. The C ABI supplies production options explicitly,
including the default 250,000 bit/s grid; no benchmark is run when loading it.
