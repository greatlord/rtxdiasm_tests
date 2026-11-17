# rtxdiasm_tests

Unified test/module-test repository for rtxdisam and ACuda, split from the main rtxdisam repo.

## Layout
- `tests/rtxdisam/` — root rtxdisam regression and PTX/SASS corpora tests
  - `RegressionTests/`
  - `ptx_to_spv_success/`, `ptx_to_spv_error/`
  - `sass_to_ptx_success/`, `sass_to_ptx_error/`
- `tests/acuda/` - ACuda test suite (bench, integration, ptx, python, sass, unit, windows-kit)
- `tests/acuda/python-examples/` — ACuda Python example/integration scripts (moved from `src/acuda/python-test`)

## Integration
Consumed by the main `rtxdisam` repo as a git submodule at `rtxdiasm_modules/rtxdiasm_tests`, with build/test references rewired to this layout.

## Copyright and authorship
Copyright 2026 by Magnus Olsen.

This code had assistance by models DeepFlash 4.1, Muse Spark 1.3, and Iceman.

## License
MIT (see `LICENSE`).
