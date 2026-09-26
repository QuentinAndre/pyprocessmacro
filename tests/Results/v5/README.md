# PROCESS 5 comparison files

Output of Andrew F. Hayes's PROCESS for R, version 5.0 (released June 2025), run on the data in
`tests/Data` by `tests/fixtures/regenerate.py`. The first two lines of each file record the generation
date and the exact `process()` call. The files beside this directory in `tests/Results` were produced
with PROCESS 2.16 for SPSS in 2017 and remain the accuracy reference for the 2.x releases; these are the
reference for the PROCESS 5 parity work planned for 3.0 (see issue #82).

Options match the accuracy tests: 5000 bootstrap resamples, seed 123456, total and contrast effects,
HC3 standard errors, `intprobe=1` so that conditional effects are always printed (except for Model 3 with the
binary outcome, where PROCESS 5.0 fails to probe the three-way interaction and keeps its default); effect sizes for models 4 and 6
with a continuous outcome. Intervals are PROCESS 5's
default, percentile bootstrap. PROCESS 5 has only the moderators W and Z, so the models of the 2.16
numbering that need three or four moderators (23 to 27 and 30 to 57) have no file here, and neither does
model 74, which PROCESS 5 no longer defines.

These files are output, not the macro. PROCESS itself is free to download from
https://www.afhayes.com/download.html for personal use and may not be redistributed, so `process.R` is
never part of this repository; the generator reads its location from the `PROCESS_R` environment variable.
