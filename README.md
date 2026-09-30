# Transformer Self-Attention HLS Experiments

Independent course project exploring a single self-attention head: QK^T, softmax, and PV. Source variants investigate pipelining, unrolling, array partitioning, parallel reduction, precision and alternative softmax implementations.

- `attention.h`: shared interface (N=1024, DK=128).
- `attention_s*.cpp`: alternative implementations, not files to compile together.
- `testbench.cpp`, `testbench_s15.cpp`, `testbench_s20.cpp`: floating/fixed-point test drivers.
- `reports/`: archived HLS report excerpts with top-level timing, latency and resource summaries. The original generated solution scripts sometimes select a different source than their folder name suggests; report names alone do not establish an exact source-to-result mapping.

## Reproduce a variant

Use Vivado HLS 2019.1 with Xilinx HLS headers. From this directory:

```sh
vivado_hls -f run_hls.tcl
```

The provided script runs baseline C simulation and synthesis for xc7a200tfbg484-1 at a requested 10 ns clock. Edit the selected source/testbench to explore another variant, checking its interface and precision. Do not add all variants to one HLS solution because each defines `attention_head`.

Full C/RTL co-simulation and implementation are not automatically run by this script. Archived reports are historical evidence, not newly reproduced benchmarks.
