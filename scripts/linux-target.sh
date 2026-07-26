#!/usr/bin/env bash
# Bring a Linux host to the §6.1 measurement discipline and run the protocol.
#
# Scope note. On Fedora Asahi (Linux on Apple Silicon) this validates the
# INSTRUMENTATION, not the thesis: same silicon means the same bandwidth
# ratio r ~ 0.6 and the same absence of AMX. What it does buy:
#   - the perf_event_open backend actually executes (untested on macOS)
#   - isolcpus / pinned frequency / hugepages, so E3 and E4 stop fighting
#     the scheduler
#   - the designed-h vs counter-measured-h cross-check runs for real
# On a Sapphire Rapids / Xeon 6 host it validates the thesis as written.
set -euo pipefail

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "Host"
lscpu | grep -E '^(Architecture|Model name|CPU\(s\)|Thread|Core)' || true
lscpu -C 2>/dev/null || true
echo "AMX / VNNI: $(lscpu | grep -o 'amx_tile\|amx_int8\|amx_bf16\|avx512_vnni' | tr '\n' ' ')"
[ -r /proc/cpuinfo ] && grep -m1 'model name' /proc/cpuinfo || true

say "Counter permissions"
echo "perf_event_paranoid = $(cat /proc/sys/kernel/perf_event_paranoid)"
if [ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 0 ]; then
  echo "  -> need <= 0 for user-space counting:"
  echo "     echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi
if command -v perf >/dev/null; then
  echo "available L2 events:"; perf list 2>/dev/null | grep -iE 'l2_|mem_load' | head -20
else
  echo "  perf not installed:  sudo dnf install -y perf   # or linux-tools"
fi

say "Measurement discipline (§6.1)"
cat <<'TIPS'
  sudo cpupower frequency-set -g performance
  sudo cpupower idle-set -D 0
  echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
  # kernel cmdline, then reboot:
  #   isolcpus=8-15 nohz_full=8-15 rcu_nocbs=8-15
TIPS

say "Event encodings"
cat <<'EV'
  The defaults in kernels/ostir_kernel.c are INTEL (Golden Cove) and are
  wrong on any other uarch. Override without recompiling:
     export OSTIR_EVENT_L2_HIT=<ev>,<umask>
     export OSTIR_EVENT_L2_MISS=<ev>,<umask>
  On Apple Silicon under Asahi the PMU is exposed via the m1-pmu driver but
  the event numbers differ entirely from Intel's -- read `perf list` and map
  them before trusting any h_measured value. E4 will tell you if you got it
  wrong: it cross-checks counter-measured h against designed h and fails at
  MAD > 0.05. That cross-check is the whole point of running here.
EV

say "Run"
cd "$(dirname "$0")/.."
make -C kernels
CORE="${OSTIR_CORE:-8}"
echo "pinning to core $CORE (override with OSTIR_CORE=)"
if command -v numactl >/dev/null; then
  exec taskset -c "$CORE" numactl --membind=0 python3 run_all.py "$@"
else
  exec taskset -c "$CORE" python3 run_all.py "$@"
fi
