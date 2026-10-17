#!/bin/bash
# Resume P1/P1b FULLY SELF-CONTAINED on the instance (2026-10-16 redesign):
# the Mac does a one-shot setup (start instance, upload scripts + deploy key,
# launch tmux) and can then disconnect. The VM itself:
#   - commits+pushes every completed training/eval to GitHub (deploy key,
#     repo-scoped, revocable: gh api repos/jverene/neuromodulation/keys)
#   - computes the preregistered decision rules into ANALYSIS.md at the end
#   - powers off on completion (or the watchdog's 22h cap -> push+poweroff)
# Progress is visible from anywhere as commits on GitHub.
# Usage: bash scripts/resume_p1.sh [instance_id]   (default 56183964)
set -uo pipefail
REPO_DIR="/Users/hjiang/Developer/neuromodulation"
INST="${1:-56183964}"
KEYLOC="$HOME/.config/neuromodulation/p1_deploy"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"

[ -f "$KEYLOC" ] || { echo "FAIL: deploy key missing at $KEYLOC"; exit 1; }

echo "[1/4] restarting instance $INST..."
vastai start instance "$INST" || { echo "FAIL: start"; exit 1; }
for i in $(seq 1 60); do
  URL=$(vastai ssh-url "$INST" 2>/dev/null) || true
  HP=$(echo "$URL" | sed 's|ssh://root@||'); H=$(echo "$HP" | cut -d: -f1); P=$(echo "$HP" | cut -d: -f2)
  [ -n "${H:-}" ] && ssh $SSH_OPTS -p "$P" root@"$H" "echo ok" 2>/dev/null | grep -q ok && break
  sleep 20
done
echo "    ssh: $H:$P"

echo "[2/4] env check..."
ssh $SSH_OPTS -p "$P" root@$H 'bash -s' <<'EOF'
cd /workspace/nca-mod && git pull -q origin main
if [ -x /workspace/venv/bin/python ]; then PY=/workspace/venv/bin/python; else PY=/opt/conda/envs/nca/bin/python; fi
$PY -c "import jax; print('JAX OK:', jax.devices())" || { echo "ENV BROKEN"; exit 1; }
EOF

echo "[3/4] uploading driver + watchdog + key, launching..."
cat > /tmp/drv_vm.sh <<'DRV'
#!/bin/bash
set -uo pipefail
cd /workspace/nca-mod
if [ -x /workspace/venv/bin/python ]; then PY=/workspace/venv/bin/python; else PY=/opt/conda/envs/nca/bin/python; fi
export GIT_SSH_COMMAND="ssh -i /workspace/.p1_deploy -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
git remote set-url origin git@github.com:jverene/neuromodulation.git
git config user.name "Bowen Jiang"; git config user.email "eternalaxholo28@gmail.com"
OUT=experiment_results/20261012_p1_p1b; mkdir -p $OUT/parents
echo "in-VM driver start $(date)"
push_all () {
  git add $OUT >/dev/null 2>&1
  git diff --cached --quiet >/dev/null 2>&1 || git commit -q -m "data: P1/P1b in-VM incremental (auto $(date +%H:%M))"
  git push -q origin main || echo "push failed (retry at next unit)"
}
train_one () {  # cfg out_pkl run_name ; skips if checkpoint exists; pushes each save
  if [ -s "$2" ]; then echo "skip $2 (exists)"; return 0; fi
  $PY -m src.train --config "$1" >/tmp/train_last.log 2>&1 || { echo "TRAIN FAIL $1"; return 1; }
  RD=$(ls -dt results/*_"$3"/ | head -1)
  cp "$RD/params.pkl" "$2" && echo "trained $2"
  push_all
}
echo "== P1: channel-exercise pairs (mod_noise true), seeds 0-4 =="
for s in 0 1 2 3 4; do
  sed -e "s/^seed: 0/seed: $s/" -e 's/learning_rate: 2.0e-3/learning_rate: 1.0e-3/' configs/e0_baseline.yaml > /tmp/p1_k0_s$s.yaml
  train_one /tmp/p1_k0_s$s.yaml $OUT/parents/p1_k0_s$s.pkl e0_baseline
  sed -e "s/^seed: 0/seed: $s/" -e 's/learning_rate: 2.0e-3/learning_rate: 1.0e-3/' -e 's/mod_noise: false/mod_noise: true/' configs/e0_channel_aware.yaml > /tmp/p1_k3n_s$s.yaml
  train_one /tmp/p1_k3n_s$s.yaml $OUT/parents/p1_k3n_s$s.pkl e0_channel_aware
  WS=$((200+s))
  sed -e "s/^seed: 0/seed: $WS/" \
      -e "s|results/20260724_123615_e0_channel_aware/params.pkl|$OUT/parents/p1_k3n_s$s.pkl|" \
      -e "s|results/20260724_065141_e0_baseline/params.pkl|$OUT/parents/p1_k0_s$s.pkl|" \
      configs/e2_hard.yaml > /tmp/p1_eval_s$s.yaml
  $PY -m src.posthoc_metrics --config /tmp/p1_eval_s$s.yaml --label p1_s$s --out $OUT/p1_pair_s$s.csv
  push_all
done
echo "== P1b: init-luck null pairs (mod_noise false), seeds 10-13 =="
for t in 10 11 12 13; do
  sed -e "s/^seed: 0/seed: $t/" -e 's/learning_rate: 2.0e-3/learning_rate: 1.0e-3/' configs/e0_baseline.yaml > /tmp/p1b_k0_t$t.yaml
  train_one /tmp/p1b_k0_t$t.yaml $OUT/parents/p1b_k0_t$t.pkl e0_baseline
  sed -e "s/^seed: 0/seed: $t/" -e 's/learning_rate: 2.0e-3/learning_rate: 1.0e-3/' configs/e0_channel_aware.yaml > /tmp/p1b_k3_t$t.yaml
  train_one /tmp/p1b_k3_t$t.yaml $OUT/parents/p1b_k3_t$t.pkl e0_channel_aware
  WS=$((300+t))
  sed -e "s/^seed: 0/seed: $WS/" \
      -e "s|results/20260724_123615_e0_channel_aware/params.pkl|$OUT/parents/p1b_k3_t$t.pkl|" \
      -e "s|results/20260724_065141_e0_baseline/params.pkl|$OUT/parents/p1b_k0_t$t.pkl|" \
      configs/e2_hard.yaml > /tmp/p1b_eval_t$t.yaml
  $PY -m src.posthoc_metrics --config /tmp/p1b_eval_t$t.yaml --label p1b_t$t --out $OUT/p1b_pair_t$t.csv
  push_all
done
echo "== analysis (preregistered decision rules, mechanical) =="
$PY - <<'PYEOF'
import csv, glob, statistics as st
from pathlib import Path
OUT = Path("experiment_results/20261012_p1_p1b")
def conds_for(fname):
    out = {}
    for f in glob.glob(str(OUT/fname)):
        for r in csv.DictReader(open(f)):
            out[r["condition"]] = {k: float(v) for k, v in r.items() if k not in ("label","condition")}
    return out
rows = []
gaps_p1, gaps_p1_orig, gaps_p1b, gaps_p1b_orig = [], [], [], []
for s in range(5):
    conds = conds_for(f"p1_pair_s{s}.csv")
    if len(conds) < 3:
        continue
    ga = conds["k0_own"]["aligned_hamming_mean"] - conds["k3_m0_shared"]["aligned_hamming_mean"]
    gp = conds["k0_shared"]["plain_hamming_mean"] - conds["k3_m0_shared"]["plain_hamming_mean"]
    gaps_p1.append(ga); gaps_p1_orig.append(gp)
    rows.append((lab, conds["k3_m0_shared"], conds["k0_shared"], conds["k0_own"], ga, gp))
for t in range(10, 14):
    conds = {}
    for f in glob.glob(str(OUT/f"p1b_pair_t{t}.csv")):
        for r in csv.DictReader(open(f)):
            conds[r["condition"]] = {k: float(v) for k, v in r.items() if k not in ("label","condition")}
    if len(conds) < 3:
        continue
    ga = conds["k0_own"]["aligned_hamming_mean"] - conds["k3_m0_shared"]["aligned_hamming_mean"]
    gp = conds["k0_shared"]["plain_hamming_mean"] - conds["k3_m0_shared"]["plain_hamming_mean"]
    gaps_p1b.append(ga); gaps_p1b_orig.append(gp)
    rows.append((f"p1b_t{t}", conds["k3_m0_shared"], conds["k0_shared"], conds["k0_own"], ga, gp))
lines = ["# P1/P1b analysis (mechanical, preregistered rules)", "",
         "| pair | K3 own (plain/aligned) | K0 shared (plain/aligned) | K0 own (plain/aligned) | g_aligned_own | g_plain_shared |",
         "|---|---|---|---|---|---|"]
for lab, a, b, c, ga, gp in rows:
    lines.append(f"| {lab} | {a['plain_hamming_mean']:.3f}/{a['aligned_hamming_mean']:.3f} (surv {a['survival_plain']:.2f}) "
                 f"| {b['plain_hamming_mean']:.3f}/{b['aligned_hamming_mean']:.3f} (surv {b['survival_plain']:.2f}) "
                 f"| {c['plain_hamming_mean']:.3f}/{c['aligned_hamming_mean']:.3f} (surv {c['survival_plain']:.2f}) "
                 f"| {ga:+.4f} | {gp:+.4f} |")
lines += ["", "## Decision rules (PREREGISTRATION.md)"]
if gaps_p1:
    pos = sum(g > 0 for g in gaps_p1); med = st.median(gaps_p1)
    verdict = "SUPPORTED" if (pos >= 4 and med > 0.005) else "NOT SUPPORTED"
    lines += [f"- P1 (n={len(gaps_p1)}): positive g_aligned_own in {pos}/{len(gaps_p1)}; median {med:+.4f} -> channel-experience effect {verdict}.",
              f"  original-protocol gaps (plain, shared): {[round(g,4) for g in gaps_p1_orig]}"]
if gaps_p1b:
    lines += [f"- P1b (n={len(gaps_p1b)}): g_aligned_own {[round(g,4) for g in gaps_p1b]}",
              f"  original-protocol gaps (plain, shared): {[round(g,4) for g in gaps_p1b_orig]}",
              f"  signs mixed: {sum(g>0 for g in gaps_p1b)}/{len(gaps_p1b)} positive"]
(OUT/"ANALYSIS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PYEOF
push_all
echo "P1/P1B COMPLETE (in-VM, self-managed) $(date)"
touch /workspace/P1_DONE
sleep 30
sudo poweroff
DRV
cat > /tmp/wdt_vm.sh <<'WDT'
#!/bin/bash
# billing backstop: at 22h push whatever exists and power off
START=$(date +%s)
while true; do
  sleep 120
  [ -f /workspace/P1_DONE ] && exit 0
  if [ $(date +%s) -ge $((START + 79200)) ]; then
    cd /workspace/nca-mod
    export GIT_SSH_COMMAND="ssh -i /workspace/.p1_deploy -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    git add experiment_results/20261012_p1_p1b >/dev/null 2>&1
    git commit -q -m "data: P1/P1b auto-stop at 22h cap (partial)" >/dev/null 2>&1
    git push -q origin main || true
    echo "$(date) 22h cap reached" >> /workspace/self_stop.log
    sudo poweroff
  fi
done
WDT
scp -q $SSH_OPTS -P "$P" /tmp/drv_vm.sh /tmp/wdt_vm.sh root@$H:/workspace/
scp -q $SSH_OPTS -P "$P" "$KEYLOC" root@$H:/workspace/.p1_deploy
ssh $SSH_OPTS -p "$P" root@$H "chmod +x /workspace/drv_vm.sh /workspace/wdt_vm.sh; chmod 600 /workspace/.p1_deploy
tmux kill-session -t p1 2>/dev/null; tmux kill-session -t wdt 2>/dev/null
tmux new-session -d -s p1 '/workspace/drv_vm.sh 2>&1 | tee -a /workspace/p1.log'
tmux new-session -d -s wdt '/workspace/wdt_vm.sh'
sleep 5; tmux ls; tail -3 /workspace/p1.log 2>/dev/null"
echo "[4/4] LAUNCHED $(date) — instance is self-managed (in-VM git push + poweroff)."
echo "Mac can disconnect. Watch progress as commits on GitHub; ping me at the end for the writeup."
