#!/bin/bash
# Defense study launcher — fully autonomous once started. Never destroys anything.
# Self-stop watchdog bounds billing: stops instance 48080807 on completion or 30h.
cd /Users/hjiang/Developer/neuromodulation
NEW=48080807; NEWKEY=0f9b0e37125e74ec97a0dbd2e10f8428006cc954f6371a202bbacf37e220d984
OLD=47872539
SSHO="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"

echo "[1/5] env setup on $NEW..."
NURL=$(vastai ssh-url $NEW); NHP=$(echo "$NURL"|sed 's|ssh://root@||'); NH=$(echo "$NHP"|cut -d: -f1); NP=$(echo "$NHP"|cut -d: -f2)
for t in 1 2 3 4 5 6 7 8 9 10; do U=$(ssh $SSHO -p $NP root@$NH "echo ok" 2>/dev/null); [ "$U" = ok ] && break; sleep 30; done
ssh $SSHO -p $NP root@$NH 'bash -s' <<'EOF'
cd /workspace; [ -d nca-mod ] || git clone -q https://github.com/jverene/neuromodulation.git nca-mod
cd nca-mod && git pull -q origin main
conda env list | grep -q "^nca " || conda create -y -n nca python=3.12 >/tmp/conda.log 2>&1
/opt/conda/envs/nca/bin/python -c "import jax" 2>/dev/null || /opt/conda/envs/nca/bin/pip install -q -U "jax[cuda12]" -r requirements.txt >/tmp/pip.log 2>&1
/opt/conda/envs/nca/bin/python -c "import jax; print('JAX:', jax.devices())"
EOF
echo "[1/5] env done"

echo "[2/5] parents from old instance $OLD (patient, up to 3h)..."
vastai start instance $OLD >/dev/null 2>&1
for i in $(seq 1 180); do
  S=$(vastai --raw show instances 2>/dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin)
if isinstance(d,dict) and 'instances' in d: d=d['instances']
print([x['actual_status'] for x in d if x['id']==$OLD][0])" 2>/dev/null)
  [ "$S" = running ] && break; sleep 60
done
OURL=$(vastai ssh-url $OLD); OHP=$(echo "$OURL"|sed 's|ssh://root@||'); OH=$(echo "$OHP"|cut -d: -f1); OP=$(echo "$OHP"|cut -d: -f2)
PDIR=results_local/per_parent_20260817/parents; mkdir -p $PDIR
scp -q $SSHO -P $OP root@$OH:/workspace/nca-mod/results/20260816_145911_e0_baseline/params.pkl $PDIR/k0_s0.pkl
scp -q $SSHO -P $OP root@$OH:/workspace/nca-mod/results/20260816_152635_e0_channel_aware/params.pkl $PDIR/k3_s0.pkl
for s in 1 2 3 4; do
  K0=$(ssh $SSHO -p $OP root@$OH "ls -dt /workspace/nca-mod/results/*_e0_baseline_s$s/params.pkl | head -1")
  K3=$(ssh $SSHO -p $OP root@$OH "ls -dt /workspace/nca-mod/results/*_e0_channel_aware_s$s/params.pkl | head -1")
  scp -q $SSHO -P $OP root@$OH:$K0 $PDIR/k0_s$s.pkl; scp -q $SSHO -P $OP root@$OH:$K3 $PDIR/k3_s$s.pkl
done
N=$(ls $PDIR/*.pkl 2>/dev/null | wc -l | tr -d ' '); vastai stop instance $OLD >/dev/null 2>&1
[ "$N" != 10 ] && { echo "FAIL: $N/10 parents"; exit 1; }
echo "[2/5] 10/10 parents, old stopped"

echo "[3/5] driver + watchdog..."
cat > /tmp/drv.sh <<'DRV'
#!/bin/bash
cd /workspace/nca-mod; PY=/opt/conda/envs/nca/bin/python
echo "start $(date); per (s,e) config seed=100*e+s; damage seeds index-based" > /workspace/defense_manifest.txt
for s in 0 1 2 3 4; do for e in 1 2; do
  SEED=$((100*e+s))
  sed -e "s/^seed: 0/seed: $SEED/" \
      -e "s|results/20260724_123615_e0_channel_aware/params.pkl|/workspace/parents/k3_s$s.pkl|" \
      -e "s|results/20260724_065141_e0_baseline/params.pkl|/workspace/parents/k0_s$s.pkl|" \
      configs/e2_hard.yaml > /workspace/def_s${s}_e${e}.yaml
  echo "=== SEED $s EVO $e ==="
  $PY -m src.evolve --config /workspace/def_s${s}_e${e}.yaml 2>&1 | tee /workspace/def_s${s}_e${e}.log
  EDIR=$(ls -dt results/*_e2_hard*/ | head -1)
  grep -q "k3_s$s" "$EDIR/config.yaml" || { echo "mismatch, skip"; continue; }
  mkdir -p results/defense_s${s}_e${e}
  cp "$EDIR/controller_params.pkl" results/defense_s${s}_e${e}/own_controller.pkl
  cp "$EDIR/evolve_metrics.csv" "$EDIR/metrics.csv" results/defense_s${s}_e${e}/ 2>/dev/null
  $PY -m src.zero_control --config /workspace/def_s${s}_e${e}.yaml --controller results/defense_s${s}_e${e}/own_controller.pkl --out results/defense_s${s}_e${e}/own 2>&1 | tail -14 | tee /workspace/def_zc_s${s}_e${e}.log
  $PY -m src.m_series --config /workspace/def_s${s}_e${e}.yaml --controller results/defense_s${s}_e${e}/own_controller.pkl --out results/defense_s${s}_e${e}/m_series.csv 2>&1 | tail -5 | tee /workspace/def_ms_s${s}_e${e}.log
  echo "SEED $s EVO $e DONE"
done; done
echo "DEFENSE STUDY COMPLETE"
DRV
cat > /tmp/wdt.sh <<WDT
#!/bin/bash
DEADLINE=$(( $(date +%s) + 108000 ))
echo "$(date) watchdog armed 30h" >> /workspace/self_stop.log
while true; do
  grep -q "DEFENSE STUDY COMPLETE" /workspace/defense.log 2>/dev/null && break
  [ \$(date +%s) -ge \$DEADLINE ] && break
  sleep 60
done
sleep 300
curl -s --max-time 30 -X PUT --url "https://console.vast.ai/api/v0/instances/$NEW/" \
  -H "Authorization: Bearer $NEWKEY" -H "Content-Type: application/json" \
  -d '{"state": "stopped"}' >> /workspace/self_stop.log 2>&1
echo " $(date) stopped" >> /workspace/self_stop.log
WDT
ssh $SSHO -p $NP root@$NH "mkdir -p /workspace/parents"
for s in 0 1 2 3 4; do scp -q $SSHO -P $NP $PDIR/k0_s$s.pkl $PDIR/k3_s$s.pkl root@$NH:/workspace/parents/; done
scp -q $SSHO -P $NP /tmp/drv.sh /tmp/wdt.sh root@$NH:/workspace/

echo "[4/5] launch..."
ssh $SSHO -p $NP root@$NH "chmod +x /workspace/drv.sh /workspace/wdt.sh
tmux kill-session -t defense 2>/dev/null; tmux kill-session -t selfstop 2>/dev/null
tmux new-session -d -s defense '/workspace/drv.sh 2>&1 | tee /workspace/defense.log'
tmux new-session -d -s selfstop '/workspace/wdt.sh'
sleep 10; tmux ls; tail -2 /workspace/defense.log 2>/dev/null"
echo "[4/5] LAUNCHED $(date)"

echo "[5/5] local watcher (pull + git push every 10 min; never stops anything)..."
while true; do
  sleep 600
  NURL=$(vastai ssh-url $NEW 2>/dev/null); NHP=$(echo "$NURL"|sed 's|ssh://root@||'); NH=$(echo "$NHP"|cut -d: -f1); NP=$(echo "$NHP"|cut -d: -f2)
  for d in $(ssh $SSHO -o ConnectTimeout=20 -p $NP root@$NH "ls -d /workspace/nca-mod/results/defense_s* 2>/dev/null" 2>/dev/null); do
    n=$(basename $d); mkdir -p results_local/defense/$n experiment_results/20260818_evoseed_defense/$n
    for f in own/zero_control.csv m_series.csv evolve_metrics.csv own_controller.pkl; do
      scp -q $SSHO -P $NP root@$NH:$d/$f results_local/defense/$n/$(basename $f) 2>/dev/null
    done
    cp results_local/defense/$n/*.csv experiment_results/20260818_evoseed_defense/$n/ 2>/dev/null
  done
  git add experiment_results/20260818_evoseed_defense >/dev/null 2>&1
  git diff --cached --quiet >/dev/null 2>&1 || git commit -q -m "data: defense incremental (auto $(date +%H:%M))"
  git push -q origin main 2>/dev/null
  DONE=$(ssh $SSHO -o ConnectTimeout=20 -p $NP root@$NH "grep -c 'DEFENSE STUDY COMPLETE' /workspace/defense.log 2>/dev/null" 2>/dev/null)
  [ "$DONE" = 1 ] && { echo "DEFENSE COMPLETE $(date)"; break; }
  ssh $SSHO -o ConnectTimeout=20 -p $NP root@$NH "echo ok" >/dev/null 2>&1 || echo "$(date) instance unreachable (watchdog owns billing)"
done
echo "ALL DONE"
