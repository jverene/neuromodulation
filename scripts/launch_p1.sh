#!/bin/bash
# P1/P1b launcher — preregistered in experiment_results/20261012_p1_p1b/.
# Auth: uses the local vastai CLI's stored API key (vastai set api-key ...).
# No key appears in this file or on the instance.
# Runs 18 parent trainings + the Appendix-I eval battery, ~6-14 h on one GPU.
set -uo pipefail
REPO_DIR="/Users/hjiang/Developer/neuromodulation"
SSH_OPTS="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25"
OUT_REL="experiment_results/20261012_p1_p1b"

echo "[1/5] checking auth + picking offer..."
vastai whoami >/dev/null 2>&1 || { echo "FAIL: vastai CLI has no valid API key (run: vastai set api-key <KEY>)"; exit 1; }
OFFER=$(vastai --raw search offers 'reliability>0.99 cuda_vers>=12.0 gpu_ram>=16 num_gpus=1 inet_down>=50 disk>=20' --order 'dph*' 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
offers = d if isinstance(d,list) else d.get('offers',[])
for o in offers:
    if o.get('gpu_name') in ('RTX 3090','RTX 4090','A100'):
        print(o['id'], o['gpu_name'], round(o['dph'],3)); break
")
[ -z "${OFFER:-}" ] && { echo "FAIL: no suitable offer"; exit 1; }
OID=$(echo $OFFER | cut -d' ' -f1); ONAME=$(echo $OFFER | cut -d' ' -f2); ODPH=$(echo $OFFER | cut -d' ' -f3)
echo "    picked: $OFFER"

echo "[2/5] creating instance $OID ($ONAME @ \$$ODPH/h)..."
vastai create instance "$OID" --disk 24 --image pytorch/pytorch:latest -q >/dev/null 2>&1 || { echo "FAIL: create"; exit 1; }
INST=$(vastai --raw show instances 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
if isinstance(d,dict) and 'instances' in d: d=d['instances']
rows=[x for x in d if x['label']=='']
import time
# newest instance running our image
best=max(rows,key=lambda x:x['id']) if rows else None
print(best['id'] if best else '')")
echo "    instance $INST; waiting for ssh..."
for i in $(seq 1 60); do
  URL=$(vastai ssh-url "$INST" 2>/dev/null) || true
  HP=$(echo "$URL" | sed 's|ssh://root@||'); H=$(echo "$HP" | cut -d: -f1); P=$(echo "$HP" | cut -d: -f2)
  [ -n "${H:-}" ] && [ "$H" != "" ] && ssh $SSH_OPTS -p "$P" root@"$H" "echo ok" 2>/dev/null | grep -q ok && break
  sleep 20
done
URL=$(vastai ssh-url "$INST"); HP=$(echo "$URL" | sed 's|ssh://root@||'); H=$(echo "$HP" | cut -d: -f1); P=$(echo "$HP" | cut -d: -f2)
echo "    ssh ready: $H:$P  (export P1_HOST=$H P1_PORT=$P P1_INST=$INST before re-running watcher-only)"

echo "[3/5] env setup..."
ssh $SSH_OPTS -p "$P" root@$H 'bash -s' <<'EOF'
cd /workspace
[ -d nca-mod ] || git clone -q https://github.com/jverene/neuromodulation.git nca-mod
cd nca-mod && git pull -q origin main
python3 -m venv /workspace/venv 2>/dev/null || conda create -y -n nca python=3.12 >/dev/null 2>&1
if [ -x /workspace/venv/bin/python ]; then PY=/workspace/venv/bin/python; else PY=/opt/conda/envs/nca/bin/python; fi
$PY -m pip install -q -U pip >/dev/null 2>&1
$PY -c "import jax" 2>/dev/null || $PY -m pip install -q -U "jax[cuda12]" -r requirements.txt >/tmp/pip.log 2>&1
$PY -c "import jax; print('JAX OK:', jax.devices())"
EOF

echo "[4/5] uploading driver + launching..."
cat > /tmp/drv_p1.sh <<'DRV'
#!/bin/bash
set -uo pipefail
cd /workspace/nca-mod
if [ -x /workspace/venv/bin/python ]; then PY=/workspace/venv/bin/python; else PY=/opt/conda/envs/nca/bin/python; fi
OUT=experiment_results/20261012_p1_p1b; mkdir -p $OUT/parents
echo "driver start $(date)"

train_one () {  # cfg out_pkl
  $PY -m src.train --config "$1" >/tmp/train_last.log 2>&1
  RD=$(ls -dt results/*_"$3"/ | head -1)
  cp "$RD/params.pkl" "$2" && echo "trained $2 from $RD"
}

echo "== smoke (50 steps) =="
sed -e 's/^seed: 0/seed: 900/' -e 's/learning_rate: 2.0e-3/learning_rate: 1.0e-3/' configs/e0_baseline.yaml > /tmp/smoke.yaml
$PY -m src.train --config /tmp/smoke.yaml --steps 50 >/dev/null 2>&1 && echo "smoke OK" || { echo "SMOKE FAIL"; touch /workspace/P1_FAIL; exit 1; }

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
  $PY -m src.posthoc_metrics --config /tmp/p1_eval_s$s.yaml --label p1_s$s --out $OUT/p1_pairs.csv
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
  $PY -m src.posthoc_metrics --config /tmp/p1b_eval_t$t.yaml --label p1b_t$t --out $OUT/p1b_pairs.csv
done

echo "P1/P1B COMPLETE $(date)"
touch /workspace/P1_DONE
sudo poweroff
DRV
cat > /tmp/wdt_p1.sh <<'WDT'
#!/bin/bash
# billing backstop: hard poweroff after 16h regardless of driver state
DEADLINE=$(( $(date +%s) + 57600 ))
while [ ! -f /workspace/P1_DONE ] && [ ! -f /workspace/P1_FAIL ]; do
  [ $(date +%s) -ge $DEADLINE ] && { echo "$(date) 16h cap" >> /workspace/self_stop.log; sudo poweroff; }
  sleep 60
done
WDT
scp -q $SSH_OPTS -P "$P" /tmp/drv_p1.sh /tmp/wdt_p1.sh root@$H:/workspace/
ssh $SSH_OPTS -p "$P" root@$H "chmod +x /workspace/drv_p1.sh /workspace/wdt_p1.sh
tmux kill-session -t p1 2>/dev/null; tmux kill-session -t wdt 2>/dev/null
tmux new-session -d -s p1 '/workspace/drv_p1.sh 2>&1 | tee /workspace/p1.log'
tmux new-session -d -s wdt '/workspace/wdt_p1.sh'
sleep 5; tmux ls"
echo "[4/5] LAUNCHED $(date)"

echo "[5/5] local watcher (pull + commit + push every 10 min; stops instance when done)..."
cd "$REPO_DIR"
while true; do
  sleep 600
  URL=$(vastai ssh-url "$INST" 2>/dev/null) || { echo "$(date) ssh-url failed"; continue; }
  HP=$(echo "$URL" | sed 's|ssh://root@||'); HH=$(echo "$HP" | cut -d: -f1); PP=$(echo "$HP" | cut -d: -f2)
  ssh $SSH_OPTS -o ConnectTimeout=20 -p "$PP" root@"$HH" "echo ok" >/dev/null 2>&1 || { echo "$(date) instance unreachable"; break; }
  mkdir -p "$OUT_REL/parents"
  scp -q $SSH_OPTS -P "$PP" "root@$HH:/workspace/nca-mod/$OUT_REL/parents/*.pkl" "$OUT_REL/parents/" 2>/dev/null
  scp -q $SSH_OPTS -P "$PP" "root@$HH:/workspace/nca-mod/$OUT_REL/*.csv" "$OUT_REL/" 2>/dev/null
  scp -q $SSH_OPTS -P "$PP" "root@$HH:/workspace/p1.log" "$OUT_REL/driver.log" 2>/dev/null
  git add "$OUT_REL" >/dev/null 2>&1
  git diff --cached --quiet >/dev/null 2>&1 || git commit -q -m "data: P1/P1b incremental (auto $(date +%H:%M))"
  git push -q origin main 2>/dev/null
  DONE=$(ssh $SSH_OPTS -o ConnectTimeout=20 -p "$PP" root@"$HH" "ls /workspace/P1_DONE /workspace/P1_FAIL 2>/dev/null | wc -l" 2>/dev/null)
  [ "${DONE:-0}" != "0" ] && [ -n "${DONE:-}" ] && { echo "RUN FINISHED $(date)"; break; }
done
vastai stop instance "$INST" >/dev/null 2>&1
echo "ALL DONE $(date)"
