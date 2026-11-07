#!/bin/bash
# 一个实验的完整流程（README「实验流程」）：训练 → PredCls dump（±GT 类别）→ 与官方 RA 同口径评测 → 过关判断 → DONE。
# 用法（守护进程，一次一个实验；排队就在一个 bash -c 里顺序调用两次）：
#   cd <repo> && NAME=p50k_base_e4 DATA=RA-4M/data_proxy50k.yaml EPOCHS=4 \
#     setsid nohup bash tools/run_exp.sh > logs/p50k_base_e4.queue.log 2>&1 < /dev/null &
#   消融：YOLOVO_CAND_CE=0.5 NAME=p50k_candce_e4 REF_RUN=runs/test/p50k_base_e4 bash tools/run_exp.sh
# 产物：runs/test/$NAME/{results.csv,weights/}、logs/$NAME.log、reports/baseline_cmp/$NAME.{cls,nocls}.pt、
#       reports/baseline_cmp/$NAME.json、runs/test/$NAME/post/{gate.json,DONE}
# 停止：kill -- -<PGID>（PGID 写在 queue log 第一行；训练子进程同组）。
set -u
# ---------------- 配置区（环境变量可覆盖，默认值写在这里） ----------------
ROOT=/root/rivermind-data/YOLOVOmodal_3871538535
NAME=${NAME:?实验名，如 p50k_base_e4}
DATA=${DATA:-RA-4M/data_proxy50k.yaml}      # 50k 子集 + 全 10,370 词表 + 同一 1,024 图 full_monitor val；全量换 RA-4M/data_full_monitor.yaml
EPOCHS=${EPOCHS:-4}                          # 短跑 4；过关后 20（从头起，见 README）
BATCH=${BATCH:-32}                           # 4080 上 15.4 G 显存；卡被占时 16
PRIMARY=${PRIMARY:-weights/primary_dior20_best.pt}   # 一级模型（冻结），与 train.py 默认回收站路径 md5 相同
REF_RUN=${REF_RUN:-}                         # 参照 run 目录（同数据同轮数），空 = 不做 c2/c3，只看 c1
DO_PROTOCOL=${DO_PROTOCOL:-1}                # 1 = dump + 与官方 RA 同口径评测（要求 DATA 的 val 是 full_monitor、词表 10,370）
CMP=reports/baseline_cmp
RA_DUMP=$CMP/ra_official_vits16plus.pt
# 过关阈值（README；用户 2026-11-07 05:12 确认）
GATE_R50_GAIN=0.5                            # c2：pc_R@50 ≥ 参照 + 0.5 点，或
GATE_MR50_GAIN=1.0                           #     pc_mR@50 ≥ 参照 + 1.0 点
GATE_PAIR_ACC_TOL=0.5                        # c3：pc_pair_acc 不低于参照 0.5 点以上
PY=/opt/conda/bin/python
export PYTHONPATH=/tmp/yolovo-audit-numpy2
export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
# -------------------------------------------------------------------------
cd "$ROOT" || exit 1
RUN=runs/test/$NAME
OUT=$RUN/post
mkdir -p "$OUT" logs
rm -f "$OUT/DONE"
echo "pgid=$(ps -o pgid= -p $$ | tr -d ' ')  start $(date '+%F %T')  NAME=$NAME DATA=$DATA EPOCHS=$EPOCHS REF_RUN=${REF_RUN:-none}"
env | grep '^YOLOVO_' | sed 's/^/  env: /'
declare -A RC

# 1) 训练（前台子进程，与本脚本同进程组）
if [ -s "$RUN/results.csv" ] && [ "$(($(wc -l < "$RUN/results.csv") - 1))" -ge "$EPOCHS" ] && [ -s "$RUN/weights/best.pt" ]; then
  echo "[$(date '+%F %T')] results.csv 已有 $EPOCHS 轮，跳过训练（重跑请先删 $RUN）"; RC[train]=0
else
  t0=$(date +%s)
  YOLOVO_RELATE_DATA=$DATA YOLOVO_PRIMARY=$PRIMARY YOLOVO_EPOCHS=$EPOCHS YOLOVO_BATCH=$BATCH YOLOVO_NAME=$NAME YOLOVO_CACHE=0 \
    $PY train.py > "logs/$NAME.log" 2>&1
  RC[train]=$?
  echo "[$(date '+%F %T')] train rc=${RC[train]} ($(( ($(date +%s) - t0) / 60 )) min); results.csv rows: $(($(wc -l < "$RUN/results.csv" 2>/dev/null || echo 1) - 1))"
  tail -c 1500 "logs/$NAME.log" | tr '\r' '\n' | grep -v "^\s*$" | tail -n 3
fi
$PY - "$RUN/results.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ["epoch", "pc_R@50", "pc_mR@50", "pc_F1@50", "pc_pair_acc", "train/nce", "lr/pg1"]
print("  " + " ".join(c.rjust(12) for c in cols))
for r in rows:
    print("  " + " ".join((f"{100*float(r[c]):12.2f}" if c.startswith("pc_") else f"{float(r[c]):12.4g}" if c in r else "".rjust(12)) for c in cols))
PYEOF

# 2) dump + 3) 同口径评测
RC[dump]=skipped; RC[protocol]=skipped
if [ "${RC[train]}" -eq 0 ] && [ "$DO_PROTOCOL" = "1" ] && [ -s "$RUN/weights/best.pt" ]; then
  $PY tools/dump_predcls_ours.py "$RUN/weights/best.pt" RA-4M/data_full_monitor.yaml "$CMP/$NAME.cls.pt"            > "$OUT/dump_cls.log"   2>&1; r1=$?
  $PY tools/dump_predcls_ours.py "$RUN/weights/best.pt" RA-4M/data_full_monitor.yaml "$CMP/$NAME.nocls.pt" --no-cls > "$OUT/dump_nocls.log" 2>&1; r2=$?
  RC[dump]=$(( r1 | r2 ))
  echo "[$(date '+%F %T')] dump rc=${RC[dump]}  $(grep -h '^dumped' "$OUT/dump_cls.log" "$OUT/dump_nocls.log" | sed 's/.*validator/validator/' | tr '\n' ' ')"
  if [ "${RC[dump]}" -eq 0 ]; then
    refs=(); refn=()
    if [ -n "$REF_RUN" ] && [ -s "$CMP/$(basename "$REF_RUN").cls.pt" ]; then
      refs+=("$CMP/$(basename "$REF_RUN").cls.pt" "$CMP/$(basename "$REF_RUN").nocls.pt"); refn+=("ref_cls" "ref_nocls")
    fi
    $PY tools/sgg_protocol_eval.py "$RA_DUMP" "${refs[@]}" "$CMP/$NAME.cls.pt" "$CMP/$NAME.nocls.pt" \
        --names RA_official "${refn[@]}" "${NAME}_cls" "${NAME}_nocls" --json "$CMP/$NAME.json" 2>&1 | grep -v -i warn | tee "$OUT/protocol_eval.txt"
    RC[protocol]=${PIPESTATUS[0]}
  fi
fi

# 4) 过关判断（results.csv 口径：训练内验证器，与参照同 val）
RC[gate]=skipped
if [ "${RC[train]}" -eq 0 ]; then
  $PY - "$RUN/results.csv" "${REF_RUN:+$REF_RUN/results.csv}" "$EPOCHS" "$GATE_R50_GAIN" "$GATE_MR50_GAIN" "$GATE_PAIR_ACC_TOL" "$OUT/gate.json" <<'PYEOF'
import csv, json, sys
csv_path, ref_csv, E, g_r, g_mr, tol, out = sys.argv[1:8]
E, g_r, g_mr, tol = int(E), float(g_r), float(g_mr), float(tol)
rows = {int(r["epoch"]): r for r in csv.DictReader(open(csv_path))}
last = max(rows)
best = max(rows, key=lambda e: float(rows[e]["pc_R@50"]))
res = {"epochs_done": last, "best_epoch_by_pc_R@50": best,
       "pc_R@50": {e: round(100 * float(rows[e]["pc_R@50"]), 2) for e in rows},
       "pc_mR@50": {e: round(100 * float(rows[e]["pc_mR@50"]), 2) for e in rows},
       "pc_F1@50": {e: round(100 * float(rows[e]["pc_F1@50"]), 2) for e in rows},
       "pc_pair_acc": {e: round(100 * float(rows[e]["pc_pair_acc"]), 2) for e in rows}}
c1 = (E in rows and (E - 1) in rows and float(rows[E]["pc_R@50"]) > float(rows[E - 1]["pc_R@50"]))
res["c1_still_rising(pc_R@50 e%d > e%d)" % (E, E - 1)] = c1
if ref_csv:
    ref = {int(r["epoch"]): r for r in csv.DictReader(open(ref_csv))}
    rb = max(ref, key=lambda e: float(ref[e]["pc_R@50"]))
    dr = 100 * (float(rows[best]["pc_R@50"]) - float(ref[rb]["pc_R@50"]))
    dmr = 100 * (float(rows[best]["pc_mR@50"]) - float(ref[rb]["pc_mR@50"]))
    dpa = 100 * (float(rows[best]["pc_pair_acc"]) - float(ref[rb]["pc_pair_acc"]))
    c2 = dr >= g_r or dmr >= g_mr
    c3 = dpa >= -tol
    res.update({"ref": ref_csv, "ref_best_epoch": rb, "delta_pc_R@50": round(dr, 2), "delta_pc_mR@50": round(dmr, 2),
                "delta_pc_pair_acc": round(dpa, 2), "c2_improved": c2, "c3_no_regression": c3, "pass": bool(c1 and c2 and c3)})
else:
    res.update({"ref": None, "pass": bool(c1), "note": "no REF_RUN: only c1 evaluated"})
json.dump(res, open(out, "w"), indent=1)
print("GATE", json.dumps(res))
sys.exit(0 if res["pass"] else 3)
PYEOF
  RC[gate]=$?
  echo "[$(date '+%F %T')] gate rc=${RC[gate]} (0=过关, 3=不过关)"
fi

{
  echo "end $(date '+%F %T')"
  for k in train dump protocol gate; do echo "$k=${RC[$k]:-notrun}"; done
} | tee "$OUT/DONE"
