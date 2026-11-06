#!/bin/bash
# 全量实验 glp_full_e4_resume 的收尾流水线（守护进程，GPU 空出后自动接力）。README「实验流程」的第一次执行：
#   1) 等训练主进程退出
#   2) best/last 权重的 PredCls dump（GT 框 ± GT 类别，1,024 张 full_monitor val）
#   3) 与官方 RelateAnything 权重同口径评测（tools/sgg_protocol_eval.py）
#   4) 过关判断（README 三条；本 run 没有同轮数参照，用 epoch 2 快照当参照）→ 过关就起 20 轮 glp_full_e20
#   5) BARV 难负例 cache 重建（barv.prepare）→ 几何 LR / 视觉 scorer 对照（barv.run_contact）
# 启动：cd <repo> && mkdir -p runs/test/glp_full_e4_resume/post && \
#       setsid nohup bash tools/post_full_e4.sh > runs/test/glp_full_e4_resume/post/run.log 2>&1 < /dev/null &
# 进度：tail -f runs/test/glp_full_e4_resume/post/run.log     停止：kill -- -<PGID>（PGID 见 run.log 第一行）
# 完成标记：runs/test/glp_full_e4_resume/post/DONE（每步退出码 + 时间；任一步失败后续依赖步骤跳过）
# 注意：bash 按偏移量边读边执行脚本，运行中不要原地编辑本文件；要改就 kill 掉重启。
set -u
# ---------------- 配置区 ----------------
ROOT=/root/rivermind-data/YOLOVOmodal_3871538535
TRAIN_PID=${TRAIN_PID:-82761}        # glp_full_e4_resume 的 python train.py 主进程（cmdline 含 train.py 才算活着）；已退出则直接收尾
RUN=runs/test/glp_full_e4_resume
OUT=$RUN/post                        # 日志 / DONE 标记都放这里（数据盘，不放 /tmp）
DATA=RA-4M/data_full_monitor.yaml    # 与 reports/baseline_cmp 同一 1,024 张 val、原始 10,370 标签
CMP=reports/baseline_cmp             # 官方 RA dump（ra_official_vits16plus.pt）与 epoch1/2 dump 已在这里
POLL_SEC=60                          # 等训练退出的轮询间隔；训练 1 epoch ≈ 3.9 h，没必要更密
# 过关标准（README「实验流程」；用户 2026-11-07 05:12 确认）。本 run 无同轮数参照，参照 = epoch 2 快照 protocol_eval_full_monitor_ep2.json
GATE_REF_NOCLS_R50=24.35             # epoch 2 只给框 R@50；e4 不低于它 = 还在向官方 26.61 逼近
GATE_REF_CLS_PAIR_ACC=54.93          # epoch 2 框+类别 pair_acc；允许下降 GATE_PAIR_ACC_TOL
GATE_PAIR_ACC_TOL=0.5
# 过关后起的 20 轮长跑：同配置、从头起（lr 调度按总轮数算，不续 4 轮权重），见 README
E20_NAME=glp_full_e20
E20_ENV="YOLOVO_RELATE_DATA=RA-4M/data_full_monitor.yaml YOLOVO_PRIMARY=weights/primary_dior20_best.pt YOLOVO_EPOCHS=20 YOLOVO_NAME=$E20_NAME YOLOVO_CACHE=0"
# BARV
PSG_DONE=/root/rivermind-data/psg_hf/CONVERT_DONE   # barv.hf_psg 转换完成标记，内容 exit=0 才进入 BARV 步骤
BARV_CFG=configs/barv_psg_contact.yaml              # 难负例配置；cache 写 data/barv_cache_contact（不覆盖 pilot）
BARV_OUT=runs/barv_contact/seed42                   # run_contact 输出（几何 LR / no_visual / 视觉 scorer + 按来源 AUROC）
PY=/opt/conda/bin/python
export PYTHONPATH=/tmp/yolovo-audit-numpy2                                   # numpy 2.2.6 overlay，必须
export LD_LIBRARY_PATH=/root/rivermind-data/cudnn9/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}   # conda torch 缺 libcudnn.so.9
export CUDA_VISIBLE_DEVICES=0
# ----------------------------------------
cd "$ROOT" || exit 1
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo "pgid=$(ps -o pgid= -p $$ | tr -d ' ')  start $(date '+%F %T')"
declare -A RC

# 1) 等训练退出（校验 cmdline，防 PID 复用后死等）
step1_start=$(date +%s)
while [ -r /proc/$TRAIN_PID/cmdline ] && tr '\0' ' ' < /proc/$TRAIN_PID/cmdline | grep -q "train.py"; do
  n=$(( ($(date +%s) - step1_start) / POLL_SEC ))
  if [ $((n % 30)) -eq 0 ]; then
    echo "[$(date '+%F %T')] waiting train pid=$TRAIN_PID; $(tail -c 300 logs/full_e4_resume.log | tr '\r' '\n' | tail -n 1 | cut -c1-140)"
  fi
  sleep "$POLL_SEC"
done
echo "[$(date '+%F %T')] train process gone. results.csv rows: $(($(wc -l < $RUN/results.csv) - 1))"
echo "--- train log tail ---"; tail -c 2000 logs/full_e4_resume.log | tr '\r' '\n' | tail -n 6; echo "----------------------"
ls -la "$RUN/weights/"
RC[wait]=0
[ -s "$RUN/weights/best.pt" ] || { echo "best.pt missing, abort"; RC[wait]=1; }

# 2) PredCls dump：best（±GT 类别）、last（GT 类别）
if [ "${RC[wait]}" -eq 0 ]; then
  $PY tools/dump_predcls_ours.py "$RUN/weights/best.pt" "$DATA" "$CMP/ours_full_e4_best_cls.pt"            > "$OUT/dump_best_cls.log"   2>&1; RC[dump_best_cls]=$?
  $PY tools/dump_predcls_ours.py "$RUN/weights/best.pt" "$DATA" "$CMP/ours_full_e4_best_nocls.pt" --no-cls > "$OUT/dump_best_nocls.log" 2>&1; RC[dump_best_nocls]=$?
  $PY tools/dump_predcls_ours.py "$RUN/weights/last.pt" "$DATA" "$CMP/ours_full_e4_last_cls.pt"            > "$OUT/dump_last_cls.log"   2>&1; RC[dump_last_cls]=$?
  for k in dump_best_cls dump_best_nocls dump_last_cls; do echo "[$(date '+%F %T')] $k rc=${RC[$k]}  $(grep -h '^dumped' "$OUT/$k.log" | tail -n 1)"; done
fi

# 3) 同口径评测（官方 RA + epoch1 + epoch2 快照 + 本次）
if [ "${RC[dump_best_cls]:-1}" -eq 0 ] && [ "${RC[dump_best_nocls]:-1}" -eq 0 ] && [ "${RC[dump_last_cls]:-1}" -eq 0 ]; then
  $PY tools/sgg_protocol_eval.py \
      "$CMP/ra_official_vits16plus.pt" "$CMP/ours_full_e1_cls.pt" "$CMP/ours_full_e1_nocls.pt" \
      "$CMP/ours_full_ep2_cls.pt" "$CMP/ours_full_ep2_nocls.pt" \
      "$CMP/ours_full_e4_best_cls.pt" "$CMP/ours_full_e4_best_nocls.pt" "$CMP/ours_full_e4_last_cls.pt" \
      --names RA_official ours_ep1_cls ours_ep1_nocls ours_ep2_cls ours_ep2_nocls ours_e4best_cls ours_e4best_nocls ours_e4last_cls \
      --json "$CMP/protocol_eval_full_monitor_e4.json" 2>&1 | grep -v -i warn | tee "$OUT/protocol_eval.txt"
  RC[protocol_eval]=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] protocol_eval rc=${RC[protocol_eval]}"
else
  echo "[$(date '+%F %T')] protocol_eval skipped (dump failed)"; RC[protocol_eval]=skipped
fi

# 4) 过关判断 → 起 20 轮
RC[gate]=skipped; RC[launch_e20]=skipped
if [ "${RC[protocol_eval]}" = "0" ]; then
  $PY - "$RUN/results.csv" "$CMP/protocol_eval_full_monitor_e4.json" "$GATE_REF_NOCLS_R50" "$GATE_REF_CLS_PAIR_ACC" "$GATE_PAIR_ACC_TOL" "$OUT/gate.json" <<'PYEOF'
import csv, json, sys
csv_path, json_path, ref_nocls_r50, ref_pair_acc, tol, out = sys.argv[1:7]
ref_nocls_r50, ref_pair_acc, tol = float(ref_nocls_r50), float(ref_pair_acc), float(tol)
rows = {int(r["epoch"]): r for r in csv.DictReader(open(csv_path))}
ev = json.load(open(json_path))
pc = {e: float(rows[e]["pc_R@50"]) for e in rows}
c1 = (3 in pc and 4 in pc and pc[4] > pc[3])                      # 第 4 轮还在涨
e4_nocls_r50 = 100 * ev["ours_e4best_nocls"]["R@50"]
e4_cls_pair = 100 * ev["ours_e4best_cls"]["pair_acc"]
c2 = e4_nocls_r50 >= ref_nocls_r50                                # 只给框 R@50 不低于 epoch 2 快照（继续逼近官方 26.61）
c3 = e4_cls_pair >= ref_pair_acc - tol                            # pair_acc 没退化
res = {"pc_R@50_by_epoch": pc, "c1_still_rising": c1,
       "e4best_nocls_R@50": e4_nocls_r50, "ref_nocls_R@50": ref_nocls_r50, "c2_nocls_not_worse": c2,
       "e4best_cls_pair_acc": e4_cls_pair, "ref_cls_pair_acc": ref_pair_acc, "tol": tol, "c3_pair_acc_ok": c3,
       "pass": bool(c1 and c2 and c3)}
json.dump(res, open(out, "w"), indent=1)
print("GATE", json.dumps(res))
sys.exit(0 if res["pass"] else 3)
PYEOF
  RC[gate]=$?
  echo "[$(date '+%F %T')] gate rc=${RC[gate]} (0=过关, 3=不过关, 其他=脚本出错)"
  if [ "${RC[gate]}" -eq 0 ]; then
    env $E20_ENV setsid nohup bash -c 'echo $$ > logs/'$E20_NAME'.pid; exec '$PY' train.py' > "logs/$E20_NAME.log" 2>&1 < /dev/null &
    sleep 90
    e20_pid=$(cat "logs/$E20_NAME.pid" 2>/dev/null)
    if [ -n "$e20_pid" ] && [ -r /proc/$e20_pid/cmdline ]; then
      RC[launch_e20]=0
      echo "[$(date '+%F %T')] launched $E20_NAME pid=$e20_pid pgid=$(ps -o pgid= -p $e20_pid | tr -d ' ') log=logs/$E20_NAME.log"
    else
      RC[launch_e20]=1
      echo "[$(date '+%F %T')] $E20_NAME failed to start; log tail:"; tail -n 20 "logs/$E20_NAME.log"
    fi
  fi
fi

# 5) BARV 难负例 cache（需要 GPU 抽特征；resnet18 很轻，可与 e20 并行）→ 对照实验
if [ -s "$PSG_DONE" ] && grep -q "^exit=0" "$PSG_DONE"; then
  $PY -m barv.prepare --config "$BARV_CFG" > "$OUT/barv_prepare.log" 2>&1; RC[barv_prepare]=$?
  echo "[$(date '+%F %T')] barv_prepare rc=${RC[barv_prepare]}  $(grep -v -i warn "$OUT/barv_prepare.log" | tail -n 2 | cut -c1-300)"
else
  echo "[$(date '+%F %T')] barv_prepare skipped: $PSG_DONE missing or exit!=0 ($(cat "$PSG_DONE" 2>/dev/null))"; RC[barv_prepare]=skipped
fi
if [ "${RC[barv_prepare]}" = "0" ] && [ -f barv/run_contact.py ]; then
  $PY -m barv.run_contact --config "$BARV_CFG" --out "$BARV_OUT" > "$OUT/barv_run_contact.log" 2>&1; RC[barv_run_contact]=$?
  echo "[$(date '+%F %T')] barv_run_contact rc=${RC[barv_run_contact]}"; grep -v -i warn "$OUT/barv_run_contact.log" | tail -n 25
else
  echo "[$(date '+%F %T')] barv_run_contact skipped (prepare rc=${RC[barv_prepare]}, script exists: $([ -f barv/run_contact.py ] && echo yes || echo no))"; RC[barv_run_contact]=skipped
fi

{
  echo "end $(date '+%F %T')"
  for k in wait dump_best_cls dump_best_nocls dump_last_cls protocol_eval gate launch_e20 barv_prepare barv_run_contact; do echo "$k=${RC[$k]:-notrun}"; done
} | tee "$OUT/DONE"
