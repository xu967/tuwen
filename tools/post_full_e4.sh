#!/bin/bash
# 全量实验 glp_full_e4_resume 的收尾流水线（守护进程，GPU 空出后自动接力）：
#   1) 等训练主进程退出  2) best/last 权重的 PredCls dump（GT 框 ± GT 类别，1,024 张 full_monitor val）
#   3) 与官方 RelateAnything 权重同口径评测（tools/sgg_protocol_eval.py）  4) BARV 难负例 cache 重建（barv.prepare）
# 启动：cd <repo> && mkdir -p runs/test/glp_full_e4_resume/post && \
#       setsid nohup bash tools/post_full_e4.sh > runs/test/glp_full_e4_resume/post/run.log 2>&1 < /dev/null &
# 进度：tail -f runs/test/glp_full_e4_resume/post/run.log     停止：kill -- -<PGID>（PGID 见 run.log 第一行）
# 完成标记：runs/test/glp_full_e4_resume/post/DONE（每步退出码 + 时间；任一步失败后续依赖步骤跳过）
set -u
# ---------------- 配置区 ----------------
ROOT=/root/rivermind-data/YOLOVOmodal_3871538535
TRAIN_PID=${TRAIN_PID:-82761}        # glp_full_e4_resume 的 python train.py 主进程（cmdline 含 train.py 才算活着）；已退出则直接收尾
RUN=runs/test/glp_full_e4_resume
OUT=$RUN/post                        # 日志 / DONE 标记都放这里（数据盘，不放 /tmp）
DATA=RA-4M/data_full_monitor.yaml    # 与 reports/baseline_cmp 同一 1,024 张 val、原始 10,370 标签
CMP=reports/baseline_cmp             # 官方 RA dump（ra_official_vits16plus.pt）与 epoch1 dump 已在这里
PSG_DONE=/root/rivermind-data/psg_hf/CONVERT_DONE   # barv.hf_psg 转换完成标记，内容 exit=0 才进入 BARV 步骤
BARV_CFG=configs/barv_psg_contact.yaml              # 难负例配置；cache 写 data/barv_cache_contact（不覆盖 pilot）
POLL_SEC=60                          # 等训练退出的轮询间隔；训练 1 epoch ≈ 3.9 h，没必要更密
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

# 3) 同口径评测（官方 RA + epoch1 + 本次）
if [ "${RC[dump_best_cls]:-1}" -eq 0 ] && [ "${RC[dump_best_nocls]:-1}" -eq 0 ] && [ "${RC[dump_last_cls]:-1}" -eq 0 ]; then
  $PY tools/sgg_protocol_eval.py \
      "$CMP/ra_official_vits16plus.pt" "$CMP/ours_full_e1_cls.pt" "$CMP/ours_full_e1_nocls.pt" \
      "$CMP/ours_full_e4_best_cls.pt" "$CMP/ours_full_e4_best_nocls.pt" "$CMP/ours_full_e4_last_cls.pt" \
      --names RA_official ours_e1_cls ours_e1_nocls ours_e4best_cls ours_e4best_nocls ours_e4last_cls \
      --json "$CMP/protocol_eval_full_monitor_e4.json" 2>&1 | grep -v -i warn | tee "$OUT/protocol_eval.txt"
  RC[protocol_eval]=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] protocol_eval rc=${RC[protocol_eval]}"
else
  echo "[$(date '+%F %T')] protocol_eval skipped (dump failed)"; RC[protocol_eval]=skipped
fi

# 4) BARV 难负例 cache（需要 GPU 抽特征，所以排在训练之后；依赖 hf_psg 转换成功）
if [ -s "$PSG_DONE" ] && grep -q "^exit=0" "$PSG_DONE"; then
  $PY -m barv.prepare --config "$BARV_CFG" > "$OUT/barv_prepare.log" 2>&1; RC[barv_prepare]=$?
  echo "[$(date '+%F %T')] barv_prepare rc=${RC[barv_prepare]}  $(grep -v -i warn "$OUT/barv_prepare.log" | tail -n 2 | cut -c1-300)"
else
  echo "[$(date '+%F %T')] barv_prepare skipped: $PSG_DONE missing or exit!=0 ($(cat "$PSG_DONE" 2>/dev/null))"; RC[barv_prepare]=skipped
fi

{
  echo "end $(date '+%F %T')"
  for k in wait dump_best_cls dump_best_nocls dump_last_cls protocol_eval barv_prepare; do echo "$k=${RC[$k]:-notrun}"; done
} | tee "$OUT/DONE"
