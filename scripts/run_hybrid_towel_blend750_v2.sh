#!/usr/bin/env bash
# One-command dual-Piper towel-folding run:
#   1) safe reset to the proven ACT start pose
#   2) 750-step ACT + continuous authority-blend SmolVLA execution

set -o pipefail

CONDA_SH=/home/databall_02/miniconda3/etc/profile.d/conda.sh
ACT_ROOT=/home/databall_02/VLA/lerobot_piper
SMOL_ROOT=/home/databall_02/VLA/smolvla_piper_runtime/lerobot_official
RESET_SCRIPT="$ACT_ROOT/reset_piper_pose.py"
HYBRID_SCRIPT="$SMOL_ROOT/scripts/act_smolvla_hybrid_towel_blend.py"

ACT_MODEL="$ACT_ROOT/outputs/train/towel_fold_act_v4_scratch60k/checkpoints/040000/pretrained_model"
EXPERIMENT_ROOT=/home/databall_02/VLA/experiments/smolvla_newonly_from50k_b8_5k_v2
SMOL_MODEL="$EXPERIMENT_ROOT/checkpoints/005000/pretrained_model"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$EXPERIMENT_ROOT/blend_one_click_runs_v2/$RUN_ID"
RESET_LOG="$RUN_DIR/reset.log"
RUN_LOG="$RUN_DIR/hybrid_full750.log"
SUMMARY="$RUN_DIR/summary.txt"

mkdir -p "$RUN_DIR"

trap 'echo; echo "收到停止请求；取消一键任务。"; exit 130' INT TERM

finish() {
    local status="$1"
    echo
    echo "============================================================"
    echo "一键任务结束，状态码=$status"
    echo "运行目录：$RUN_DIR"
    echo "归位日志：$RESET_LOG"
    echo "执行日志：$RUN_LOG"
    echo "ACT和SmolVLA代码、模型均未修改"
    echo "============================================================"
    return "$status"
}

fail() {
    echo "ERROR: $1" >&2
    finish 1
    exit 1
}

echo "============================================================"
echo "双Piper毛巾折叠：ACT + SmolVLA 一键完整750步"
echo "运行编号：$RUN_ID"
echo "前300步：V4 ACT主控"
echo "300步后：SmolVLA连续权威混合（无二进制切换）"
echo "  target = ACT + authority*(SmolVLA_EMA - ACT)"
echo "  authority∈[0,0.95]；不合格/越界动作使authority衰减，命令平滑回到ACT"
echo "============================================================"

[[ -f "$CONDA_SH" ]] || fail "找不到Conda初始化脚本：$CONDA_SH"
[[ -f "$RESET_SCRIPT" ]] || fail "找不到归位脚本：$RESET_SCRIPT"
[[ -f "$HYBRID_SCRIPT" ]] || fail "找不到组合脚本：$HYBRID_SCRIPT"
[[ -f "$SMOL_ROOT/scripts/blend_core.py" ]] || fail "找不到blend_core.py：$SMOL_ROOT/scripts/blend_core.py"
[[ -s "$ACT_MODEL/model.safetensors" ]] || fail "ACT V4模型不完整：$ACT_MODEL"
[[ -s "$SMOL_MODEL/model.safetensors" ]] || fail "SmolVLA模型不完整：$SMOL_MODEL"

if pgrep -af '[l]erobot-record|[r]obot_client|[a]ct_smolvla_hybrid_towel_(safe|smooth|phase_smooth|armonly_smooth|locked_smooth|residual|blend).py' >/dev/null; then
    echo "检测到可能占用机械臂的进程：" >&2
    pgrep -af '[l]erobot-record|[r]obot_client|[a]ct_smolvla_hybrid_towel_safe.py' >&2
    fail "请先停止上面的机械臂控制进程"
fi

# shellcheck source=/dev/null
source "$CONDA_SH" || fail "Conda初始化失败"

echo
echo "===== 第一阶段：双臂归位 ====="
echo "无需输入MOVE或等待倒计时；机械臂将立即使能并归位。"
echo "运行脚本前必须清空工作区、支撑机械臂并准备急停。"

conda activate lerobot_v30 || fail "无法激活lerobot_v30"
cd "$ACT_ROOT" || fail "无法进入ACT目录"

printf 'MOVE\n' | python -B "$RESET_SCRIPT" \
    --arm both \
    --left-can can1 \
    --right-can can0 \
    --execute \
    2>&1 | tee "$RESET_LOG"
RESET_STATUS=${PIPESTATUS[1]}

if [[ "$RESET_STATUS" -ne 0 ]]; then
    fail "归位脚本失败，状态码=$RESET_STATUS"
fi

if ! grep -q 'Pose reset complete' "$RESET_LOG"; then
    fail "没有检测到归位完成标志；可能取消了MOVE或归位未收敛"
fi

echo
echo "===== 第二阶段：完整750步组合执行 ====="

conda activate smolvla_piper || fail "无法激活smolvla_piper"
cd "$SMOL_ROOT" || fail "无法进入SmolVLA运行目录"

LEFT_DEV="$(readlink -f /dev/camera_left 2>/dev/null || true)"
MIDDLE_DEV="$(readlink -f /dev/camera_middle 2>/dev/null || true)"
RIGHT_DEV="$(readlink -f /dev/camera_right 2>/dev/null || true)"

[[ "$LEFT_DEV" =~ ^/dev/video[0-9]+$ ]] || fail "左相机设备无效：$LEFT_DEV"
[[ "$MIDDLE_DEV" =~ ^/dev/video[0-9]+$ ]] || fail "中相机设备无效：$MIDDLE_DEV"
[[ "$RIGHT_DEV" =~ ^/dev/video[0-9]+$ ]] || fail "右相机设备无效：$RIGHT_DEV"

LEFT_IDX="${LEFT_DEV#/dev/video}"
MIDDLE_IDX="${MIDDLE_DEV#/dev/video}"
RIGHT_IDX="${RIGHT_DEV#/dev/video}"

echo "相机：left=$LEFT_IDX middle=$MIDDLE_IDX right=$RIGHT_IDX"
echo "ACT模型：$ACT_MODEL"
echo "SmolVLA模型：$SMOL_MODEL"

export PIPER_ACTION_FILTER_ALPHA=1.0
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Guarantee that an old manual handoff marker cannot enable SmolVLA early.
rm -f /tmp/piper_hybrid_handoff

python -B "$HYBRID_SCRIPT" \
    --act-model "$ACT_MODEL" \
    --smol-model "$SMOL_MODEL" \
    --left-index "$LEFT_IDX" \
    --middle-index "$MIDDLE_IDX" \
    --right-index "$RIGHT_IDX" \
    --left-can can1 \
    --right-can can0 \
    --fps 30 \
    --max-actions 750 \
    --handoff-step 300 \
    --guidance-weight 2.5 \
    --execution-horizon 10 \
    --queue-threshold 30 \
    --act-joint-step-limit 1.0 \
    --act-tracking-limit 3.0 \
    --joint-step-limit 0.10 \
    --policy-disagreement 0.14 \
    --smol-lowpass-alpha 0.25 \
    --smol-raw-step-limit 0.60 \
    --smol-raw-policy-disagreement 0.50 \
    --max-authority 0.95 \
    --authority-up-rate 0.15 \
    --authority-decay 0.85 \
    --authority-hard-decay 0.5 \
    --smol-correction-limit 0.18 \
    --correction-alpha 0.5 \
    --correction-step-limit 0.025 \
    --global-lowpass-alpha 0.85 \
    --max-consecutive-rejects 500 \
    --seed 42 \
    --countdown 0 \
    --execute \
    2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}

ACT_COUNT="$(grep -c 'source=ACT ' "$RUN_LOG" 2>/dev/null || true)"
SMOL_COUNT="$(grep -c 'source=SMOL_BLEND ' "$RUN_LOG" 2>/dev/null || true)"
REJECT_COUNT="$(grep -c 'rejected:' "$RUN_LOG" 2>/dev/null || true)"
EMPTY_COUNT="$(grep -c 'queue empty' "$RUN_LOG" 2>/dev/null || true)"

{
    echo "run_id=$RUN_ID"
    echo "status=$RUN_STATUS"
    echo "act_actions=$ACT_COUNT"
    echo "smolvla_blend_actions=$SMOL_COUNT"
    echo "smolvla_rejected=$REJECT_COUNT"
    echo "queue_empty_fallbacks=$EMPTY_COUNT"
    echo "act_model=$ACT_MODEL"
    echo "smolvla_model=$SMOL_MODEL"
    echo "run_log=$RUN_LOG"
} | tee "$SUMMARY"

ln -sfn "$RUN_DIR" "$EXPERIMENT_ROOT/blend_one_click_runs_v2/latest"

echo
echo "===== 本次结果 ====="
echo "纯ACT动作：$ACT_COUNT"
echo "SmolVLA混合动作：$SMOL_COUNT"
echo "SmolVLA拒绝：$REJECT_COUNT"
echo "队列为空回退：$EMPTY_COUNT"

grep -E \
    'HANDOFF_REQUEST|HYBRID_COMPLETE|Safety stop|Traceback|held|takeover cancelled' \
    "$RUN_LOG" | tail -n 100 || true

finish "$RUN_STATUS"
exit "$RUN_STATUS"
