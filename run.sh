#!/usr/bin/env bash
# 同时启动 Python 后端 API 和 Bun 前端；Ctrl+C 一起停止。
#
# 环境变量：
#   BACKEND_HOST / BACKEND_PORT   后端地址（默认 127.0.0.1 / 8765）
#   FRONTEND_HOST / FRONTEND_PORT 前端地址（默认 127.0.0.1 / 5173）
#   PYTHON                        Python 解释器（默认 python3）
#   MODE                          dev（热重载，默认）或 prod
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8765}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON="${PYTHON:-python3}"
MODE="${MODE:-dev}"

die() { echo "错误：$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "找不到 $PYTHON，请安装 Python ≥ 3.12 或设置 PYTHON"
command -v bun >/dev/null 2>&1 || die "找不到 bun，请先安装：https://bun.sh"

port_busy() {
  (exec 3<>"/dev/tcp/$1/$2") 2>/dev/null && { exec 3>&- 3<&- 2>/dev/null || true; return 0; }
  return 1
}
port_busy "$BACKEND_HOST" "$BACKEND_PORT" && die "端口 $BACKEND_PORT 已被占用（后端是否已在运行？）"
port_busy "$FRONTEND_HOST" "$FRONTEND_PORT" && die "端口 $FRONTEND_PORT 已被占用"

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  echo "安装前端依赖……"
  (cd "$ROOT/frontend" && bun install)
fi

pids=()
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "正在停止……"
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# 不要求先 pip install：直接把 src 加入 PYTHONPATH
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -m cognitive_ai.cli ui --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
pids+=($!)

# 等后端就绪（最多约 10 秒）
for _ in $(seq 1 50); do
  port_busy "$BACKEND_HOST" "$BACKEND_PORT" && break
  kill -0 "${pids[0]}" 2>/dev/null || die "后端启动失败"
  sleep 0.2
done

cd "$ROOT/frontend"
export COGNITIVE_AI_API="http://$BACKEND_HOST:$BACKEND_PORT" HOST="$FRONTEND_HOST" PORT="$FRONTEND_PORT"
if [ "$MODE" = "prod" ]; then
  NODE_ENV=production bun server.js &
else
  bun --hot server.js &
fi
pids+=($!)

echo "打开 http://$FRONTEND_HOST:$FRONTEND_PORT  （Ctrl+C 停止）"
wait -n "${pids[@]}"
