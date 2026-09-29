#!/usr/bin/env bash
#
# llm-visualizer 开发环境一键脚本
#
#   ./dev.sh up       启动开发服务器，并自动打开浏览器  ← 最常用
#   ./dev.sh down     停止开发服务器
#   ./dev.sh restart  重启
#   ./dev.sh logs     实时查看日志（Ctrl+C 退出）
#   ./dev.sh status   查看运行状态
#   ./dev.sh open     只打开浏览器
#   ./dev.sh check    类型检查
#   ./dev.sh verify   端到端校验（模型数学 + 插拔性）
#   ./dev.sh build    构建静态站点到 ./build
#   ./dev.sh install  重新安装依赖
#
# 端口默认 9000，可用环境变量覆盖：PORT=9100 ./dev.sh up
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 端口的默认值只写在这一处：`docker-compose.yml` 里用的是 `${PORT:-9000}`，
# 这里 export 出去，容器的端口映射与 Vite 的 `--port` 就永远一致。
PORT="${PORT:-9000}"
export PORT
URL="http://localhost:${PORT}"

# ── 颜色 ─────────────────────────────────────────────
if [ -t 1 ]; then
	B=$'\033[1m'; DIM=$'\033[2m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
else
	B=''; DIM=''; G=''; Y=''; R=''; N=''
fi
info() { printf '%s→%s %s\n' "$B$Y" "$N" "$*"; }
ok()   { printf '%s✓%s %s\n' "$B$G" "$N" "$*"; }
err()  { printf '%s✗%s %s\n' "$B$R" "$N" "$*" >&2; }

# ── 依赖工具检查 ─────────────────────────────────────
need_docker() {
	command -v docker >/dev/null 2>&1 || {
		err "找不到 docker 命令。请先安装 Docker：https://docs.docker.com/get-docker/"
		exit 1
	}
}

docker_ready() { docker info >/dev/null 2>&1; }

# 确保 Docker 守护进程在运行；没起来就尝试拉起并等待
ensure_docker() {
	need_docker
	if docker_ready; then
		ok "Docker 已在运行"
		return 0
	fi

	info "Docker 守护进程没在运行，尝试启动…"
	# 桌面版 Docker 一般可以这样拉起；拉不起来也不报错，下面等到超时会给出提示
	open -a Docker >/dev/null 2>&1 || true

	printf '   等待 Docker 就绪'
	for _ in $(seq 1 60); do
		if docker_ready; then
			printf '\n'
			ok "Docker 已就绪"
			return 0
		fi
		printf '.'
		sleep 1
	done
	printf '\n'
	err "连不上 Docker 守护进程。请先启动 Docker，再重跑本脚本。"
	exit 1
}

# 依赖是否已装好（用 node_modules 里有没有 svelte 判断）
deps_ready() {
	docker compose run --rm --no-deps dev-install sh -c 'test -d node_modules/svelte' >/dev/null 2>&1
}

ensure_deps() {
	if deps_ready; then
		ok "依赖已就绪"
		return 0
	fi
	info "首次运行，正在安装依赖（约 1 分钟）…"
	docker compose run --rm dev-install
	ok "依赖安装完成"
}

http_ok() { curl -sf -o /dev/null --max-time 2 "$URL"; }

# 等网页真正可访问
wait_http() {
	printf '   等待网页就绪'
	for _ in $(seq 1 60); do
		if http_ok; then
			printf '\n'
			return 0
		fi
		printf '.'
		sleep 1
	done
	printf '\n'
	return 1
}

open_browser() {
	if command -v open >/dev/null 2>&1; then
		open "$URL"
	else
		info "请手动在浏览器打开：$URL"
	fi
}

report_ready() {
	printf '\n  %s网页地址：%s%s%s\n' "$B" "$B$G" "$URL" "$N"
	printf '  %s停止：./dev.sh down   ｜  日志：./dev.sh logs%s\n\n' "$DIM" "$N"
	open_browser
}

# ── 子命令 ───────────────────────────────────────────
cmd_up() {
	ensure_docker
	ensure_deps

	# 已经能用就什么都不做：不碰容器，HMR 状态也保留
	if http_ok; then
		ok "开发服务器已在运行"
		report_ready
		return 0
	fi

	info "启动开发服务器…"
	# `--force-recreate`：容器可能"在跑但进程状态坏了"（比如页面一直 500）。
	# 那时 `up -d` 是空操作、救不回来，必须换一个新容器才行。
	docker compose up -d --force-recreate dev

	if wait_http; then
		ok "已就绪"
		report_ready
	else
		err "网页未能就绪。用 ./dev.sh logs 查看日志。"
		exit 1
	fi
}

cmd_down() {
	need_docker
	info "停止开发服务器…"
	docker compose down
	ok "已停止"
}

cmd_logs() {
	need_docker
	info "日志（Ctrl+C 退出）…"
	docker compose logs -f dev
}

cmd_status() {
	ensure_docker
	printf '\n'
	docker compose ps
	printf '\n'
	if http_ok; then
		ok "网页可访问：$URL"
	else
		info "网页未在运行。用 ./dev.sh up 启动。"
	fi
}

cmd_open() {
	if http_ok; then
		open_browser
	else
		info "服务没在跑，先执行 ./dev.sh up"
		exit 1
	fi
}

cmd_check()  { ensure_docker; ensure_deps; docker compose run --rm check;  }
cmd_verify() { ensure_docker; ensure_deps; docker compose run --rm verify; }
cmd_build()  { ensure_docker; ensure_deps; docker compose run --rm build; }
cmd_install() {
	ensure_docker
	info "重新安装依赖…"
	docker compose run --rm dev-install
	ok "完成"
}

cmd_help() {
	sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

case "${1:-help}" in
	up)      cmd_up ;;
	down)    cmd_down ;;
	restart) cmd_down; cmd_up ;;
	logs)    cmd_logs ;;
	status)  cmd_status ;;
	open)    cmd_open ;;
	check)   cmd_check ;;
	verify)  cmd_verify ;;
	build)   cmd_build ;;
	install) cmd_install ;;
	help|-h|--help|"") cmd_help ;;
	*) err "未知命令：$1"; printf '\n'; cmd_help; exit 1 ;;
esac
