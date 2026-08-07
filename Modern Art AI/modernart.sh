#!/usr/bin/env bash
# ブラウザ版を起動する。どのディレクトリからでも動く。
#   ~/projects/art/modernart.sh              # ブラウザ版
#   ~/projects/art/modernart.sh --time 20    # 思考時間を伸ばす
#   ~/projects/art/modernart.sh --cli        # ターミナル版
cd "$(dirname "$(readlink -f "$0")")" || exit 1

if [ "$1" = "--cli" ]; then
    shift
    exec python3 -m modernart "$@"
fi
exec python3 -m modernart.server "$@"
