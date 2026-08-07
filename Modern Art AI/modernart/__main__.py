import sys

from .cli import main

# Windows では並列処理がプロセスの作り直しで実装されており、その際にこのファイルが
# もう一度読み込まれる。ガードがないと起動が入れ子になって増え続ける。
if __name__ == "__main__":
    sys.exit(main())
