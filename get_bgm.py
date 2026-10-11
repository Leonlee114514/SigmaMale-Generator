"""抓取固定 BGM（get_bgm.py）
================================================================
把 "gigachad theme (phonk house version)" 抓下来放进 bgm/。

⚠ 本机网络实测结论（2026-09-22）：
    YouTube **完全不可达**。
      - 直连：TCP 超时
      - 走环境代理（127.0.0.1:59692）：SSL UNEXPECTED_EOF（代理丢弃 TLS 流）
    而 PyPI 两条路都通（1.1~1.4s）。
    => 本脚本在本机**当前网络下大概率跑不通**，
       但它对"网络能到 YouTube"的环境是有效的，所以保留。
       若跑不通，请手动下载后把文件放进 bgm/ —— 见 README 的说明。

用法：
    python get_bgm.py                          # 按默认关键词搜第一首
    python get_bgm.py --query "关键词"
    python get_bgm.py --url "https://..."      # 直接给链接
    python get_bgm.py --list                    # 只列出候选不下载
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
BGM_DIR = os.path.join(BASE, "bgm")

DEFAULT_QUERY = "gigachad theme phonk house version"
# 目标：只要音频，转成 mp3，固定文件名方便下游引用
TARGET_NAME = "gigachad_phonk.mp3"


def proxy_of_env():
    """从环境变量取代理。注意端口每次会话都变，不要写死。"""
    for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def make_opts(proxy, list_only=False):
    opts = {
        "quiet": False,
        "no_warnings": False,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 20,
        # 网络受限环境下宁可慢也不要被限速搞死
        "nocheckcertificate": True,
    }
    if proxy:
        opts["proxy"] = proxy
    if list_only:
        opts.update({"skip_download": True, "extract_flat": True})
    else:
        opts.update({
            "format": "bestaudio/best",
            "outtmpl": os.path.join(BGM_DIR, "%(id)s.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        })
    return opts


def main():
    ap = argparse.ArgumentParser(description="抓取固定 BGM")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--url")
    ap.add_argument("--list", action="store_true", help="只列候选")
    ap.add_argument("--no-proxy", action="store_true", help="强制直连")
    a = ap.parse_args()

    try:
        import yt_dlp
    except ImportError:
        print("缺 yt-dlp。安装：\n  <python> -m pip install yt-dlp")
        return 1

    os.makedirs(BGM_DIR, exist_ok=True)
    proxy = None if a.no_proxy else proxy_of_env()
    print(f"yt-dlp {yt_dlp.version.__version__}")
    print(f"代理   {proxy or '（直连）'}")
    print(f"目标   {a.url or ('ytsearch:' + a.query)}")

    target = a.url or f"ytsearch1:{a.query}"

    # ---- 先探测，失败时给出**可操作的**结论而不是堆栈 ----
    try:
        with yt_dlp.YoutubeDL(make_opts(proxy, list_only=True)) as y:
            info = y.extract_info(target, download=False)
    except Exception as e:
        print("\n抓取失败。")
        print(f"  原因: {type(e).__name__}: {str(e)[:220]}")
        print("\n本机已知的情况：YouTube 不可达（直连超时 / 代理断 TLS）。")
        print("可行的替代做法：")
        print(f"  1. 在能上网的机器上下载，然后把音频文件拷到:\n     {BGM_DIR}\\")
        print(f"     文件名随意（bgm_pipeline 会取目录里第一个音频文件），")
        print(f"     建议命名 {TARGET_NAME}")
        print("  2. 或改用你已有的本地 phonk 音频文件")
        print("  3. 或先跑 `python bgm_pipeline.py --analyze-only --bgm 你的文件`"
              " 检查 BPM 识别是否正常")
        return 2

    entries = info.get("entries") if info and "entries" in info else [info]
    entries = [e for e in entries if e]
    if a.list or entries is None:
        print("\n候选：")
        for e in entries[:10]:
            print(f"  {e.get('title')}  | {e.get('duration')}s | {e.get('id')}")
        return 0

    if not entries:
        print("没有搜到结果")
        return 1
    if len(entries) > 1 and not a.url:
        print("\n选中第一首：", entries[0].get("title"))

    vid = entries[0].get("id") or entries[0].get("url")
    try:
        with yt_dlp.YoutubeDL(make_opts(proxy)) as y:
            y.download([vid])
    except Exception as e:
        print(f"\n下载失败: {type(e).__name__}: {str(e)[:220]}")
        return 3

    # ---- 归一化成固定文件名 ----
    mp3s = [f for f in os.listdir(BGM_DIR) if f.lower().endswith(".mp3")]
    if not mp3s:
        print("下载完成但没有 mp3 产物 —— 检查 ffmpeg 是否可用")
        return 4
    newest = max(mp3s, key=lambda f: os.path.getmtime(os.path.join(BGM_DIR, f)))
    dst = os.path.join(BGM_DIR, TARGET_NAME)
    src = os.path.join(BGM_DIR, newest)
    if src != dst:
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
    print(f"\n完成 -> {dst}  ({os.path.getsize(dst)/1024:.0f} KB)")
    print(f"下一步：python bgm_pipeline.py --analyze-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
