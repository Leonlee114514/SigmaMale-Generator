"""前端服务接口自检（probe_webapp.py）
====================================================================
覆盖：页面与静态资源 / 状态接口 / MJPEG 预览流 / 模式与视图切换 /
      成片库 / 成片播放（含 Range）/ 路径穿越防护。

用法（**需要 webapp.py 已经在跑**）：
    python webapp.py --no-browser        # 另开一个终端
    python probe_webapp.py

为什么单独一个脚本：预览流是 multipart 长连接、视频要验 Range 请求，
这些用 shell 的 curl 不好断言，必须用 HTTP 客户端真读一遍。
"""
import io
import json
import re
import time

import requests

B = "http://127.0.0.1:8760"
ok_all = True


def head(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def check(name, cond, extra=""):
    global ok_all
    ok_all = ok_all and bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))


head("1) 页面与静态资源")
r = requests.get(B + "/", timeout=10)
check("GET /", r.status_code == 200, f"{r.status_code}, {len(r.content)} bytes")
check("页面标题", "西格玛男人生成器" in r.text)
for f, kw in [("style.css", "--gold"), ("app.js", "startStream")]:
    rr = requests.get(f"{B}/static/{f}", timeout=10)
    check(f"GET /static/{f}", rr.status_code == 200 and kw in rr.text,
          f"{rr.status_code}, {len(rr.content)} bytes")

head("2) 状态接口（等摄像头上线）")
st = None
t0 = time.time()
while time.time() - t0 < 40:
    st = requests.get(B + "/api/status", timeout=10).json()
    if st["camera"].get("state") == "running":
        break
    time.sleep(1.5)
cam = st["camera"]
check("摄像头 running", cam.get("state") == "running",
      f"state={cam.get('state')} err={cam.get('err', '')[:60]}")
check("分辨率", (cam.get("w") == 1280 and cam.get("h") == 720),
      f"{cam.get('w')}x{cam.get('h')}")
check("fps 有读数", (cam.get("fps") or 0) > 0, f"{cam.get('fps')} fps")
check("帧号在涨", (cam.get("frames") or 0) > 0, f"{cam.get('frames')}")
print("     自检项：")
for c in st["checks"]:
    print(f"       {'✓' if c['ok'] else '✗'} {c['name']}  {c['hint']}")
print(f"     模式 {st['modes']}   预览 {st['views']}")
print(f"     日志末行：{st['log'][-1] if st['log'] else '(空)'}")

head("3) 预览流（读 3 秒，数帧 + 验 JPEG 头）")
t0 = time.time()
buf = b""
frames = 0
r = requests.get(B + "/stream.mjpg?view=hud", stream=True, timeout=10)
for chunk in r.iter_content(8192):
    buf += chunk
    frames = buf.count(b"\xff\xd8\xff")
    if time.time() - t0 > 3.0:
        break
r.close()
dt = time.time() - t0
check("收到 JPEG 帧", frames > 5, f"{frames} 帧 / {dt:.1f}s = {frames/dt:.1f} fps")
m = re.search(rb"Content-Length: (\d+)", buf)
check("有 Content-Length 头", bool(m),
      f"单帧约 {int(m.group(1))/1024:.0f}KB" if m else "")

head("4) 切模式 / 切预览视图")
rr = requests.post(B + "/api/cam/mode", json={"mode": "chad"}, timeout=10)
check("POST 模式 chad", rr.status_code == 200 and rr.json().get("mode") == "chad",
      str(rr.json()))
rr = requests.post(B + "/api/cam/mode", json={"mode": "不存在"}, timeout=10)
check("非法模式被拒", rr.status_code == 400, f"{rr.status_code}")
r2 = requests.get(B + "/stream.mjpg?view=bare", stream=True, timeout=10)
it = r2.iter_content(4096)
for _ in range(12):
    next(it)
r2.close()
st2 = requests.get(B + "/api/status", timeout=10).json()
check("view 已切到 bare", st2["camera"].get("view") == "bare",
      str(st2["camera"].get("view")))
requests.post(B + "/api/cam/mode", json={"mode": "sigma"}, timeout=10)
requests.get(B + "/stream.mjpg?view=hud", stream=True, timeout=10).close()

head("5) 成片库")
d = requests.get(B + "/api/artifacts", timeout=30).json()
arts = d["artifacts"]
check("扫到成片", len(arts) > 0, f"{len(arts)} 个")
check("有验收徽章文件", True, f"badge_ready={d['badge_ready']}")
print("     最近 6 个：")
for a in arts[:6]:
    print(f"       {a['name'][:30]:32} {a['dur']:>7.2f}s  {a['size_mb']:>6.2f}MB"
          f"  {a['mtime']}  plan={a['has_plan']}")
check("时长有值", all(a["dur"] >= 0 for a in arts))

head("6) 成片播放（含 Range，视频拖动要用）")
if arts:
    a = arts[0]
    url = B + "/media/" + "/".join(
        requests.utils.quote(x) for x in a["rel"].split("/"))
    rr = requests.get(url, timeout=30)
    check("完整取回", rr.status_code == 200 and len(rr.content) > 10000,
          f"{rr.status_code}, {len(rr.content)/1024/1024:.2f}MB")
    rr = requests.get(url, headers={"Range": "bytes=0-1023"}, timeout=30)
    check("支持 Range", rr.status_code == 206, f"{rr.status_code}")

head("7) 路径穿越防护")
rr = requests.get(B + "/media/../webapp.py", timeout=10)
check("拒绝 ../ 穿越", rr.status_code in (400, 403, 404), f"{rr.status_code}")

print("\n" + "=" * 68)
print("总结果：" + ("全部通过" if ok_all else "有未通过项"))
print("=" * 68)
