"""把 Codex 的 rollout jsonl 提取成可读的对话文本。"""
import json
import os
import sys

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
if not os.path.isabs(path):
    path = os.path.join(os.path.expanduser("~"), ".codex", path)

n = 0
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type") or o.get("role") or ""
        # 只关心对话内容
        payload = o.get("payload") or o
        role = payload.get("role") or payload.get("type") or t
        content = payload.get("content")
        if not content:
            continue
        if isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    parts.append(c.get("text") or c.get("content") or "")
                else:
                    parts.append(str(c))
            text = "\n".join(p for p in parts if p)
        else:
            text = str(content)
        text = text.strip()
        if not text:
            continue
        if role not in ("user", "assistant", "system"):
            continue
        n += 1
        if n > limit:
            break
        tag = {"user": "【用户】", "assistant": "【助手】",
               "system": "【系统】"}.get(role, f"【{role}】")
        print(f"\n{tag} {text[:2500]}")
