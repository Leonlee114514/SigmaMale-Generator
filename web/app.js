"use strict";

const $ = (id) => document.getElementById(id);

const STATE_LABEL = {
  init: "初始化", loading: "加载模型", running: "运行中",
  error: "异常", stopped: "已停止", absent: "未启动"
};
const VIEW_LABEL = {
  hud: "调试画面 · 含 HUD",
  bare: "录制所见 · 零叠加"
};
const VIEW_HINT = {
  hud: "屏幕上看到的：模式的框与标签，再加上状态栏、录制红点等 HUD 元素。",
  bare: "录制实际取到的画面 —— 所见即所录。除 raw 外，各模式都会带上" +
        "自己的框与标签（那是效果本体，不是水印）。要纯原图请把模式切成 raw。"
};
// 模式说明。raw 单独解释，因为"原画"和"带效果"是两条不同的路。
const MODE_HINT = {
  raw: "完全原画：零处理，逐像素等于摄像头采集帧（仅镜像）。" +
       "后期要跑 grade.py 调色就用它录。",
  debug: "检测可视化：青色人脸线框 + 绿色姿态骨架。不调色，会画框。",
  sigma: "片子主 look：整体压暗 + 脸上暖色方形框 + 金色角框 + 暗角。",
  chad: "方形黑白高对比 + 两行标语。",
  scan: "绿色扫描风格 + 网格 + 锁定框 + 读数。帧率偏低。"
};
// 步骤名的**兜底**显示（还没跑过生产线时占位用）。
// 真跑起来后步骤列表完全以流水线写的 JSON 为准 —— 它能表达固定名字
// 表达不了的东西：哪一步被跳过、每步花了多久、卡在哪一步。
const STEP_NAMES = ["合并", "检测", "切片", "调色", "成片", "验收片", "验收"];
const STEP_LABEL = { wait: "待", run: "跑", done: "完成",
                     skip: "跳过", fail: "失败" };
const REC_LABEL = {
  idle: "空闲", countdown: "倒数中", recording: "录制中",
  saving: "编码中", done: "已保存", error: "出错"
};
const PIPE_LABEL = {
  idle: "待机", running: "运行中", done: "已完成", failed: "失败"
};
const PIPE_HINT_IDLE = "检测 → 切片 → 成片 → 验收。"
                     + "验收不通过会整条线失败并写明是哪一项。";

let S = { modes: [], views: [], mode: "sigma", view: "hud",
          segBuilt: false, playing: null, tab: "live", deepLinked: false,
          rec: {}, pipe: {} };

// 录制倒数的本地锚点。
// 状态轮询是 1.5 秒一次，直接拿它显示倒数会一跳一跳的；这里记下
// "上次同步时还剩几秒 + 同步的时刻"，两次轮询之间用本地时钟插值，
// 画面上的数字就是连续滚动的（每次轮询再把它校准回后端真值）。
let recAnchor = null;
let stepsSig = "";

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  return r.json();
}

function buildSeg(host, items, cur, onPick) {
  host.innerHTML = "";
  items.forEach((v) => {
    const b = document.createElement("button");
    b.textContent = v;
    b.dataset.v = v;
    if (v === cur) b.className = "on";
    b.onclick = () => onPick(v);
    host.appendChild(b);
  });
}
function markSeg(host, cur) {
  Array.prototype.forEach.call(host.children, (b) => {
    b.className = (b.dataset.v === cur) ? "on" : "";
  });
}

function startStream() {
  $("live").src = "/stream.mjpg?view=" + encodeURIComponent(S.view)
                + "&t=" + Date.now();
}

function renderChips(checks) {
  const host = $("chips");
  host.innerHTML = "";
  checks.forEach((c) => {
    const el = document.createElement("span");
    el.className = "chip " + (c.ok ? "ok" : "bad");
    el.title = c.hint || "";
    el.innerHTML = '<span class="d"></span>' + c.name;
    host.appendChild(el);
  });
}

function fitStage() {
  // 画面框的宽高比跟着内容走：预览 16:9（摄像头），播放 9:16（成片竖屏）。
  // 必须同时试两种受约束方向 —— 横的在窄窗口里受宽度限，竖的在矮窗口里
  // 受高度限，只判一边就会出现"框比画面大、四角括号漂在黑边里"。
  const stage = document.querySelector(".stage");
  const mon = stage.parentElement;
  const strip = stage.nextElementSibling;
  const availW = mon.clientWidth;
  const availH = mon.clientHeight - strip.offsetHeight - 10;
  if (availW <= 40 || availH <= 40) return;
  const ar = (S.tab === "live") ? (16 / 9) : (9 / 16);
  let w = availW;
  let h = w / ar;
  if (h > availH) {
    h = availH;
    w = h * ar;
  }
  stage.style.width = Math.floor(w) + "px";
  stage.style.height = Math.floor(h) + "px";
}

function renderSteps(steps) {
  // 只在内容真的变了才重建：轮询 1.5s 一次，每轮都 innerHTML 重建
  // 会让列表轻微闪动。
  const sig = JSON.stringify(steps || []);
  if (sig === stepsSig) return;
  stepsSig = sig;

  const host = $("steps");
  host.innerHTML = "";
  if (!steps || !steps.length) {
    STEP_NAMES.forEach((n, i) => {
      const d = document.createElement("div");
      d.className = "step";
      d.innerHTML = "<span>" + (i + 1) + " " + n + "</span>"
                  + '<span class="s">待</span>';
      host.appendChild(d);
    });
    return;
  }
  steps.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "step " + (s.state || "wait");
    const secs = s.secs ? " " + s.secs.toFixed(0) + "s" : "";
    d.innerHTML = "<span>" + (i + 1) + " " + s.name + "</span>"
                + '<span class="s">' + (STEP_LABEL[s.state] || s.state)
                + secs + "</span>";
    if (s.note) d.title = s.note;
    host.appendChild(d);
  });
}

// ---- 录制状态显示（含本地插值，见 recAnchor 的说明）----
function anchorRec(r) {
  S.rec = r || {};
  if (r && (r.state === "countdown" || r.state === "recording")) {
    recAnchor = { state: r.state, left: r.left || 0,
                  elapsed: r.elapsed || 0, at: performance.now() };
  } else {
    recAnchor = null;
  }
}

function recLeft() {
  if (!recAnchor || recAnchor.state !== "countdown") return 0;
  return Math.max(0, recAnchor.left - (performance.now() - recAnchor.at) / 1000);
}

function recElapsed() {
  if (!recAnchor || recAnchor.state !== "recording") return 0;
  return recAnchor.elapsed + (performance.now() - recAnchor.at) / 1000;
}

function tickRec() {
  const r = S.rec || {};
  const ov = $("recOverlay");
  const big = $("recBig");
  const st = $("recState");
  const pg = $("recProg");

  if (r.state === "countdown") {
    const left = recLeft();
    ov.className = "recoverlay on";
    big.textContent = left > 0.05 ? String(Math.ceil(left)) : "开始";
    st.textContent = "倒数 " + left.toFixed(1) + "s";
    pg.textContent = "0 帧";
  } else if (r.state === "recording") {
    const e = recElapsed();
    ov.className = "recoverlay on rec";
    big.textContent = "● REC " + e.toFixed(1) + "s · " + (r.frames || 0) + " 帧";
    st.textContent = "录制中（目标 " + Math.round(r.secs || 0) + "s）";
    pg.textContent = (r.frames || 0) + " 帧 / " + e.toFixed(1) + "s";
  } else {
    ov.className = "recoverlay";
    st.textContent = REC_LABEL[r.state] || "—";
    if (r.state === "done") pg.textContent = r.out || "已保存";
    else if (r.state === "saving") pg.textContent = (r.frames || 0) + " 帧编码中…";
    else if (r.state === "error") pg.textContent = r.err || "出错";
    else pg.textContent = "—";
  }
}

function syncButtons(c) {
  const r = S.rec || {};
  const p = S.pipe || {};
  const busy = ["countdown", "recording", "saving"].indexOf(r.state) >= 0;
  $("btnRec").disabled = (c.state !== "running") || busy;
  $("btnStop").disabled = !(r.state === "countdown" || r.state === "recording");
  $("btnPipe").disabled = !!p.running;
  $("btnPipeStop").disabled = !p.running;
}

async function pollStatus() {
  let d;
  try { d = await jget("/api/status"); } catch (e) { return; }

  S.modes = d.modes;
  S.views = d.views;
  renderChips(d.checks);

  const c = d.camera || {};
  if (c.mode) S.mode = c.mode;
  $("vCam").textContent = (c.cam === undefined) ? "—" : ("cam " + c.cam);
  $("vRes").textContent = c.w ? (c.w + "×" + c.h) : "—";
  $("vState").textContent = STATE_LABEL[c.state] || c.state || "—";
  $("rFps").textContent = (c.fps === undefined) ? "—" : c.fps;
  $("rFrames").textContent = (c.frames === undefined) ? "—" : c.frames;
  $("rFace").textContent =
    (c.has_face === undefined) ? "—" : (c.has_face ? "有" : "无");
  $("rPose").textContent =
    (c.has_pose === undefined) ? "—" : (c.has_pose ? "有" : "无");

  const ov = $("stageOverlay");
  if (S.tab === "live") {
    if (c.state === "running") {
      ov.classList.add("hide");
    } else {
      ov.classList.remove("hide");
      $("ovText").textContent = c.err
        || (c.state === "loading" ? "正在加载 MediaPipe 模型…" : "摄像头未就绪");
    }
  }

  if (!S.segBuilt) {
    buildSeg($("segMode"), S.modes, S.mode, (v) => {
      S.mode = v;
      markSeg($("segMode"), v);
      if ($("modeHint")) $("modeHint").textContent = MODE_HINT[v] || "";
      jpost("/api/cam/mode", { mode: v });
    });
    buildSeg($("segView"), S.views, S.view, (v) => {
      S.view = v;
      markSeg($("segView"), v);
      $("vLabel").textContent = VIEW_LABEL[v] || v;
      $("viewHint").textContent = VIEW_HINT[v] || "";
      if (S.tab === "live") startStream();
    });
    $("vLabel").textContent = VIEW_LABEL[S.view];
    $("viewHint").textContent = VIEW_HINT[S.view] || "";
    S.segBuilt = true;
  } else {
    markSeg($("segMode"), S.mode);
  }
  // 放在两个分支之外统一刷新：S.mode 可能被后端 config 同步改写
  // （见上面 `if (c.mode) S.mode = c.mode`），写进 if 里会显示滞后。
  if ($("modeHint")) $("modeHint").textContent = MODE_HINT[S.mode] || "";

  // ---- 录制 / 生产线状态 ----
  anchorRec(c.rec);
  S.pipe = d.pipeline || {};
  renderSteps(S.pipe.steps);
  syncButtons(c);

  // 刚录完：立刻刷新成果库，并把新素材填进实拍素材（原本为空时）。
  // 不主动刷的话要等 20 秒轮询才出现，用户会以为没录上 / 选不到最新那条。
  const rs = (S.rec || {}).state;
  if (rs === "done" && S.lastRecState !== "done") {
    loadArtifacts();
    if (S.rec.out && !$("plFootage").value) {
      $("plFootage").value = "output/" + S.rec.out;
    }
    // 人脸命中率太低就直接说清楚：这段做不了特写/锁头。
    // 不说的话，用户要到出片后才发现"特写没生效"，还以为是功能坏了。
    const fr = S.rec.face_rate;
    if (fr !== undefined && fr !== null && fr < 0.3) {
      $("recHint").textContent = "⚠ 这段素材人脸命中仅 " + Math.round(fr * 100)
        + "% —— 人脸特写 / 锁头会失效（它们依赖人脸检测）。"
        + "要特写就重录：正对镜头、脸在画面里、别太远。";
    }
  }
  S.lastRecState = rs;

  $("pipeState").textContent = PIPE_LABEL[S.pipe.state] || "";
  const outs = (S.pipe.outputs || []).length;
  $("pipeHint").textContent = S.pipe.failed
    ? "失败：" + S.pipe.failed
    : (S.pipe.running
        ? "进行中 " + Math.round(S.pipe.elapsed || 0) + "s"
          + (S.pipe.log ? " · " + S.pipe.log : "")
        : (S.pipe.state === "done"
            ? "上次完成，产出 " + outs + " 个成片（见右栏成果库）"
            : PIPE_HINT_IDLE));

  // 交错素材默认填上后端探测到的那个（用户手动改过就不覆盖）
  if (!$("plMaterial").value && d.material) $("plMaterial").value = d.material;

  const lg = d.log || [];
  if (lg.length) $("logline").textContent = lg[lg.length - 1];
}

function mediaUrl(rel) {
  return "/media/" + rel.split("/").map(encodeURIComponent).join("/");
}

async function loadArtifacts() {
  let d;
  try { d = await jget("/api/artifacts"); } catch (e) { return; }
  const host = $("artifacts");
  host.innerHTML = "";
  $("artCount").textContent = "(" + d.artifacts.length + ")";
  if (!d.artifacts.length) {
    host.innerHTML = '<p class="hintline">output/ 里还没有成片</p>';
    return;
  }
  if (!d.badge_ready) {
    const n = document.createElement("p");
    n.className = "hintline";
    n.style.margin = "0 0 7px";
    n.textContent = "验收徽章在里程碑 5 接入 —— 给 verify_*.py 加 --json 之后点亮。";
    host.appendChild(n);
  }
  // 实拍素材的候选。两个要点：
  //   ① 用 a.path（相对**项目根**）而不是 a.rel（相对 output/）——
  //      流水线以项目根为工作目录，喂 rel 会报"素材不存在"。
  //   ② 素材类排前面：成片_/卡点_/复刻版_ 是成品，拿来当实拍源会出怪东西。
  const pick = $("plPick");
  if (pick) {
    const keep = pick.value;
    const isFoot = (n) => /^(录制_|新素材_|my_footage)/.test(n);
    pick.innerHTML = '<option value="">— 从成果库挑一个 —</option>';
    d.artifacts.slice().sort((a, b) => (isFoot(a.name) ? 0 : 1)
                                    - (isFoot(b.name) ? 0 : 1))
      .slice(0, 40).forEach((a) => {
        const o = document.createElement("option");
        o.value = a.path || a.rel;
        o.textContent = (isFoot(a.name) ? "★ " : "· ") + a.name
                      + (a.dur ? "  " + a.dur.toFixed(1) + "s" : "")
                      + "  " + a.mtime;
        pick.appendChild(o);
      });
    pick.value = keep;          // 刷新时保持用户已选的那个
  }

  d.artifacts.forEach((a) => {
    const el = document.createElement("div");
    el.className = "art" + (S.playing === a.rel ? " on" : "");
    el.dataset.rel = a.rel;
    const dur = a.dur ? (a.dur.toFixed(1) + "s") : "—";
    const tags = [];
    if (a.has_plan) tags.push('<span class="pill">有方案</span>');
    el.innerHTML =
      '<div class="n">' + a.name + "</div>" +
      '<div class="m"><span>' + dur + " · " + a.size_mb + "MB</span><span>"
        + a.mtime + "</span></div>" +
      '<div class="m">' + tags.join("") + "<span>" + a.group + "</span></div>";
    el.onclick = () => playArtifact(a);
    host.appendChild(el);
  });

  // 深链：#play=<文件名 或 相对路径> 直接打开某个成片
  if (!S.deepLinked && location.hash.indexOf("#play=") === 0) {
    const want = decodeURIComponent(location.hash.slice(6));
    const hit = d.artifacts.find((a) => a.name === want || a.rel === want);
    if (hit) {
      S.deepLinked = true;
      playArtifact(hit);
    }
  }
}

function markPlaying() {
  Array.prototype.forEach.call($("artifacts").children, (c) => {
    c.classList.toggle("on", c.dataset.rel === S.playing);
  });
}

function playArtifact(a) {
  S.playing = a.rel;
  markPlaying();
  setTab("play");
  const v = $("player");
  v.src = mediaUrl(a.rel);
  v.play().catch(() => {});
}

function setTab(which) {
  const live = (which === "live");
  S.tab = which;
  $("tabLive").classList.toggle("on", live);
  $("tabPlay").classList.toggle("on", !live);
  $("live").style.display = live ? "block" : "none";
  $("player").style.display = live ? "none" : "block";

  if (live) {
    $("player").pause();
    S.playing = null;
    markPlaying();
    startStream();
  } else if (!S.playing) {
    $("stageOverlay").classList.remove("hide");
    $("ovText").textContent = "在右侧成果库里点一个成片来播放";
  }
  fitStage();
}

$("tabLive").onclick = () => setTab("live");
$("tabPlay").onclick = () => setTab("play");

$("btnRec").onclick = async () => {
  const secs = parseFloat($("recSecs").value) || 25;
  const cd = parseFloat($("recCd").value) || 0;
  $("btnRec").disabled = true;      // 先禁用，防连点起两次
  const r = await jpost("/api/rec/start", { secs: secs, countdown: cd });
  if (!r.ok) $("recHint").textContent = "⚠ " + (r.err || "启动失败");
  pollStatus();
};

$("btnStop").onclick = async () => {
  $("btnStop").disabled = true;
  await jpost("/api/rec/stop", {});
  pollStatus();
};

$("btnPipe").onclick = async () => {
  const body = {
    material: $("plMaterial").value.trim(),
    // 每行一段 —— 多段素材交给流水线先合并
    footage: $("plFootage").value.split(/\r?\n/).map((s) => s.trim())
                              .filter(Boolean),
    template: $("plTemplate").value,
    aspect: $("plAspect").value,
    grade: $("plGrade").checked,
    face_crop: parseFloat($("plFace").value) || 0,
    face_lock: $("plLock").checked,
    dry_run: $("plDry").checked
  };
  $("btnPipe").disabled = true;
  const r = await jpost("/api/pipeline/start", body);
  if (!r.ok) $("pipeHint").textContent = "⚠ " + (r.err || "启动失败");
  pollStatus();
};

$("btnPipeStop").onclick = async () => {
  $("btnPipeStop").disabled = true;
  await jpost("/api/pipeline/stop", {});
  pollStatus();
};

// 下拉选中就**追加**一段（不是替换）—— 多段素材要一段段加进来。
// 追加而非覆盖本身就反直觉，所以另配「清空」，并在提示里说清结果是几段。
$("plPick").onchange = (e) => {
  const v = e.target.value;
  if (!v) return;
  const lines = $("plFootage").value.split(/\r?\n/)
                                  .map((s) => s.trim()).filter(Boolean);
  if (lines.indexOf(v) < 0) lines.push(v);
  $("plFootage").value = lines.join("\n");
  e.target.value = "";          // 归位，方便接着挑下一段
  $("pipeHint").textContent = "已加入 " + lines.length + " 段素材，"
                            + "出片时会先合并";
};

// 「用最新」：取候选第一条（列表已按"素材类优先 + 时间倒序"排过），替换整框
$("btnFresh").onclick = () => {
  const opt = $("plPick").options[1];
  if (opt && opt.value) {
    $("plFootage").value = opt.value;
    $("pipeHint").textContent = "已填入最新：" + opt.textContent.trim();
  } else {
    $("pipeHint").textContent = "成果库里还没有可用素材，先在上面录一段";
  }
};

$("btnClear").onclick = () => {
  $("plFootage").value = "";
  $("pipeHint").textContent = PIPE_HINT_IDLE;
};

renderSteps();
fitStage();
startStream();
pollStatus();
loadArtifacts();
setInterval(pollStatus, 1500);
setInterval(tickRec, 100);      // 倒数/录制秒数走本地插值，100ms 刷新才顺滑
setInterval(loadArtifacts, 20000);
window.addEventListener("resize", fitStage);
window.addEventListener("focus", loadArtifacts);
