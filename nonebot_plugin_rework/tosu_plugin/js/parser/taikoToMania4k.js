// js/parser/taikoToMania4k.js
// osu!taiko → osu!mania 4K 文本转换（移植 taiko_to_mania4k.py）。
//
// 转换规则与 Python 版一致：
//   - 鼓边(kat/蓝) → 1/2 轨单键；鼓中(don/红) → 3/4 轨单键
//   - 同色且相邻间隔 < 110ms 的非重击单键分组：
//       组内恰好 2 个：鼓边 [1,2]、鼓中 [4,3]；其他数量：鼓边 [2,1,2,1…]、鼓中 [3,4,3,4…]
//   - 重击(finish) → 两轨同时单键（鼓边 1+2、鼓中 3+4）
//   - [General] Mode → 3；[Difficulty] CircleSize → 4
//   - [TimingPoints] 保留 uninherited，并在每个点后追加值为 主BPM/该BPM 的 SV 点
//
// 与 Python 版的差异（按插件需求）：连打(slider) 与 气球(spinner) 直接忽略，
// 不生成任何物件。

const COL_X = { 1: 64, 2: 192, 3: 320, 4: 448 };
const Y_DEFAULT = 192;
const GROUP_GAP_MS = 110;

const MODE_LINE_RE = /^\s*Mode\s*:\s*(\d+)\s*$/im;
const MODE_KEY_RE = /^\s*Mode\s*:/i;
const CIRCLE_SIZE_RE = /^\s*CircleSize\s*:/i;

/**
 * 该 .osu 文本是否为太鼓谱面（Mode: 1）。
 * @param {string} osuText
 * @returns {boolean}
 */
export function isTaikoBeatmap(osuText) {
    if (typeof osuText !== "string") {
        return false;
    }
    const m = MODE_LINE_RE.exec(osuText);
    return m ? Number(m[1]) === 1 : false;
}

/**
 * 太鼓谱面文本 → mania 4K 谱面文本。
 * 非太鼓文本请先判断 isTaikoBeatmap（本函数不做模式检查，按太鼓格式解析）。
 * @param {string} osuText
 * @returns {string}
 */
export function convertTaikoToMania4k(osuText) {
    const text = String(osuText ?? "").replace(/^\uFEFF/, "");
    const lines = text.split(/\r?\n/);

    // ---- 切分 section（标题行 + [start, end) 正文范围）----
    const headers = [];
    for (let i = 0; i < lines.length; i += 1) {
        const s = lines[i].trim();
        if (s.startsWith("[") && s.endsWith("]")) {
            headers.push({ name: s, index: i });
        }
    }
    const sections = headers.map((h, idx) => ({
        name: h.name,
        index: h.index,
        start: h.index + 1,
        end: idx + 1 < headers.length ? headers[idx + 1].index : lines.length,
    }));
    const sectionOf = (name) => sections.find((s) => s.name === name) || null;

    // ---- 解析 TimingPoints ----
    const uninherited = [];
    const timingSection = sectionOf("[TimingPoints]");
    if (timingSection) {
        for (let i = timingSection.start; i < timingSection.end; i += 1) {
            const line = lines[i].trim();
            if (!line || line.startsWith("//")) continue;
            const parts = line.split(",");
            if (parts.length < 7) continue;
            const time = Number.parseFloat(parts[0]);
            const beatLength = Number.parseFloat(parts[1]);
            const flag = Number.parseInt(parts[6], 10);
            if (!Number.isFinite(time) || !Number.isFinite(beatLength) || flag !== 1) continue;
            uninherited.push({
                time,
                beatLength,
                meter: (parts[2] ?? "").trim(),
                sampleSet: (parts[3] ?? "").trim(),
                sampleIndex: (parts[4] ?? "").trim(),
                volume: (parts[5] ?? "").trim(),
                effects: parts.length > 7 ? parts[7].trim() : "0",
            });
        }
    }

    // ---- 主 BPM：uninherited 中出现次数最多的 BPM（并列取最早）----
    const bpmStats = new Map(); // rounded bpm -> { count, firstIndex }
    uninherited.forEach((p, idx) => {
        if (!(p.beatLength > 0)) return;
        const key = Math.round((60000 / p.beatLength) * 1000) / 1000;
        const entry = bpmStats.get(key);
        if (entry) {
            entry.count += 1;
        } else {
            bpmStats.set(key, { count: 1, firstIndex: idx });
        }
    });
    let mainBpm = 120;
    let best = null;
    for (const [bpm, entry] of bpmStats) {
        if (!best
            || entry.count > best.count
            || (entry.count === best.count && entry.firstIndex < best.firstIndex)) {
            best = entry;
            mainBpm = bpm;
        }
    }

    // ---- 新 TimingPoints：保留 uninherited，并追加 SV 点 ----
    const newTimingLines = [];
    for (const p of uninherited) {
        const bpm = p.beatLength > 0 ? 60000 / p.beatLength : mainBpm;
        const svValue = bpm > 0 ? mainBpm / bpm : 1;
        const svBeatLength = svValue !== 0 ? -100 / svValue : -100;
        newTimingLines.push(
            `${p.time},${p.beatLength},${p.meter},${p.sampleSet},${p.sampleIndex},${p.volume},1,${p.effects}`,
        );
        newTimingLines.push(
            `${p.time},${svBeatLength},${p.meter},${p.sampleSet},${p.sampleIndex},${p.volume},0,${p.effects}`,
        );
    }

    // ---- 解析 HitObjects（只取 circle；slider/spinner 忽略）----
    const circles = [];
    const strongCircles = [];
    const hitSection = sectionOf("[HitObjects]");
    if (hitSection) {
        for (let i = hitSection.start; i < hitSection.end; i += 1) {
            const line = lines[i].trim();
            if (!line || line.startsWith("//")) continue;
            const parts = line.split(",");
            if (parts.length < 5) continue;
            const time = Math.trunc(Number.parseFloat(parts[2]));
            const objType = Number.parseInt(parts[3], 10);
            const hitSound = Number.parseInt(parts[4], 10);
            if (!Number.isFinite(time) || !Number.isFinite(objType) || !Number.isFinite(hitSound)) {
                continue;
            }
            if ((objType & 1) === 0) {
                continue; // slider(连打) / spinner(气球)：按需求忽略
            }
            const isRim = (hitSound & 2) !== 0 || (hitSound & 8) !== 0;
            const entry = {
                time,
                color: isRim ? "rim" : "centre",
                hitSample: parts[parts.length - 1] !== "" ? parts[parts.length - 1] : "0:0:0:0:",
                hitSound,
            };
            if ((hitSound & 4) !== 0) {
                strongCircles.push(entry);
            } else {
                circles.push(entry);
            }
        }
    }

    // ---- 同色且相邻间隔 < GROUP_GAP_MS 的非重击单键分组 ----
    circles.sort((a, b) => a.time - b.time);
    const groups = [];
    let current = [];
    for (const c of circles) {
        if (current.length === 0) {
            current.push(c);
            continue;
        }
        const gap = c.time - current[current.length - 1].time;
        if (c.color === current[current.length - 1].color && gap < GROUP_GAP_MS) {
            current.push(c);
        } else {
            groups.push(current);
            current = [c];
        }
    }
    if (current.length > 0) {
        groups.push(current);
    }

    // ---- 生成 mania 音符 ----
    const out = [];
    const addSingle = (time, col, hitSample, hitSound) => {
        const x = COL_X[col];
        out.push({ time, x, line: `${x},${Y_DEFAULT},${time},1,${hitSound},${hitSample}` });
    };

    for (const group of groups) {
        const size = group.length;
        let cols;
        if (group[0].color === "rim") {
            cols = size === 2
                ? [1, 2]
                : Array.from({ length: size }, (_, i) => (i % 2 === 0 ? 2 : 1));
        } else {
            cols = size === 2
                ? [4, 3]
                : Array.from({ length: size }, (_, i) => (i % 2 === 0 ? 3 : 4));
        }
        for (let i = 0; i < size; i += 1) {
            addSingle(group[i].time, cols[i], group[i].hitSample, group[i].hitSound);
        }
    }

    for (const c of strongCircles) {
        if (c.color === "rim") {
            addSingle(c.time, 1, c.hitSample, c.hitSound);
            addSingle(c.time, 2, c.hitSample, c.hitSound);
        } else {
            addSingle(c.time, 3, c.hitSample, c.hitSound);
            addSingle(c.time, 4, c.hitSample, c.hitSound);
        }
    }

    out.sort((a, b) => (a.time - b.time) || (a.x - b.x));
    const hitLines = out.map((o) => o.line);

    // ---- 重建文本 ----
    const output = [];
    let cursor = 0;
    const handled = new Set();
    for (const sec of sections) {
        for (; cursor <= sec.index; cursor += 1) {
            output.push(lines[cursor]);
        }
        const body = lines.slice(sec.start, sec.end);

        if (sec.name === "[TimingPoints]" && !handled.has(sec.name)) {
            handled.add(sec.name);
            output.push(...newTimingLines);
        } else if (sec.name === "[HitObjects]" && !handled.has(sec.name)) {
            handled.add(sec.name);
            output.push(...hitLines);
        } else if (sec.name === "[General]") {
            for (const ln of body) {
                output.push(MODE_KEY_RE.test(ln) ? "Mode: 3" : ln);
            }
        } else if (sec.name === "[Difficulty]") {
            let csFound = false;
            for (const ln of body) {
                if (CIRCLE_SIZE_RE.test(ln)) {
                    output.push("CircleSize: 4");
                    csFound = true;
                } else {
                    output.push(ln);
                }
            }
            if (!csFound) {
                output.push("CircleSize: 4");
            }
        } else {
            output.push(...body);
        }
        cursor = sec.end;
    }
    for (; cursor < lines.length; cursor += 1) {
        output.push(lines[cursor]);
    }

    return output.join("\n");
}

export default convertTaikoToMania4k;
