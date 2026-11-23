// RC 键型区间：把谱面按「排（同刻合并）→ 合并 → 区间」切成 Jacks / Stream / Chordstream，
// 用于 RC 难度-进度图下方的涂色条。
//
// 规则与 diff-analysis 的 lib/rcpatterns.mjs 一致：
// 1. 同刻物件合并为排；
// 2. 相邻排可合并：间距 < 15ms、两排无同轨、间距 < 上下间距最大值的 1/4；
// 3. 相邻排之间是一个区间：两端排有同轨 = Jacks，否则 Stream；
// 4. 相邻两个 Stream 区间，若「前区间首排」与「后区间尾排」有同轨，则两区间都改判 Chordstream。

export const MERGE_GAP_MS = 15;
export const MERGE_RATIO = 1 / 4;

const shareLane = (a, b) => {
    for (const l of a) if (b.includes(l)) return true;
    return false;
};

/** 从 .osu 文本里取 [HitObjects] 的 (t, lane)；lane = floor(x*4/512) 兼容 4K。 */
export function parseNotes(text, keyCount = 4) {
    const out = [];
    const lines = String(text).split(/\r?\n/);
    let inObj = false;
    const cols = Math.max(1, Number(keyCount) || 4);
    for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith("//")) continue;
        if (line.startsWith("[")) {
            inObj = line.toLowerCase() === "[hitobjects]";
            continue;
        }
        if (!inObj) continue;
        const p = line.split(",");
        if (p.length < 5) continue;
        const t = Number(p[2]);
        const x = Number(p[0]);
        const typ = Number.parseInt(p[3], 10);
        if (!Number.isFinite(t) || !Number.isFinite(x)) continue;
        // 与 osu! 一致：尾时间无效的 LN 整个物件被忽略（这类错误物件会虚增物量）。
        if ((typ & 128) !== 0) {
            const tail = Number.parseFloat(String(p[5] || "").split(":")[0]);
            if (!Number.isFinite(tail) || tail <= t || tail < 0) continue;
        }
        const lane = Math.min(cols - 1, Math.max(0, Math.floor((x * cols) / 512)));
        out.push({ t, lane });
    }
    out.sort((a, b) => a.t - b.t || a.lane - b.lane);
    return out;
}

export function buildRows(notes) {
    const rows = [];
    for (const nt of notes) {
        const last = rows[rows.length - 1];
        if (last && last.t === nt.t) {
            if (!last.lanes.includes(nt.lane)) last.lanes.push(nt.lane);
        } else {
            rows.push({ t: nt.t, lanes: [nt.lane], n: 1 });
        }
    }
    for (const r of rows) r.n = r.lanes.length;
    return rows;
}

export function mergeRows(rows) {
    const list = rows.map((r) => ({ t: r.t, lanes: r.lanes.slice(), n: r.n }));
    for (;;) {
        const gaps = [];
        for (let i = 0; i + 1 < list.length; i += 1) gaps.push(list[i + 1].t - list[i].t);
        let pick = -1;
        let pickRatio = Infinity;
        for (let i = 0; i + 1 < list.length; i += 1) {
            const g = gaps[i];
            if (g >= MERGE_GAP_MS) continue;
            if (shareLane(list[i].lanes, list[i + 1].lanes)) continue;
            const above = i > 0 ? gaps[i - 1] : Infinity;
            const below = i + 2 < list.length ? gaps[i + 1] : Infinity;
            const ref = Math.max(Number.isFinite(above) ? above : -Infinity, Number.isFinite(below) ? below : -Infinity);
            const limit = Number.isFinite(ref) && ref > 0 ? ref * MERGE_RATIO : Infinity;
            if (!(g < limit)) continue;
            const ratio = g / (limit || 1);
            if (ratio < pickRatio) {
                pick = i;
                pickRatio = ratio;
            }
        }
        if (pick < 0) break;
        const a = list[pick];
        const b = list[pick + 1];
        list.splice(pick, 2, {
            t: Math.round((a.t * a.n + b.t * b.n) / (a.n + b.n)),
            lanes: [...new Set([...a.lanes, ...b.lanes])].sort((x, y) => x - y),
            n: a.n + b.n,
        });
    }
    return list;
}

export function buildIntervals(merged) {
    const out = [];
    for (let i = 0; i + 1 < merged.length; i += 1) {
        const A = merged[i];
        const B = merged[i + 1];
        out.push({
            t0: A.t,
            t1: B.t,
            type: shareLane(A.lanes, B.lanes) ? "Jacks" : "Stream",
            weight: A.n + B.n,
        });
    }
    for (let i = 0; i + 1 < out.length; i += 1) {
        const i1 = out[i];
        const i2 = out[i + 1];
        if (i1.type === "Jacks" || i2.type === "Jacks") continue;
        if (!shareLane(merged[i].lanes, merged[i + 2].lanes)) continue;
        i1.type = "Chordstream";
        i2.type = "Chordstream";
    }
    return out;
}

/** 文本 -> { t0, t1, segs: [{t0,t1,type}] }；解析失败返回 null。 */
export function buildRcIntervals(text, keyCount = 4) {
    if (!text) return null;
    const notes = parseNotes(text, keyCount);
    if (notes.length < 4) return null;
    const merged = mergeRows(buildRows(notes));
    if (merged.length < 3) return null;
    const segs = buildIntervals(merged);
    if (!segs.length) return null;
    return { t0: merged[0].t, t1: merged[merged.length - 1].t, segs };
}

const WEIGHT = { Jacks: 1, Chordstream: 2, Stream: 3 };

/** 把区间压成固定分辨率的分段（按物量投票），返回 [{from,to,type}] 比例在 0..1。 */
export function ribbonSegments(rcIntervals, bins = 420) {
    const data = rcIntervals;
    if (!data) return [];
    const span = data.t1 - data.t0;
    if (!(span > 0)) return [];
    const votes = new Array(bins);
    for (const seg of data.segs) {
        const a = Math.max(0, Math.min(bins - 1, Math.floor(((seg.t0 - data.t0) / span) * bins)));
        const b = Math.max(0, Math.min(bins - 1, Math.ceil(((seg.t1 - data.t0) / span) * bins) - 1));
        const dur = Math.max(1, seg.t1 - seg.t0);
        for (let i = a; i <= b; i += 1) {
            if (!votes[i]) votes[i] = { Jacks: 0, Chordstream: 0, Stream: 0 };
            votes[i][seg.type] += dur / Math.max(1, b - a + 1);
        }
    }
    const out = [];
    for (let i = 0; i < bins; i += 1) {
        const v = votes[i];
        let type = "gap";
        if (v) {
            let best = -1;
            for (const k of Object.keys(v)) {
                if (v[k] > best) {
                    best = v[k];
                    type = k;
                }
            }
            if (v.Jacks === 0 && v.Chordstream === 0 && v.Stream === 0) type = "gap";
        }
        if (out.length && out[out.length - 1].type === type) out[out.length - 1].to = i + 1;
        else out.push({ from: i, to: i + 1, type });
    }
    return out.map((s) => ({ left: s.from / bins, width: (s.to - s.from) / bins, type: s.type }));
}

export { WEIGHT };
