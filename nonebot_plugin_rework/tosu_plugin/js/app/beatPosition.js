// Malody 风格的节拍位置：[拍:格/分度]，拍从 1 开始（与 malody 编辑器口径一致）。
const DIVISORS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

function parseTiming(rawText) {
    const lines = String(rawText || '').split(/\r?\n/);
    let inTiming = false;
    let offset = 0;
    let beatLength = 500;
    let found = false;
    for (const line of lines) {
        const t = line.trim();
        if (!t) continue;
        if (t.startsWith('[')) {
            inTiming = /^\[TimingPoints\]$/i.test(t);
            continue;
        }
        if (!inTiming) continue;
        const parts = t.split(',');
        if (parts.length < 2) continue;
        const time = Number.parseFloat(parts[0]);
        const bl = Number.parseFloat(parts[1]);
        if (!Number.isFinite(time) || !Number.isFinite(bl) || bl <= 0) continue;
        offset = time;
        beatLength = bl;
        found = true;
        break;
    }
    if (!found) {
        offset = 0;
        beatLength = 500;
    }
    return { offset, beatLength };
}

function formatTime(ms) {
    const total = Math.max(0, ms);
    const minutes = Math.floor(total / 60000);
    const seconds = Math.floor((total % 60000) / 1000);
    const millis = Math.round(total % 1000);
    const mm = String(minutes).padStart(2, '0');
    const ss = String(seconds).padStart(2, '0');
    const xxx = String(millis).padStart(3, '0');
    return `${mm}:${ss}.${xxx}`;
}

export function makeBeatLabeller(rawText) {
    const { offset, beatLength } = parseTiming(rawText);
    const measureMs = beatLength * 4;
    return (timeMs, lane, isLn) => {
        const posMs = timeMs - offset;
        let measure = Math.floor(posMs / measureMs);
        let frac = (posMs - measure * measureMs) / measureMs;
        if (frac < 0) {
            measure -= 1;
            frac += 1;
        }
        let bestG = 0;
        let bestD = 4;
        let bestErr = Infinity;
        for (const d of DIVISORS) {
            const g = Math.round(frac * d);
            const err = Math.abs(frac - g / d);
            if (err < bestErr - 1e-12) {
                bestErr = err;
                bestG = g;
                bestD = d;
            }
        }
        let m = measure;
        let g = bestG;
        if (g >= bestD) {
            m += 1;
            g = 0;
        }
        const beat = `[${m + 1}:${g}/${bestD}]`;
        const laneText = Number.isFinite(lane) ? ` 轨 ${lane + 1}` : '';
        const lnText = isLn ? ' [ln]' : '';
        return `${beat} ${formatTime(timeMs)}${laneText}${lnText}`;
    };
}

// 把 overlaps.positions 转成可显示的标签列表：
//   优先显示重叠（红）位置，其次近似重叠（黄）；同类内先 RC 后 LN，各自按时间排序；最多 8 条。
export function buildOverlapLabels(overlaps, rawText) {
    if (!overlaps || typeof overlaps !== 'object') return null;
    const pos = overlaps.positions || {};
    const sortByTime = (arr) => arr.slice().sort((a, b) => (a.t - b.t) || (a.lane - b.lane));
    const redList = [...sortByTime(pos.rc || []), ...sortByTime(pos.ln || [])];
    const nearList = [...sortByTime(pos.nearRc || []), ...sortByTime(pos.nearLn || [])];
    const source = redList.length > 0 ? redList : nearList;
    const kind = redList.length > 0 ? 'red' : 'near';
    const label = makeBeatLabeller(rawText);
    const shown = source.slice(0, 8).map((p) => ({ text: label(p.t, p.lane, p.ln === 1), kind }));
    return {
        red: redList.length,
        near: nearList.length,
        shown,
        truncated: source.length > 8,
        total: source.length,
    };
}
