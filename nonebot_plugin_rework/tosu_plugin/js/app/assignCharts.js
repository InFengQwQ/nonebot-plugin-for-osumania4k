// 难度-进度图：把 RC / LN 通道的逐物件赋值序列画成「难度 vs 谱面进度」两张小图。
// - y 轴用赋值原始值（线性），上方数据稀疏、下方密集；
// - 折线每段颜色 = 该物件 value 在左侧 value 胶囊上的自动背景色；
// - 整数 value 水平标注线：只取上方稀疏区的一部分，左侧标注该 value 的段位名；
// - 游玩时叠加一根竖直线表示当前游戏进度（tosu 实时时间）。
import { APP_CONFIG, mainCardEl, state } from "./appContext.js";
import { starColorFor } from "./display.js";
import { fixedDensityRcLabel, fixedDensityLnLabel, fixedDensityDanTickLabel } from "../fixedDensity/index.js";
import { getInterpolatedPlaybackTime } from "./graph.js";
import { ribbonSegments } from "./rcintervals.js";

const VB_W = 492;
const VB_H = 104;
const PAD_L = 34;
const PAD_R = 4;
const PAD_T = 6;
const PAD_B = 8;
const PLOT_W = VB_W - PAD_L - PAD_R;
const PLOT_H = VB_H - PAD_T - PAD_B;
const MAX_BUCKETS = 360;
const TICK_MIN_GAP_PX = 10;
const SVG_NS = "http://www.w3.org/2000/svg";

const CHART_CFG = APP_CONFIG.fixedDensityCharts || {};
const TICKS_ABOVE = Number.isFinite(CHART_CFG.ticksAbove) ? CHART_CFG.ticksAbove : 4;
const TICKS_BELOW = Number.isFinite(CHART_CFG.ticksBelow) ? CHART_CFG.ticksBelow : 8;
const TICK_LINE_GAP = Number.isFinite(CHART_CFG.minTickLineGapPx) ? CHART_CFG.minTickLineGapPx : 4;
const TICK_LABEL_GAP = Number.isFinite(CHART_CFG.minTickLabelGapPx) ? CHART_CFG.minTickLabelGapPx : 7;
const MAX_TICK_LABELS = Number.isFinite(CHART_CFG.maxTickLabels) ? CHART_CFG.maxTickLabels : 10;

const charts = { rc: null, ln: null };
let rafId = 0;
let gradientSeq = 0;

function valueColor(value) {
    const v = Number(value);
    const star = Number.isFinite(v) ? (v < 5 ? 1 : 1 + (v - 5) * 0.5) : 1;
    return starColorFor(star);
}

// 深色（高段位接近纯黑）的标注线/文字在深色背景上不可读：标签文字统一向白色提亮。
function lighten(hex, amount) {
    const h = String(hex).replace("#", "");
    if (h.length !== 6) return hex;
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    const mix = (c) => Math.round(c + (255 - c) * amount);
    const toHex = (c) => c.toString(16).padStart(2, "0");
    return `#${toHex(mix(r))}${toHex(mix(g))}${toHex(mix(b))}`;
}

function svgEl(name, attrs) {
    const el = document.createElementNS(SVG_NS, name);
    if (attrs) {
        for (const [key, value] of Object.entries(attrs)) {
            el.setAttribute(key, String(value));
        }
    }
    return el;
}

function percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    const pos = (sorted.length - 1) * Math.max(0, Math.min(1, p));
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function buildChartData(sequence, ticks, labelFn, width = VB_W, anchor = null, timeRange = null, markerTimes = null, nearTimes = null) {
    if (!sequence || !Array.isArray(sequence.times) || sequence.times.length < 2) {
        return null;
    }
    const plotW = width - PAD_L - PAD_R;
    const times = sequence.times;
    const raws = sequence.raw;
    const values = sequence.value;
    const n = times.length;
    if (!Array.isArray(raws) || raws.length !== n) return null;

    const order = new Array(n);
    for (let i = 0; i < n; i += 1) order[i] = i;
    order.sort((a, b) => times[a] - times[b]);

    // RC / LN 两张图共用同一个时间窗口，横向坐标才能一一对应；
    // 传入 timeRange 时以它为基准（谱面整体时间跨度），否则退回本序列首尾。
    const seqT0 = times[order[0]];
    const seqT1 = times[order[n - 1]];
    const rangeT0 = Number(timeRange?.t0);
    const rangeT1 = Number(timeRange?.t1);
    const hasRange = Number.isFinite(rangeT0) && Number.isFinite(rangeT1) && rangeT1 - rangeT0 > 1e-6;
    const t0 = hasRange ? rangeT0 : seqT0;
    const t1 = hasRange ? rangeT1 : seqT1;
    if (!Number.isFinite(t0) || !Number.isFinite(t1) || t1 - t0 < 1e-6) {
        return null;
    }

    const rawSorted = new Float64Array(n);
    for (let i = 0; i < n; i += 1) rawSorted[i] = raws[i];
    rawSorted.sort();
    const p995 = percentile(rawSorted, 0.995);
    // 纵轴上限取三者最大值：平滑后曲线 p99.5、未平滑 p99.5 提示（保持轴稳定）、
    // 以及锚线以上 3%（避免平滑压低峰值后锚线被挤出画面）。
    const anchorValue = Number(anchor?.value);
    const anchorRaw = Number(anchor?.raw);
    const yMaxHint = Number(sequence.yMax);
    // 纵轴上限需同时容纳曲线峰值、锚线、以及「锚线以上 TICKS_ABOVE 档」的刻度线，
    // 否则难度波动较大时上方参考线会被裁掉。
    let tickTopRaw = 0;
    if (Number.isFinite(anchorValue) && Array.isArray(ticks)) {
        const tvMax = Math.floor(anchorValue) + TICKS_ABOVE;
        for (const tick of ticks) {
            const tv = Number(tick.value);
            const tr = Number(tick.raw);
            if (Number.isFinite(tv) && Number.isFinite(tr) && tr > 0 && tv <= tvMax && tr > tickTopRaw) {
                tickTopRaw = tr;
            }
        }
    }
    const yMax = Math.max(
        p995 > 0 ? p995 * 1.08 : 0,
        Number.isFinite(yMaxHint) && yMaxHint > 0 ? yMaxHint : 0,
        Number.isFinite(anchorRaw) && anchorRaw > 0 ? anchorRaw * 1.03 : 0,
        tickTopRaw > 0 ? tickTopRaw * 1.06 : 0,
        1e-6,
    );

    const timeSpan = t1 - t0;
    const buckets = Math.max(2, Math.min(MAX_BUCKETS, n));
    const isSmoothed = sequence.smoothed === true;
    const bucketRaw = new Float64Array(buckets).fill(-Infinity);
    const bucketSum = new Float64Array(buckets).fill(0);
    const bucketCnt = new Int32Array(buckets).fill(0);
    const bucketIdx = new Int32Array(buckets).fill(-1);
    for (let k = 0; k < n; k += 1) {
        const i = order[k];
        const raw = Number(raws[i]);
        if (!Number.isFinite(raw)) continue;
        const b = Math.min(buckets - 1, Math.max(0, Math.floor(((times[i] - t0) / timeSpan) * (buckets - 1))));
        if (isSmoothed) {
            // 已经平滑过的曲线：桶内取均值，避免峰值重新变尖
            bucketSum[b] += raw;
            bucketCnt[b] += 1;
        } else if (raw > bucketRaw[b]) {
            bucketRaw[b] = raw;
        }
        if (bucketIdx[b] < 0) bucketIdx[b] = i;
    }

    const xs = [];
    const ys = [];
    const rawOut = [];
    for (let b = 0; b < buckets; b += 1) {
        const i = bucketIdx[b];
        if (i < 0) continue;
        const raw = Math.max(0, isSmoothed
            ? (bucketCnt[b] > 0 ? bucketSum[b] / bucketCnt[b] : 0)
            : bucketRaw[b]);
        xs.push(PAD_L + ((times[i] - t0) / timeSpan) * plotW);
        ys.push(PAD_T + PLOT_H * (1 - Math.min(raw, yMax) / yMax));
        rawOut.push(raw);
    }
    if (xs.length < 2) return null;

    // 整谱 value 锚线（与左侧胶囊同值）：y 位置由序列的 75% 分位平移后的标定决定，
    // 使图上数值与胶囊值严格对应；整数刻度只画到锚线以下，避免出现远高于胶囊的数值。
    const hasAnchor = Number.isFinite(anchorValue) && Number.isFinite(anchorRaw)
        && anchorRaw > 0 && anchorRaw <= yMax;
    const anchorData = hasAnchor
        ? {
            raw: anchorRaw,
            y: PAD_T + PLOT_H * (1 - anchorRaw / yMax),
            label: String(anchor.label || ""),
            color: lighten(valueColor(anchorValue), 0.45),
            labelColor: lighten(valueColor(anchorValue), 0.75),
        }
        : null;

    // raw → value 的单调映射（用整数刻度 + 锚点，按 ln(raw) 分段线性），
    // 供「按线所在高度着色」使用：颜色只取决于 y，而不是线段末端取值。
    const mapPoints = [];
    for (const tick of ticks || []) {
        const tv = Number(tick.value);
        const tr = Number(tick.raw);
        if (Number.isFinite(tv) && Number.isFinite(tr) && tr > 0) {
            mapPoints.push({ v: tv, r: Math.log(tr) });
        }
    }
    if (hasAnchor) {
        mapPoints.push({ v: anchorValue, r: Math.log(anchorRaw) });
    }
    mapPoints.sort((a, b) => a.r - b.r);
    const valueAtRaw = (raw) => {
        if (!(raw > 0) || mapPoints.length === 0) return Number.NaN;
        const lr = Math.log(raw);
        if (mapPoints.length === 1) return mapPoints[0].v;
        if (lr <= mapPoints[0].r) {
            const a = mapPoints[0];
            const b = mapPoints[1];
            return a.v + ((lr - a.r) / (b.r - a.r || 1)) * (b.v - a.v);
        }
        const last = mapPoints.length - 1;
        if (lr >= mapPoints[last].r) {
            const a = mapPoints[last - 1];
            const b = mapPoints[last];
            return a.v + ((lr - a.r) / (b.r - a.r || 1)) * (b.v - a.v);
        }
        for (let i = 1; i < mapPoints.length; i += 1) {
            const a = mapPoints[i - 1];
            const b = mapPoints[i];
            if (lr <= b.r) {
                const t = (lr - a.r) / (b.r - a.r || 1);
                return a.v + t * (b.v - a.v);
            }
        }
        return mapPoints[last].v;
    };

    const segmentColors = [];
    for (let i = 1; i < rawOut.length; i += 1) {
        const midRaw = (rawOut[i - 1] + rawOut[i]) * 0.5;
        segmentColors.push(valueColor(valueAtRaw(midRaw)));
    }

    // 沿线颜色用「竖向线性渐变」实现：颜色只由 y（即 height→value）决定，
    // 因此折线在任意位置的色相随高度连续变化，陡峭突变处也不会被切成
    // 一段一段不同颜色的折线段（不再有颜色撕裂）。
    let gradientStops = null;
    if (mapPoints.length > 0) {
        gradientStops = [];
        const GRADIENT_STEPS = 64;
        for (let i = 0; i <= GRADIENT_STEPS; i += 1) {
            const t = i / GRADIENT_STEPS;
            const raw = Math.max(yMax * (1 - t), yMax * 0.0001);
            gradientStops.push({ offset: t, color: valueColor(valueAtRaw(raw)) });
        }
    }

    // 整数 value 标注线：锚线以下按高度自适应，尽量多画线；标签只给不重叠的一部分。
    // 参考线一直画到绘图区顶端（不再限制锚线以上档数），避免顶部大片空白影响观感。
    const maxTickValue = Infinity;
    const minTickValue = Number.isFinite(anchorValue) ? Math.max(0, Math.floor(anchorValue) - TICKS_BELOW) : -Infinity;
    const tickAll = [];
    const seenTickLabels = new Set();
    const tickMinRaw = yMax * 0.01;
    for (const tick of ticks || []) {
        const tv = Number(tick.value);
        if (Number.isFinite(maxTickValue) && tv > maxTickValue) continue;
        if (Number.isFinite(minTickValue) && tv < minTickValue) continue;
        if (Number.isFinite(anchorValue) && tv === Math.round(anchorValue)) continue;
        const raw = Number(tick.raw);
        if (!Number.isFinite(raw) || raw <= 0 || raw > yMax || raw < tickMinRaw) continue;
        const y = PAD_T + PLOT_H * (1 - raw / yMax);
        const label = labelFn ? labelFn(tv) : "";
        if (!label) continue;
        // 段位名标签（6K/7K）是分档的：相邻刻度值可能落在同一段位，重复标签只保留最上面一个。
        if (seenTickLabels.has(label)) continue;
        seenTickLabels.add(label);
        const color = valueColor(tv);
        tickAll.push({ y, label, color: lighten(color, 0.35), labelColor: lighten(color, 0.6) });
    }
    tickAll.sort((a, b) => a.y - b.y);

    const tickLines = [];
    let lastLineY = -Infinity;
    for (const tick of tickAll) {
        if (tick.y < PAD_T + 1 || tick.y > PAD_T + PLOT_H - 1) continue;
        if (tick.y - lastLineY < TICK_LINE_GAP) continue;
        tickLines.push(tick);
        lastLineY = tick.y;
    }

    const tickLabels = [];
    let lastLabelY = -Infinity;
    for (const tick of tickLines) {
        if (tickLabels.length >= MAX_TICK_LABELS) break;
        if (tick.y - lastLabelY < TICK_LABEL_GAP) continue;
        if (anchorData && Math.abs(tick.y - anchorData.y) < 6) continue;
        tickLabels.push(tick);
        lastLabelY = tick.y;
    }

    // 重叠键红线 / 近似重叠黄线位置（RC / LN 各自一组）
    const toXs = (arr) => {
        const out = [];
        if (!Array.isArray(arr)) return out;
        for (const raw of arr) {
            const t = Number(raw);
            if (!Number.isFinite(t) || t < t0 || t > t1) continue;
            out.push(PAD_L + ((t - t0) / (t1 - t0)) * plotW);
        }
        return out;
    };
    const markerXs = toXs(markerTimes);
    const nearXs = toXs(nearTimes);

    return {
        t0,
        t1,
        xs,
        ys,
        markerXs,
        nearXs,
        segmentColors,
        gradientStops,
        tickLines,
        tickLabels,
        anchor: anchorData,
        width,
        plotW,
    };
}

function drawChart(svg, data) {
    if (!svg) return;
    const width = data.width || VB_W;
    svg.setAttribute("viewBox", `0 0 ${width} ${VB_H}`);
    while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
    }

    // 先画难度锚线，再画图表参考线：锚线在下、参考线在上，避免锚线遮住被覆盖的刻度线。
    if (data.anchor) {
        svg.appendChild(svgEl("line", {
            x1: PAD_L, x2: width - PAD_R, y1: data.anchor.y, y2: data.anchor.y,
            class: "fd-assign-anchor", stroke: data.anchor.color,
        }));
    }
    for (const tick of data.tickLines || []) {
        svg.appendChild(svgEl("line", {
            x1: PAD_L, x2: width - PAD_R, y1: tick.y, y2: tick.y,
            class: "fd-assign-tick", stroke: tick.color,
        }));
    }
    for (const tick of data.tickLabels || []) {
        const label = svgEl("text", {
            x: PAD_L - 2, y: tick.y + 2.2, class: "fd-assign-tick-label",
            fill: tick.labelColor || tick.color, "text-anchor": "end",
        });
        label.textContent = tick.label;
        svg.appendChild(label);
    }
    if (data.anchor) {
        const anchorLabel = svgEl("text", {
            x: PAD_L - 2, y: data.anchor.y + 2.2, class: "fd-assign-anchor-label",
            fill: data.anchor.labelColor, "text-anchor": "end",
        });
        anchorLabel.textContent = data.anchor.label;
        svg.appendChild(anchorLabel);
    }

    // 折线路径：整条线一个 path，颜色交给竖向渐变，保证颜色随高度连续过渡。
    const pathParts = [];
    for (let i = 0; i < data.xs.length; i += 1) {
        pathParts.push(`${i === 0 ? "M" : "L"} ${data.xs[i].toFixed(2)} ${data.ys[i].toFixed(2)}`);
    }
    const linePath = pathParts.join(" ");

    // 先铺一层浅色描边（光晕），让深色 value（高段位，胶囊背景接近纯黑）也能看清
    const haloGroup = svgEl("g", { class: "fd-assign-halo" });
    haloGroup.appendChild(svgEl("path", { d: linePath }));
    svg.appendChild(haloGroup);

    if (Array.isArray(data.gradientStops) && data.gradientStops.length > 1) {
        const gradientId = `fd-assign-grad-${gradientSeq += 1}`;
        const defs = svgEl("defs", {});
        const gradient = svgEl("linearGradient", {
            id: gradientId,
            gradientUnits: "userSpaceOnUse",
            x1: 0,
            y1: PAD_T,
            x2: 0,
            y2: PAD_T + PLOT_H,
        });
        for (const stop of data.gradientStops) {
            gradient.appendChild(svgEl("stop", {
                offset: stop.offset.toFixed(4),
                "stop-color": stop.color,
            }));
        }
        defs.appendChild(gradient);
        svg.appendChild(defs);
        svg.appendChild(svgEl("path", {
            d: linePath,
            class: "fd-assign-seg",
            stroke: `url(#${gradientId})`,
        }));
    } else {
        const lineGroup = svgEl("g", {});
        for (let i = 1; i < data.xs.length; i += 1) {
            lineGroup.appendChild(svgEl("path", {
                d: `M ${data.xs[i - 1].toFixed(2)} ${data.ys[i - 1].toFixed(2)} L ${data.xs[i].toFixed(2)} ${data.ys[i].toFixed(2)}`,
                class: "fd-assign-seg",
                stroke: data.segmentColors[i - 1] || "#a6e0ff",
            }));
        }
        svg.appendChild(lineGroup);
    }

    // 近似重叠：黄色竖线（仅图表标注，不影响数值）。
    if (Array.isArray(data.nearXs) && data.nearXs.length > 0) {
        const nearGroup = svgEl("g", { class: "fd-assign-near-overlap" });
        for (const x of data.nearXs) {
            nearGroup.appendChild(svgEl("line", {
                x1: x.toFixed(2), x2: x.toFixed(2),
                y1: PAD_T, y2: PAD_T + PLOT_H,
                class: "fd-assign-overlap-line fd-assign-near-line",
                stroke: "#ffd60a",
            }));
        }
        svg.appendChild(nearGroup);
    }

    // 重叠键：红色竖线（每根线代表 n 个叠放物件中的 n-1 处重叠）。
    if (Array.isArray(data.markerXs) && data.markerXs.length > 0) {
        const markerGroup = svgEl("g", { class: "fd-assign-overlap" });
        for (const x of data.markerXs) {
            markerGroup.appendChild(svgEl("line", {
                x1: x.toFixed(2), x2: x.toFixed(2),
                y1: PAD_T, y2: PAD_T + PLOT_H,
                class: "fd-assign-overlap-line",
                stroke: "#ff3b30",
            }));
        }
        svg.appendChild(markerGroup);
    }

    const cursor = svgEl("line", {
        x1: 0, x2: 0, y1: PAD_T, y2: PAD_T + PLOT_H,
        class: "fd-assign-cursor",
    });
    cursor.style.display = "none";
    svg.appendChild(cursor);
    data.cursor = cursor;
}

export function updateAssignmentCursors() {
    const time = getInterpolatedPlaybackTime();
    const playing = state.isInPlayState && Number.isFinite(time);
    for (const key of ["rc", "ln"]) {
        const chart = charts[key];
        if (!chart || !chart.cursor) continue;
        if (!playing) {
            chart.cursor.style.display = "none";
            continue;
        }
        const t = Math.max(chart.t0, Math.min(time, chart.t1));
        const x = PAD_L + ((t - chart.t0) / (chart.t1 - chart.t0)) * (chart.plotW || PLOT_W);
        chart.cursor.setAttribute("x1", x.toFixed(2));
        chart.cursor.setAttribute("x2", x.toFixed(2));
        chart.cursor.style.display = "";
    }
}

function startAnimationLoop() {
    if (rafId) return;
    const tick = () => {
        updateAssignmentCursors();
        rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
}

function stopAnimationLoop() {
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
    }
}

/** RC 图下方的键型色条：Jack 红 / Chordstream 绿 / Stream 蓝。 */
function renderRcRibbon(input, timeRange = null) {
    const el = document.getElementById("fd-assign-rc-ribbon");
    if (!el) return;
    let segs = [];
    let srcT0 = Number.NaN;
    let srcT1 = Number.NaN;
    try {
        if (Array.isArray(input)) {
            // 旧缓存只有分段数组（比例基于它自己的时间范围），无法重投影。
            segs = input;
        } else if (input && Array.isArray(input.segments)) {
            segs = input.segments;
            srcT0 = Number(input.t0);
            srcT1 = Number(input.t1);
        } else if (input && Array.isArray(input.segs)) {
            segs = ribbonSegments(input);
            srcT0 = Number(input.t0);
            srcT1 = Number(input.t1);
        }
    } catch (_) {
        segs = [];
    }
    // 图表横轴使用 RC/LN 序列并集；色条按同一窗口重新投影，保证与曲线对齐。
    const dstT0 = Number(timeRange?.t0);
    const dstT1 = Number(timeRange?.t1);
    const canRemap = Number.isFinite(srcT0) && Number.isFinite(srcT1) && srcT1 - srcT0 > 1e-6
        && Number.isFinite(dstT0) && Number.isFinite(dstT1) && dstT1 - dstT0 > 1e-6;
    if (canRemap) {
        const srcSpan = srcT1 - srcT0;
        const dstSpan = dstT1 - dstT0;
        segs = segs
            .map((seg) => {
                const left = (srcT0 + seg.left * srcSpan - dstT0) / dstSpan;
                const width = seg.width * srcSpan / dstSpan;
                return { ...seg, left, width };
            })
            .filter((seg) => seg.width > 0 && seg.left < 1 && seg.left + seg.width > 0);
    }
    const wrap = el.parentElement;
    el.textContent = "";
    if (!segs.length) {
        el.hidden = true;
        if (wrap) wrap.hidden = true;
        return;
    }
    const frag = document.createDocumentFragment();
    for (const seg of segs) {
        const cell = document.createElement("i");
        cell.className = `seg ${seg.type === "Jacks" ? "jack" : seg.type === "Chordstream" ? "chord" : seg.type === "Stream" ? "stream" : "gap"}`;
        cell.style.left = `${(seg.left * 100).toFixed(3)}%`;
        cell.style.width = `${(seg.width * 100).toFixed(3)}%`;
        if (seg.type !== "gap" && seg.type !== "Jacks" && seg.type !== "Chordstream" && seg.type !== "Stream") {
            cell.style.background = "transparent";
        }
        frag.appendChild(cell);
    }
    el.appendChild(frag);
    el.hidden = false;
    if (wrap) wrap.hidden = false;
}

function renderOverlapNote(overlaps) {
    const el = document.getElementById("fd-assign-overlap-note");
    const listEl = document.getElementById("fd-assign-overlap-list");
    const total = Number(overlaps?.total);
    const nearTotal = Number(overlaps?.nearTotal);
    const hasRed = Number.isFinite(total) && total > 0;
    const hasNear = Number.isFinite(nearTotal) && nearTotal > 0;
    if (!hasRed && !hasNear) {
        if (el) { el.textContent = ""; el.hidden = true; el.classList.remove("near"); }
        if (listEl) { listEl.textContent = ""; listEl.hidden = true; }
        return;
    }
    if (el) {
        if (hasRed) {
            el.textContent = `存在${Math.round(total)}处重叠键，请检查谱面`;
            el.classList.remove("near");
        } else {
            el.textContent = `存在${Math.round(nearTotal)}处近似重叠键，请检查谱面`;
            el.classList.add("near");
        }
        el.hidden = false;
    }
    renderOverlapList(overlaps, listEl);
}

function renderOverlapList(overlaps, listEl) {
    if (!listEl) return;
    const labels = overlaps?.labels;
    const shown = Array.isArray(labels?.shown) ? labels.shown : [];
    if (shown.length === 0) {
        listEl.textContent = "";
        listEl.hidden = true;
        return;
    }
    listEl.textContent = "";
    const frag = document.createDocumentFragment();
    for (const item of shown) {
        const row = document.createElement("div");
        row.className = `fd-assign-overlap-row ${item.kind === "near" ? "near" : "red"}`;
        row.textContent = item.text;
        frag.appendChild(row);
    }
    if (labels.truncated) {
        const more = document.createElement("div");
        more.className = `fd-assign-overlap-row fd-assign-overlap-more ${labels.shown[0]?.kind === "near" ? "near" : "red"}`;
        more.textContent = `… 其余 ${Math.max(0, Number(labels.total || 0) - shown.length)} 处未显示`;
        frag.appendChild(more);
    }
    listEl.appendChild(frag);
    listEl.hidden = false;
}

export function hideAssignmentCharts() {
    const wrap = document.getElementById("fd-assign-charts");
    if (wrap) wrap.hidden = true;
    const ribbon = document.getElementById("fd-assign-rc-ribbon");
    if (ribbon) {
        ribbon.textContent = "";
        ribbon.hidden = true;
    }
    const note = document.getElementById("fd-assign-overlap-note");
    if (note) {
        note.textContent = "";
        note.hidden = true;
    }
    const list = document.getElementById("fd-assign-overlap-list");
    if (list) {
        list.textContent = "";
        list.hidden = true;
    }
    charts.rc = null;
    charts.ln = null;
    stopAnimationLoop();
    if (mainCardEl) mainCardEl.classList.remove("fd-charts");
}

export function renderAssignmentCharts(data) {
    const wrap = document.getElementById("fd-assign-charts");
    if (!wrap) return;
    if (!data) {
        hideAssignmentCharts();
        return;
    }
    const rcWrap = document.getElementById("fd-assign-rc");
    const lnWrap = document.getElementById("fd-assign-ln");
    const rcSvg = document.getElementById("fd-assign-rc-svg");
    const lnSvg = document.getElementById("fd-assign-ln-svg");

    const lnSeq = data.ln;
    // 序列里现在混有「非 LN 物件 = 0」的补点，因此用真实 LN 事件数判断是否显示 LN 图。
    const lnEventCount = Number(lnSeq?.eventCount);
    const lnSampleCount = Array.isArray(lnSeq?.times) ? lnSeq.times.length : 0;
    // 事件数够但难度本身≈0 的谱面（例如 18th Dan：LN% 0.5%、lnValue≈7e-8）
    // 画出来只会是一条贴着底部的 ln0 直线，白占一块卡片高度，直接隐藏。
    const lnAnchorValue = Number(data.lnAnchor?.value);
    // 新版 LN 映射层下限约 0.75（无 LN 谱），阈值抬到 1.0，避免「几乎无 LN」的谱出现空白 LN 面板。
    const hasLnLevel = !Number.isFinite(lnAnchorValue) || lnAnchorValue >= 1.0;
    const lnShare = Number(data.lnShare);
    const hasLnShare = !Number.isFinite(lnShare) || lnShare >= 0.02;
    const hasLn = hasLnLevel && hasLnShare && Boolean(lnSeq && (Number.isFinite(lnEventCount) ? lnEventCount : lnSampleCount) >= 8);
    const keyCount = Number(data.keyCount);
    const hasDanTier = Boolean(fixedDensityDanTickLabel(0, keyCount));
    const rcLabelFn = (v) => (hasDanTier ? fixedDensityDanTickLabel(v, keyCount) : fixedDensityRcLabel(v, keyCount));
    // 统一时间窗口：取 RC / LN 序列首尾的并集，保证两张图横向一一对应。
    let rangeT0 = Infinity;
    let rangeT1 = -Infinity;
    for (const seq of [data.rc, data.ln]) {
        const tArr = seq?.times;
        if (!Array.isArray(tArr) || tArr.length === 0) continue;
        const a = Number(tArr[0]);
        const b = Number(tArr[tArr.length - 1]);
        if (Number.isFinite(a)) rangeT0 = Math.min(rangeT0, a);
        if (Number.isFinite(b)) rangeT1 = Math.max(rangeT1, b);
    }
    const timeRange = (Number.isFinite(rangeT0) && Number.isFinite(rangeT1) && rangeT1 - rangeT0 > 1e-6)
        ? { t0: rangeT0, t1: rangeT1 }
        : null;
    const overlaps = data.overlaps || null;
    const rcMarkerTimes = Array.isArray(overlaps?.rc) ? overlaps.rc : null;
    const lnMarkerTimes = Array.isArray(overlaps?.ln) ? overlaps.ln : null;
    const rcNearTimes = Array.isArray(overlaps?.nearRc) ? overlaps.nearRc : null;
    const lnNearTimes = Array.isArray(overlaps?.nearLn) ? overlaps.nearLn : null;
    const rcData = buildChartData(data.rc, data.rcTicks, rcLabelFn, VB_W, data.rcAnchor, timeRange, rcMarkerTimes, rcNearTimes);
    const lnLabelFn = (v) => (hasDanTier ? fixedDensityDanTickLabel(v, keyCount) : fixedDensityLnLabel(v, keyCount));
    const lnData = hasLn
        ? buildChartData(lnSeq, data.lnTicks, lnLabelFn, VB_W, data.lnAnchor, timeRange, lnMarkerTimes, lnNearTimes)
        : null;
    if (!rcData && !lnData) {
        hideAssignmentCharts();
        return;
    }

    charts.rc = rcData;
    charts.ln = lnData;
    if (rcWrap) rcWrap.hidden = !rcData;
    if (lnWrap) lnWrap.hidden = !lnData;
    if (rcData) drawChart(rcSvg, rcData);
    if (lnData) drawChart(lnSvg, lnData);
    renderRcRibbon(
        rcData ? (data.rcRibbon || data.rcIntervals) : null,
        rcData ? { t0: rcData.t0, t1: rcData.t1 } : null,
    );
    renderOverlapNote(overlaps);
    wrap.classList.toggle("single", !rcData || !lnData);
    wrap.hidden = false;
    if (mainCardEl) mainCardEl.classList.add("fd-charts");
    startAnimationLoop();
    updateAssignmentCursors();
}
