import test from "node:test";
import assert from "node:assert/strict";

import { buildOverlapLabels } from "../nonebot_plugin_rework/tosu_plugin/js/app/beatPosition.js";
import {
    buildRcIntervals,
    ribbonSegments,
} from "../nonebot_plugin_rework/tosu_plugin/js/app/rcintervals.js";
import { calculateFixedDensity } from "../nonebot_plugin_rework/tosu_plugin/js/fixedDensity/index.js";


function makeBeatmap(lanes, times) {
    const keyCount = Math.max(...lanes) + 1;
    const lines = [
        "osu file format v14",
        "[General]",
        "Mode:3",
        "[Difficulty]",
        `CircleSize:${keyCount}`,
        "[TimingPoints]",
        "0,500,4,2,1,50,1,0",
        "[HitObjects]",
    ];

    lanes.forEach((lane, index) => {
        const x = Math.floor(((lane + 0.5) * 512) / keyCount);
        lines.push(`${x},192,${times[index]},1,0,0:0:0:0:`);
    });
    return lines.join("\n");
}


test("RC ribbon classifies Jack, Stream, and Chordstream intervals", () => {
    const cases = [
        { lanes: [0, 0, 0, 0], expected: "Jacks" },
        { lanes: [0, 1, 2, 3], expected: "Stream" },
        { lanes: [0, 1, 0, 2], expected: "Chordstream" },
    ];

    for (const { lanes, expected } of cases) {
        const intervals = buildRcIntervals(makeBeatmap(lanes, [0, 100, 200, 300]));

        assert.ok(intervals);
        assert.ok(intervals.segs.some((segment) => segment.type === expected));
        assert.ok(ribbonSegments(intervals).some((segment) => segment.type === expected));
    }
});


test("Fixed Density reports exact and near overlaps with beat-position labels", () => {
    const beatmap = makeBeatmap([0, 0, 0, 0, 0, 1], [1000, 1000, 1500, 1502, 2000, 2500]);
    const result = calculateFixedDensity(beatmap);

    assert.equal(result.overlaps.total, 1);
    assert.equal(result.overlaps.rc[0], 1000);
    assert.equal(result.overlaps.nearTotal, 1);
    assert.equal(result.overlaps.nearRc[0], 1502);

    const redLabels = buildOverlapLabels(result.overlaps, beatmap);
    assert.equal(redLabels.red, 1);
    assert.equal(redLabels.near, 1);
    assert.equal(redLabels.shown[0].kind, "red");
    assert.match(redLabels.shown[0].text, /轨 1/);

    const nearLabels = buildOverlapLabels({
        ...result.overlaps,
        rc: [],
        total: 0,
        positions: { ...result.overlaps.positions, rc: [] },
    }, beatmap);
    assert.equal(nearLabels.shown[0].kind, "near");
    assert.equal(nearLabels.total, 1);
});
