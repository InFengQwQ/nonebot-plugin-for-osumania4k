// js/fixedDensity/index.js
// Fixed Density beta_1.0.5（含 LN 参考难度）— osu!mania 难度计算。
//
// 本副本在 Fixed Density v1 基础上加入：
//   * no-chord 调参后的 RC 参数（E=0，映射曲线重新拟合）；
//   * 双手合并层 cross-hand 权重（γ=0，当前关闭）；
//   * 单手两轨左右不均时间衰减序列 + 弱侧门控；
//   * 12.5+ 映射段加点/移点；
//   * 其余 RC/LN 逻辑与 v1 相同。
//
// 关于「被参数取消的逻辑」（不实现）：
//   * soft_beta > 0 ⇒ 合并算子为 softmax 加权平均 w·a+(1−w)·b；
//     RC 通道的 Lp 范数分支与 merge_power / merge2_power / contrib_power 不参与。
//   * cross_lane_mix = 0 ⇒ 贡献间距仅取同轨间距，跨轨混合逻辑不参与。
//   * irr_local_gain = 0、irr_gain_pair = 0 ⇒ 全谱熵@贡献层、轨道对熵@配对合并层不参与。
//   * ln_gain = 0 ⇒ LN 通道（松手流 / 控手 / 按住占用）关闭，n_LN 不参与 n。
//   * 拟合模型 C（irr 系数）= 0、E（chord 系数）= 0；LN 倍率的 γ_cross = 0。
//
// DT/HT/可调变速：所有物件时间戳按 1/rate 缩放后再参与计算（rate=1.5 为 DT、0.75 为 HT，
// lazer 任意正速同理）。
//
// 模块约束：DOM-free / state-free，Node 与浏览器一致。

import { OsuFileParser } from "../parser/osuFileParser.js";
import { APP_CONFIG } from "../../config.js";

// 难度-进度图的全局标定（可在 config.js 的 fixedDensityCharts 中调整）
const CHART_CFG = (APP_CONFIG && APP_CONFIG.fixedDensityCharts) || {};
const CHART_ANCHOR_Q_RC = Number.isFinite(CHART_CFG.anchorQuantileRc) ? CHART_CFG.anchorQuantileRc : 0.75;
const CHART_ANCHOR_Q_LN = Number.isFinite(CHART_CFG.anchorQuantileLn) ? CHART_CFG.anchorQuantileLn : 0.88;
// 锚点标定模式：
// - "mean"（默认）：平滑曲线的加权均值对齐胶囊值，曲线围绕锚线，不会被窗口均值拉低；
// - "quantile"：旧行为，把逐物件 value 的 anchorQuantile* 分位平移到胶囊值。
const CHART_ANCHOR_MODE = CHART_CFG.anchorMode === "quantile" ? "quantile" : "mean";
// 曲线平滑：默认 400ms 平均窗口、每 100ms 采样（config.js → fixedDensityCharts 可调）
const CHART_SMOOTH_WINDOW_MS = Number.isFinite(CHART_CFG.smoothWindowMs) ? CHART_CFG.smoothWindowMs : 400;
const CHART_SMOOTH_STEP_MS = Number.isFinite(CHART_CFG.smoothStepMs) ? CHART_CFG.smoothStepMs : 100;
// LN 曲线单独用更大的窗口：LN 取样点稀疏、赋值本身波动大，400ms 时曲线起伏过陡；
// 默认 1000ms 窗口（步长仍 100ms），让 LN 曲线更接近「段落平均」而不是瞬时尖刺。
const CHART_LN_SMOOTH_WINDOW_MS = Number.isFinite(CHART_CFG.lnSmoothWindowMs) ? CHART_CFG.lnSmoothWindowMs : 1000;
const CHART_LN_SMOOTH_STEP_MS = Number.isFinite(CHART_CFG.lnSmoothStepMs) ? CHART_CFG.lnSmoothStepMs : CHART_SMOOTH_STEP_MS;

const TIME_EPS = 1e-6;
const INV_DIST_SCALE = 1000.0; // 相邻物件间距倒数缩放：1000 / dt
const OVERLAP_PENALTY = 20000.0; // 同轨同时间戳重叠物件的额外惩罚
const N_PRECISION = 1e-12; // n 值二分精度
const TYPE_HOLD = 1 << 7; // osu!mania 长条类型位

// ---- beta_1.0.5 冻结参数（weakgate_smallweight RC + beta_1.0.3 的 LN 重标定基线）----
const HALF_LIFE_MS = 306.625;
const QUANTILE = 0.94012709;
const EXTRA_RATE = 1.7044655999999996;
const EXTRA_B = 2.337995528948387;
const EXTRA_LOG_BASE = 4.012384;
const SOFT_BETA = 0.346887; // > 0 ⇒ softmax 加权平均（Lp 范数分支不参与）
const IRR_GAIN_LANE = 0.1; // 单轨局部熵 @ 贡献层
const IRR_GAIN_GLOBAL = -0.06; // 全谱局部熵 @ 最终合并层
const IRR_GAIN_CROSS = 0.0; // 当前全量调参关闭 cross-hand
const IRR_CROSS_SCALE = 1.0; // s = 1 − exp(−(c_bar−1)/scale)
const IRR_CROSS_ALPHA = 2.5; // c_eff = max(L,R) + α·min(L,R)
const IRR_CROSS_DENSITY = 2.48; // s *= (1 − f_bar)^density
const IRR_CROSS_WINDOW = 16; // 与 irr 同窗口

// 单手两轨左右不均时间衰减序列 + 弱侧门控
const HAND_IMB_ETA = 0.4;
const HAND_IMB_TAU_MS = 60.0;
const HAND_IMB_M0 = 1.875;
const HAND_IMB_NPS0 = 3.05;
const HAND_IMB_HALF_LIFE_MS = 5000.0; // weakgate_smallweight：4350 → 5000
const HAND_IMB_EPS = 1e-6;
const WEAK_GATE_Q0 = 1.5625;
const WEAK_GATE_IMB_B = 0.35;
const WEAK_GATE_IMB_S = 0.0825;

const LOG_BASE_DENOM = Math.log(EXTRA_LOG_BASE);
const HALF_BETA = 0.5 * SOFT_BETA;
const HAND_NORM_P = 2.0; // 旧（1.0.7-）手内合并层：band 的 p-范数阶数
const HAND_NORM_SCALE = 1.0; // 旧 p-范数合并的幅值系数（beta_1.1.1 起由 HAND_MERGE_SCALE 承担）
// beta_1.1.1：手内 band 合并方式（hand-size 中性）。可选 "softmax"（默认）|"pnorm"|"rms"|"mean"|"max"。
const HAND_MERGE_MODE = "prodadd";
// 行速率项：gPred += ROW_RATE_BETA[k] · ln(rowsPerSec / ROW_RATE_REF)（逐键标定；行/秒 = 行数/跨度）
const ROW_RATE_REF = 8.0;
const ROW_RATE_BETA = { 4: 0.051, 6: 0.048, 7: 0.273 };
// 行合并核宽度（ms）：小于该量级的时间偏移视为同一行的轻微错位
const ROW_RATE_MERGE_MS = 12.0;
// 乘积式手内合并的混合强度：v = activeMean × (1 + λ·ln M)（1.1.2 实验参数，默认 0=关闭）
const MERGE_BLEND_LAMBDA = 0.0;
// 乘积附加项的分键强度（prodadd 模式；4K 单手只有一条 band，乘积因子恒为 1，天然不受影响）
const MERGE_LAMBDA_BY_KEY = { 6: 0.00, 7: 0.35 };
// 和弦尺寸变化成本（手内）：3↔1 交替（跳/衩）比同尺寸轮换更难；仅对 >2 轨手生效，4K 不变。
const MERGE_CHORD_ALT = 0.22;
const HAND_MERGE_SCALE = 1.0;
const HAND_MERGE_BOTH = 0.0;

// ---- 连续映射曲线（beta_1.0.5 = weakgate_smallweight：7.3 / 9.2 衔接点；末段归一化斜率 0.85）----
const MAP_KNOTS = [
    [0.3, 1.6], [2.45, 1.6], [3.85, 0.9], [4.5, 2.0],
    [7.3, 2.0], [9.2, 1.0], [12.05, 0.9], [12.6, 0.6], [14.2, 0.8], [15.6, 1.0],
];
const MAP_COEF = [2.044804, 0.04, 0.389044, 0.320882, 0.327904, 0.04, 0.04, 0.082575, 0.04, 0.144641, 0.279896, 0.135792, -0.039566345789796765];
const MAP_LEAD = 1 + MAP_KNOTS.length + 1; // c + σ_0..σ_K 的个数
const FIT_D_LNN = MAP_COEF[MAP_LEAD]; // ln N 系数（index 12）
// ---- RC v8 结构特征修正层（b1.0.6）----
// g = ln(n) - E*chord - D*lnN + C0 + Σ κ_i·f_i，rc = mapInvert(g)。
// f_i 全部由 lane 时间序列连续计算（无 clip / 无分段），对时长归一。
// 11 个基础特征 + 18 个交互项；κ 全为 0 时等于 b1.0.5.1 基线。
// 标定：RC benchmark 非 LN RMSE 0.597（b1.0.5.1 为 0.448）；
// 合成排列 Q1≈4.9 / Q7≈9.0 / Q8≈11.0，倍速 1.25/1.5/2.0 单调。
// 拟合：_analysis/tune_rc_v8e.py；快照：_analysis/_rc_v8e_wb1.0_wa3.0.json。
const RC_V8_C0 = 0.0;
const RC_V8_TERMS = [
    ["f_both", +1.430392], // 双手同时有键的行速率
    ["f_alt", +0.121555], // 相邻行活跃手切换速率
    ["f_osc", +0.855048], // 手内押数振荡（2121 型 > 0）
    ["f_lanealt", -0.004791], // 手内单键换轨速率
    ["f_jack", +0.418118], // 手内单键同轨（纵连）速率
    ["f_dbl", -0.304301], // 手内双押手行速率
    ["f_bal", -0.514058], // 手间平衡度 2·min/(sum)（单手=0）
    ["f_chord", +0.382890], // 平均每排押数
    ["f_notes", -0.146715], // 总音符速率
    ["f_handmax", -0.370800], // 较密那只手的音符速率
    ["f_bothrow", -0.197292], // 双手同押行占比
    ["ix_both_lanealt", -0.342213], // f_both·f_lanealt
    ["ix_both_bothrow", +1.231120], // f_both·f_bothrow
    ["ix_both_osc", -0.087058], // f_both·f_osc
    ["ix_both_bal", -1.494768], // f_both·f_bal
    ["ix_bal_notes", +0.448173], // f_bal·f_notes
    ["ix_jack_notes", -0.047902], // f_jack·f_notes
    ["ix_jack_hmax", -0.256083], // f_jack·f_handmax
    ["ix_dbl_hmax", +0.365751], // f_dbl·f_handmax
    ["ix_chord_notes", -0.152579], // f_chord·f_notes
    ["ix_hmax_chord", +0.001507], // f_handmax·f_chord
    ["ix_osc_notes", -0.226425], // f_osc·f_notes
    ["ix_lanealt_notes", +0.080601], // f_lanealt·f_notes
    ["ix_notes_bothrow", -0.319042], // f_notes·f_bothrow
    ["ix_jack_chord", -0.191479], // f_jack·f_chord
    ["ix_jack_osc", +1.313499], // f_jack·f_osc
    ["ix_lanealt_jack", -0.261515], // f_lanealt·f_jack
    ["ix_dbl_chord", +0.156590], // f_dbl·f_chord
    ["ix_both_notes", +0.091087], // f_both·f_notes
];
// ---- b1.0.5.1：6K RC 专用映射层（仅 kTotal===6 时启用；4K 仍使用上面的 MAP_KNOTS/MAP_COEF）----
// 与 4K 同构的连续斜率曲线 g(v) = c + Σ σ_j·w_j(v)，σ_j ≥ 0 ⇒ 单调。
// 参数按 44 首 6K RC 基准在 value 空间最小二乘拟合（留一档交叉验证选节点数）：
//   指数基线 M2 RMSE=0.7518 → 本映射层 RMSE=0.3735，R²=0.9860；留一档 CV RMSE=0.4930。
// 复现：_analysis6k/fit_rc_map.py；参数快照：_analysis6k/rc6_map_fit.json。
const MAP6_KNOTS = [
    [1.3, 0.9],
    [3.0, 0.9],
    [4.8, 0.9],
    [6.8, 0.9],
];
const MAP6_COEF = [2.20586, 0.15943, 0.41575, 0.04000, 0.10327, 0.08891, -0.039566345789796765];

// ---- b1.0.6：7K RC 专用映射层（仅 kTotal===7；与 4K/6K 同构的连续斜率曲线）----
// 60 首 7K Dan RC 基准拟合：value 空间 RMSE=0.8954，R²=0.9571，留一档 CV RMSE={'layout': 'K3', 'loo_dan_rmse': 0.9603230202258823, 'candidates': [{'layout': 'K3', 'K': 3, 'rmse': 0.8953513876943069, 'cv': 0.9603230202258823, 'min_slope': 0.08627980559930505, 'coef': [2.7134457835966765, 0.252424344976386, 0.05174847111629122, 0.04000000000000001, 0.11396783657307301, -0.039566345789796765]}, {'layout': 'K4', 'K': 4, 'rmse': 0.889651421964562, 'cv': 0.9744665620168235, 'min_slope': 0.08704710562684582, 'coef': [2.70929319987703, 0.263086733280996, 0.04000000000000001, 0.13519944334793055, 0.04000000000000001, 0.11527273060705141, -0.039566345789796765]}, {'layout': 'K5', 'K': 5, 'rmse': 0.8843703762719594, 'cv': 0.9891810555581965, 'min_slope': 0.0739439411999284, 'coef': [2.7108475353074684, 0.2550457974796599, 0.04000000000000001, 0.2196950495157715, 0.04000000000000001, 0.04000000000000001, 0.11751850248117413, -0.039566345789796765]}, {'layout': 'K6', 'K': 6, 'rmse': 0.8806540421421409, 'cv': 0.9906357647634653, 'min_slope': 0.07564477193171282, 'coef': [2.6902440248818755, 0.30887424910871086, 0.04000000000000001, 0.04000000000000001, 0.23835371274909736, 0.04000000000000001, 0.04000000000000001, 0.1178696261405577, -0.039566345789796765]}, {'layout': 'K6n', 'K': 6, 'rmse': 0.8762140236944764, 'cv': 0.9938440086134521, 'min_slope': 0.06261091570011192, 'coef': [2.67222700460188, 0.3153424546266918, 0.04000000000000001, 0.04000000000000025, 0.24766169213519557, 0.04000000000000001, 0.04000000000000001, 0.11776128606828624, -0.039566345789796765]}, {'layout': 'K8', 'K': 8, 'rmse': 0.8771870811273467, 'cv': 1.031813421623775, 'min_slope': 0.07821419662823663, 'coef': [2.674244562852302, 0.3366374386764853, 0.04000000000000001, 0.04000000000000001, 0.26753059708611, 0.04000000000000001, 0.04000000000000001, 0.15659091978790815, 0.04000000000000001, 0.11878250917904475, -0.039566345789796765]}]}。
// 复现：_analysis7k/fit_rc_map.py（参数快照 _analysis7k/_rc7_map_v13.json）。
const MAP7_KNOTS = [
    [1.8, 0.6],
    [2.8, 0.6],
    [3.8, 0.6],
    [4.8, 0.6],
    [5.8, 0.6],
];
const MAP7_COEF = [2.18397, 0.27576, 0.04000, 0.18527, 0.07340, 0.04000, 0.11959, -0.039566345789796765];

// 局部间隔多样性（irregularity）参数
const IRR_LOCAL_WINDOW = 16;
const IRR_LOCAL_STRIDE = 4;
const IRR_BANDWIDTH = 0.08;
const IRR_INV2H2 = 1.0 / (2.0 * IRR_BANDWIDTH * IRR_BANDWIDTH);
const IRR_LOG_DEN = Math.log(IRR_BANDWIDTH * Math.sqrt(2.0 * Math.PI));
const IRR_ENT_COEF = -0.5 * IRR_INV2H2;

// ---- ln 参考难度参数 ----
const LN_MULT_REL = 2.26321400;
// 逐键 γ_rel（均匀度/时长条件改写后重新标定：4K 1.10×、6K/7K 1.50×）
const LN_MULT_REL_BY_KEY = { 4: 2.48953540, 6: 3.39482090, 7: 3.39482090 }; // γ_rel（v6d 光滑面缝折扣后全量调参）
const LN_MULT_JACK = 9.98250600;
// 逐键 γ_jack（jack 项与映射层联合重标定后的最优：4K 6×、6K 4×、7K 1×）
const LN_MULT_JACK_BY_KEY = { 4: 59.895036, 6: 39.930024, 7: 9.982506 }; // γ_jack（v6d 全量调参）
const LN_MULT_CROSS = 1.88969303; // γ_cross（S_cross）
const LN_MULT_HOLD = 2.67786127; // γ_hold（S_hold）
const LN_MULT_ADJ = 0.28817355; // γ_adj（S_adj 头尾相邻性，v6d 全量调参）
// LN v18：衩（fork）——同手相邻三轨中，中间轨松手时左右两轨仍在保持/被按（手指独立性）。
// 仅 6K/7K（手内 ≥3 轨）有意义；4K 每手只有 2 轨，衩不被检出。
const LN_MULT_FORK = 1.0;
// 盾键（LN × RC 交集）：LN 保持期间同手相邻轨出现单点，需抬指/换指才能按到。
// γ 逐键标定（默认 0 = 关闭，保持 1.1.2 行为）；见 _analysis/_ln_shield_calib.py。
const LN_SHIELD_W = { 4: 0.0, 6: 0.0, 7: 0.0 };
const LN_SHIELD_TOL_MS = 20.0;
// LN 锚的常数项 b：u = ln(1 + LN_B·(k·n + b))，用于「线性 + 倍率」联合模型（默认 0）。
const LN_ADD_B = { 4: 0.0, 6: 0.0, 7: 0.0 };
// 盾（头侧）：同轨单点紧接 LN 头 → 头判时机难度。独立于密度的量，加在指数域：
//   u = ln(1 + LN_B·k·n + LN_B·(b + β_shield·S_shieldHead + β_rel·S_release))
// 放在指数域里，它的相对占比随密度自然淡出（“先占主头、后逐渐淡出”）。
const LN_SHIELD_HEAD_W = { 4: 0.0, 6: 0.0, 7: 0.0 };
const LN_SHIELD_HEAD_TAU = 90.0;
// 松手难度（带符号）：尾不与其它头/尾对齐 ↑、松开时同手邻近轨仍保持 ↑、
// 尾接近其它轨的头 ↓（避免与 S_jack 对反键型重复增益）。
const LN_RELEASE_W = { 4: 10.0, 6: 10.0, 7: 10.0 };
// 对齐其它起音时仍保留的基础难度（避免“对齐 = 无难度”，且保证 score 永不为负）
const LN_REL_FLOOR = 0.15;
const LN_REL_ALIGN_TAU = 150.0;
const LN_REL_HOLD_TAU = 120.0;
const LN_REL_HEAD_TAU = 100.0;
const LN_REL_HEAD_PEN = 1.0;
const LN_FORK_TOL_MS = 20.0; // 放手时刻与邻轨保持区间的重叠容差
// LN v18b：释放难度层 —— 长条在「与其它起音隔离」的时刻释放（玩家要在没有节拍提示的空隙里
// 控制放手时刻），对 6K/7K 记 S_rel；4K 保持原样。
const LN_REL_SOLO_MS = 120.0; // 释放点与最近其它起音的距离超过该值视为「孤立释放」
const LN_REL_LONG_MS = 400.0; // 只有长条（≥ 该长度）的释放才算「释放难度」
const LN_MULT_EXP = 5.07395755;
const LN_MULT_RHO_C = 0.29415144;
const LN_MULT_RHO_LIN = 0.0; // 低 ρ 段线性 toe 权重
const LN_ABSORB_GAIN = 0.25; // 松手吸收折扣 λ
const ABSORB_DIST_SCALE = 0.5; // 吸收判定距离 = 该倍数 × 本地节拍
const LN_LEN_WEIGHT_GAIN = 1.0; // ln_len_weight_gain：长度权重增益
const LN_HOLD_REF_MS = 40.0; // ln_hold_ref_ms：长度因子 ℓ/(ℓ+ℓ₀) 的参考时长

// ---- LN 模型 v5：松手节奏均匀度折扣（均匀 LN 排列几乎不产生锁手/松手需求）----
// 逐轨取全部松手间隔的变异系数 CV = sd/mean；CV → 0（节拍完全均匀，如 Malody 转谱的
// 等间距面缝流）时需求整体保留 LN_UNI_FLOOR；CV ≥ LN_UNI_CV0 视为完全不均匀（不打折）。
// 折扣同时作用于 S_rel / S_jack / S_cross；S_hold（长条本身）不折扣。
const LN_UNI_CV0 = 0.45; // 视为“完全不均匀”的 CV 阈值
const LN_UNI_FLOOR = 0.70; // 完全均匀排列的保留比例（不规整不再几乎无折扣：0.08→0.65）
const LN_UNI_POWER = 0.5; // 折扣曲线指数
const LN_UNI_T0 = 300.0;
// 独立「Chordstream 关注」项（逐键）：加性权重（乘 Chordstream 音符密度）与局部混入比例。
// 仅 4K 生效：6K/7K 上该特征无增量收益（替换或加入都会变差）。
const LN_CS_ADD_BY_KEY = { 4: 0.30, 6: 0.0, 7: 0.0 };
const LN_CS_MIX_BY_KEY = { 4: 0.50, 6: 0.0, 7: 0.0 }; // 不均性的相对时间尺度（ms）：CV = sd / (meanGap + T0)
const LN_REL_W_MIN = 0.25; // 极短长条的长度权重下限（保留弱检出，避免 <ℓ_lo 直接归零）

// ---- LN 模型 v6：S_jack 50ms 归零门控 + S_adj 头尾相邻性 ----
// S_jack 响应：单一解析函数（无分段/无截断）
//   S_jack = w·h · sigmoid((G50 − g)/SLOPE)
// g = 同轨下一次按下 − 该松手（单位 ms）。g=G50 处为 0.5 的「半衰」参考点；g 越小越接近 1，
// 50ms 附近保留冗余量（≈0.1，不再硬归零），窄面缝（≤25ms）仍明显更高；函数 C∞ 连续、处处可导。
const LN_JACK_G50_MS = 28.0;
const LN_JACK_SLOPE_MS = 10.0;
// 面缝规整度折扣：松手→同轨下次按下的间隔接近 LN_SEAM_GAP_MS 的事件越多，整谱 LN 乘子越低（连续）
const LN_SEAM_DISCOUNT = 0.80;

// 光滑 ReLU / 光滑截断：softRelu(x)=0.5*(x+sqrt(x^2+eps))，softClip01(x)=softRelu(x)-softRelu(x-1)。
// 与 clip(x,0,1) 偏差 < 1e-2，但处处可导，避免分段。
const LN_SOFT_EPS = 1e-4;
function softRelu(x) {
    return 0.5 * (x + Math.sqrt(x * x + LN_SOFT_EPS));
}
function softClip01(x) {
    return softRelu(x) - softRelu(x - 1.0);
}
const LN_SEAM_GAP_MS = 50.0; // 高斯中心
const LN_SEAM_TOL_MS = 25.0; // 过渡宽度

// beta_1.0.5 对数标定（按 103 张 LN 基准重拟合；LN v5 口径 RMSE 0.8452，v4 口径 0.8440）
// v5 相对 v4：S_rel 由 L2 改为算术平均、长度因子 h 并入 w、新增均匀度折扣 → 重新标定 A/B。
// 指数映射：u = ln(1 + LN_B·k·n)；随后经 LN 连续斜率映射层（lnMapForward）映射到等级。
// v6e（2026-10-28）：指数映射后接与 RC 同构的连续斜率映射层（C¹、无 clip），取代幂次映射。
const LN_B = 0.016473560;
const LN_A = 1.0; // 兼容旧参数快照的占位常量；实际映射由 lnMapForward 承担
const LN_C = 1.0; // 兼容旧参数快照的占位常量；实际映射由 lnMapForward 承担

// ---- beta_1.0.5 LN 模型 v4：在临界间隙处锚定 + 无上限的二次/三次重叠曲线 ----
// 间隙 g = 「该长条起点之后的下一次同轨按下」 − 释手（允许 g<0，长条重叠）。
// 临界间隙 δ* = T_loc − W_od（可为负）；归一化坐标 x = (g − δ*)/W_od。
// g ≥ δ*（x ≥ 0）：S_jack = w·h·S_a/(1 + x)            —— 在 δ* 处锚定 S_jack = w·h·S_a
// g < δ*          ：u = −x
//                   S_jack = w·h·S_a·(1 + u + b2·u² + b3·u³)
// 两段在 x=0 处 C¹ 连续；负间隙为二次/三次多项式 → 无上限、无竖直渐近线，缓慢抬升。
// S_a 为可调参数（锚定值）：在 102 张 LN 基准上扫描，S_a=8.0 时变换后 RMSE 最小（0.8248）。
const LN_JACK_ANCHOR = 8.0; // 锚定值 S_a：临界间隙处的 S_jack = w·h·S_a
const LN_JACK_OVERLAP_B2 = 0.5; // 重叠段二次项系数（u²）
const LN_JACK_OVERLAP_B3 = 0.0; // 重叠段三次项系数（u³；0 = 纯二次）
const LN_AGG_QUANTILE = 0.70; // 四项 S 的聚合分位（与 RC 的 findNQuantile 同机制）
const LN_TLOC_B = 1.0; // 局部密度下界：exp(mean(ln(contrib+b))) − b 的 b
const LN_TLOC_WINDOW = 16; // 局部窗口内的 RC contribution 个数（与 irr 窗口一致）
const LN_TLOC_MIN_MS = 10.0;
const LN_TLOC_MAX_MS = 50.0; // T_loc 上界：50ms 以下（与 OD 判定窗同量级）
// 0 = 沿用 RC 的 n（标定口径）；1 = LN 通道用自家分位 n（实验：102 张基准 RMSE 会退化到 ~1.8）
const LN_OWN_N_MODE = 0;

// LN 头/尾判定窗与长度有效性门（OD 相关）
const LN_HEAD_WINDOW_A = 64.0; // 头判窗总宽 = A − B·OD（ms）
const LN_HEAD_WINDOW_B = 3.0;
const LN_TAIL_WINDOW_RATIO = 1.5; // 尾判窗宽 / 头判窗宽
const LN_MIN_HOLD_MS = 20.0; // 一次点击的物理最短时长补足
const LN_VALID_OD_GAIN = 0.4; // 有效性门 OD 项系数
const LN_VALID_RAMP = 1.3; // 有效性门过渡宽度系数

function lowerBound(arr, target) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

function upperBound(arr, target) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// 合并两个升序分组时间序列并去重，同时输出与并集时间对齐的两侧物件数。
function mergeGroupSizes(timesA, sizesA, timesB, sizesB) {
    const times = [];
    const countsA = [];
    const countsB = [];
    let i = 0;
    let j = 0;
    while (i < timesA.length && j < timesB.length) {
        const ta = timesA[i];
        const tb = timesB[j];
        if (ta < tb) {
            times.push(ta); countsA.push(sizesA[i]); countsB.push(0); i += 1;
        } else if (ta > tb) {
            times.push(tb); countsA.push(0); countsB.push(sizesB[j]); j += 1;
        } else {
            times.push(ta); countsA.push(sizesA[i]); countsB.push(sizesB[j]); i += 1; j += 1;
        }
    }
    while (i < timesA.length) { times.push(timesA[i]); countsA.push(sizesA[i]); countsB.push(0); i += 1; }
    while (j < timesB.length) { times.push(timesB[j]); countsA.push(0); countsB.push(sizesB[j]); j += 1; }
    return { times, countsA, countsB };
}

// 单轨按相同时间戳分组 → [{ t, size, prevDt, nextDt }]，按时间升序。
function groupLane(times) {
    const groups = [];
    let i = 0;
    while (i < times.length) {
        let j = i;
        while (j + 1 < times.length && Math.abs(times[j + 1] - times[i]) < TIME_EPS) {
            j += 1;
        }
        groups.push({ t: times[i], size: j - i + 1 });
        i = j + 1;
    }
    const result = [];
    for (let idx = 0; idx < groups.length; idx += 1) {
        const prevDt = idx > 0 ? groups[idx].t - groups[idx - 1].t : null;
        const nextDt = idx + 1 < groups.length ? groups[idx + 1].t - groups[idx].t : null;
        result.push({ t: groups[idx].t, size: groups[idx].size, prevDt, nextDt });
    }
    return result;
}

// 重叠键检测：同一轨、同一时间戳（浮点精确相等）上的多个物件。
// 返回需要在 RC / LN 图上标红线的时刻列表（每线一个元素），以及总线条数。
//   - 纯单点叠放 → RC 图（n 个物件 → n-1 根线）
//   - 单点被 LN 覆盖 → RC 图（单点位置）
//   - LN 与 LN 叠放 → LN 图（位于下方的 LN 头位置，n 个 LN → n-1 根线）
function detectOverlaps(lanes) {
    const rc = [];
    const ln = [];
    const nearRc = [];
    const nearLn = [];
    const pos = { rc: [], ln: [], nearRc: [], nearLn: [] };
    const push = (arr, list, t, lane, isLn) => {
        arr.push(t);
        list.push({ t, lane, ln: isLn ? 1 : 0 });
    };
    for (let li = 0; li < lanes.length; li += 1) {
        const lane = lanes[li];
        // 精确叠放（同一时间戳）
        let i = 0;
        while (i < lane.length) {
            let j = i;
            while (j + 1 < lane.length && lane[j + 1].t === lane[i].t) {
                j += 1;
            }
            const count = j - i + 1;
            if (count > 1) {
                let lnCount = 0;
                for (let k = i; k <= j; k += 1) {
                    if (lane[k].len > 0) lnCount += 1;
                }
                const tapCount = count - lnCount;
                const rcLines = lnCount === 0 ? count - 1 : tapCount;
                for (let k = 0; k < rcLines; k += 1) push(rc, pos.rc, lane[i].t, li, false);
                for (let k = 0; k < lnCount - 1; k += 1) push(ln, pos.ln, lane[i].t, li, true);
            }
            i = j + 1;
        }
    }
    // 近似重叠（仅图表标注，不影响算法）：
    //   RC：同轨相邻间隔 < 5ms 且 < 周围间隔最小值的 1/40；
    //   LN：面缝（LN 尾 → 下一头）< 5ms 且 < 周围头/米间隔最小值的 1/80，标在面缝后的头处。
    for (let li = 0; li < lanes.length; li += 1) {
        const lane = lanes[li];
        const n = lane.length;
        if (n < 2) continue;
        const gap = new Float64Array(n - 1);
        for (let i = 1; i < n; i += 1) gap[i - 1] = Math.max(0.0, lane[i].t - lane[i - 1].t);
        // RC 侧
        for (let i = 1; i < n; i += 1) {
            const g = gap[i - 1];
            if (!(g > 0.0 && g < 5.0)) continue;
            const prevG = i >= 2 ? gap[i - 2] : 0.0;
            const nextG = i < gap.length ? gap[i] : 0.0;
            const around = Math.min(prevG, nextG);
            if (g < around / 40.0) push(nearRc, pos.nearRc, lane[i].t, li, false);
        }
        // LN 侧（面缝）
        for (let i = 0; i < n - 1; i += 1) {
            const cur = lane[i];
            if (!(cur.len > 0.0)) continue;
            const seam = lane[i + 1].t - (cur.t + cur.len);
            if (!(seam > 0.0 && seam < 5.0)) continue;
            const prevG = i >= 1 ? gap[i - 1] : 0.0;
            const nextG = i + 1 < gap.length ? gap[i + 1] : 0.0;
            const around = Math.min(prevG, nextG);
            if (seam < around / 80.0) push(nearLn, pos.nearLn, lane[i + 1].t, li, true);
        }
    }
    return {
        rc,
        ln,
        total: rc.length + ln.length,
        nearRc,
        nearLn,
        nearTotal: nearRc.length + nearLn.length,
        positions: pos,
    };
}

// 升序去重（容忍 TIME_EPS）。
function uniqueSorted(times) {
    const out = [];
    for (const t of times) {
        if (out.length === 0 || Math.abs(t - out[out.length - 1]) >= TIME_EPS) {
            out.push(t);
        }
    }
    return out;
}

// ---- RC v10 结构修正常数（连续核，全部无分段）----
const V11_EXTRA_SIGMA = 0.35;   // EXTRA 核宽度（相对局部间隔 a）
const V11_BOTH = 0.45;          // 双手同时活跃的协同加成（乘在 min(左,右) 上）
let RC11_KTOTAL = 4;
function v11Cfg(key, fallback) {
    const o = (typeof globalThis !== 'undefined') ? globalThis.__V11 : null;
    if (o && Number.isFinite(o[key])) return o[key];
    return fallback;
}

// 合并算子（soft_beta > 0）：softmax 加权平均（光滑的 max）。
//   w = ½(1+tanh(½β(a−b)))，M = w·a + (1−w)·b
function mergeOp(a, b) {
    const w = 0.5 * (1.0 + Math.tanh(HALF_BETA * (a - b)));
    return w * a + (1.0 - w) * b;
}

// 手内轨道布局：4K = 每手两轨；6K = 每手三轨（左手 0/1/2，右手 3/4/5）。
// 其它键数沿用 4K 的相邻配对规则（奇数键数时最后一手只有 1 轨）。
function handLayoutOf(kTotal) {
    // 任意键数通用：左右手 = 连续两半，左手 floor(K/2) 轨、右手其余。
    // 4K=2+2、5K=2+3、6K=3+3、7K=3+4、8K=4+4、9K=4+5、10K=5+5。
    const left = Math.floor(kTotal / 2);
    const hands = [[], []];
    for (let i = 0; i < kTotal; i += 1) {
        hands[i < left ? 0 : 1].push(i);
    }
    return hands.filter((h) => h.length > 0);
}

// 最大手宽（任意键数）：LN 的「衩/释放」层只在存在 >=3 轨的手时才有意义。
function lnHandMaxLen(lanes) {
    let m = 0;
    for (const h of handLayoutOf(lanes.length)) {
        if (h.length > m) m = h.length;
    }
    return m;
}

// 每手内的“配对轨”下标（RC 邻域额外 contribution 的 mate）。
// 4K 为唯一同手轨道；6K 中轨/外轨都偏向内侧轨（rank0→rank1、rank1→rank0、rank2→rank1）。
function handMateOf(hand, rank) {
    const n = hand.length;
    if (n < 2) {
        return hand[0];
    }
    // 通用配对规则：(0,1)(2,3)… 相邻成对；奇数轨手最后一轨与前一轨配对。
    // 2 轨 → [1,0]；3 轨 → [1,0,1]；4 轨 → [1,0,3,2]；5 轨 → [1,0,3,2,3]。
    if (n % 2 === 0 || rank < n - 1) {
        return hand[rank ^ 1];
    }
    return hand[n - 2];
}

// 每轨“同手其它轨”的下标列表（4K 长度 1；6K 长度 2；奇数键数落单轨长度 0）。
function handOthersOf(kTotal) {
    const out = [];
    for (const hand of handLayoutOf(kTotal)) {
        for (const lane of hand) {
            out[lane] = hand.filter((j) => j !== lane);
        }
    }
    return out;
}

// LN 配对轨序列：把同手其它轨的音符合并成一条按时间升序的列（4K 即原来的相邻轨）。
function lnPartnerCol(lanes, mateLanes) {
    if (!mateLanes || mateLanes.length === 0) {
        return null;
    }
    const partner = [];
    for (const j of mateLanes) {
        for (const n of lanes[j]) {
            partner.push(n);
        }
    }
    partner.sort((a, b) => a.t - b.t);
    return partner;
}
// 逐组注入贡献：base = 单件 M(prev_inv, next_inv)；多件按“超限替代”累加；
// extra = Σ_{配对轨键, |x|∈[0,4a]} size·exp(−((log_base(x/a)+extra_b)²)/2)，a = 1000/base；
// contribution = base·(1 + extra_rate·extra)
function laneContribs(groups, mateTimes, mateSizes) {
    const n = groups.length;
    const contribs = new Float64Array(n);
    if (n === 0) {
        return contribs;
    }
    for (let k = 0; k < n; k += 1) {
        const g = groups[k];
        const prevInv = (g.prevDt != null && g.prevDt > TIME_EPS) ? INV_DIST_SCALE / g.prevDt : 0.0;
        const nextInv = (g.nextDt != null && g.nextDt > TIME_EPS) ? INV_DIST_SCALE / g.nextDt : 0.0;
        let base = mergeOp(prevInv, nextInv);

        let extra = 0.0;
        if (EXTRA_RATE !== 0.0 && base > 0.0 && mateTimes.length > 0) {
            const a = INV_DIST_SCALE / base;
            const lo = lowerBound(mateTimes, g.t - 4.0 * a);
            const hi = upperBound(mateTimes, g.t + 4.0 * a);
            if (hi > lo) {
                const constTerm = EXTRA_B - Math.log(a) / LOG_BASE_DENOM;
                let total = 0.0;
                for (let m = lo; m < hi; m += 1) {
                    const x = Math.abs(mateTimes[m] - g.t);
                    if (x <= TIME_EPS) {
                        continue;
                    }
                    const z = Math.log(x / a) / LOG_BASE_DENOM + EXTRA_B;
                    total += mateSizes[m] * Math.exp(-(z * z) * 0.5);
                }
                extra = total;
            }
        }
        contribs[k] = base * (1.0 + EXTRA_RATE * extra);
    }
    return contribs;
}

// 单轨衰减序列在 unionTimes 各时刻的取值（下标对齐）：value ← value·0.5^(dt/H) + contribution。
function laneCurve(groupTimes, contribs, unionTimes) {
    const out = new Float64Array(unionTimes.length);
    let value = 0.0;
    let gi = 0;
    let lastT = null;
    for (let ti = 0; ti < unionTimes.length; ti += 1) {
        const t = unionTimes[ti];
        if (lastT !== null) {
            value *= Math.pow(0.5, (t - lastT) / HALF_LIFE_MS);
        }
        while (gi < groupTimes.length && Math.abs(groupTimes[gi] - t) < TIME_EPS) {
            value += contribs[gi];
            gi += 1;
        }
        out[ti] = value;
        lastT = t;
    }
    return out;
}

// 把“赋值处有突变的时间衰减函数”在给定时刻采样（替换语义）。
function sampleDecay(seq, times) {
    const out = new Float64Array(times.length);
    let value = 0.0;
    let si = 0;
    let lastT = null;
    for (let ti = 0; ti < times.length; ti += 1) {
        const t = times[ti];
        if (lastT !== null) {
            value *= Math.pow(0.5, (t - lastT) / HALF_LIFE_MS);
        }
        while (si < seq.length && Math.abs(seq[si][0] - t) < TIME_EPS) {
            value = seq[si][1];
            si += 1;
        }
        out[ti] = value;
        lastT = t;
    }
    return out;
}

// 合并两条“赋值突变衰减”序列：时间戳并集上采样后逐时刻 softmax 合并。
function mergeDecaySequences(seqA, seqB) {
    if (seqA.length === 0) {
        return seqB;
    }
    if (seqB.length === 0) {
        return seqA;
    }
    const lenA = seqA.length;
    const lenB = seqB.length;
    const times = new Float64Array(lenA + lenB);
    const cntA = new Int32Array(lenA + lenB);
    const cntB = new Int32Array(lenA + lenB);
    let i = 0;
    let j = 0;
    let m = 0;
    while (i < lenA && j < lenB) {
        const ta = seqA[i][0];
        const tb = seqB[j][0];
        if (ta < tb) { times[m] = ta; cntA[m] = seqA[i][2]; i += 1; }
        else if (ta > tb) { times[m] = tb; cntB[m] = seqB[j][2]; j += 1; }
        else { times[m] = ta; cntA[m] = seqA[i][2]; cntB[m] = seqB[j][2]; i += 1; j += 1; }
        m += 1;
    }
    while (i < lenA) { times[m] = seqA[i][0]; cntA[m] = seqA[i][2]; i += 1; m += 1; }
    while (j < lenB) { times[m] = seqB[j][0]; cntB[m] = seqB[j][2]; j += 1; m += 1; }

    const unionTimes = times.subarray(0, m);
    const fa = sampleDecay(seqA, unionTimes);
    const fb = sampleDecay(seqB, unionTimes);
    const out = new Array(m);
    for (let k = 0; k < m; k += 1) {
        const c = cntA[k] + cntB[k];
        if (c <= 0) {
            continue;
        }
        const both = Math.min(fa[k], fb[k]);
        const bothOn = RC11_KTOTAL === 4 || RC11_KTOTAL === 6 || RC11_KTOTAL === 7;
        const bothW = bothOn ? v11Cfg("both", V11_BOTH) : 0.0;
        let value = mergeOp(fa[k], fb[k]) + bothW * both;
        out[k] = [unionTimes[k], value, c];
    }
    return out;
}

// numpy 默认线性插值分位数。
function quantileLinear(sorted, q) {
    const n = sorted.length;
    if (n === 0) {
        return 0.0;
    }
    if (n === 1) {
        return sorted[0];
    }
    const pos = q * (n - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) {
        return sorted[lo];
    }
    const frac = pos - lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

// 加权均值（权重 <= 0 的哨兵物件不参与），用于把「平滑曲线均值」对齐到胶囊值。
function weightedMean(values, weights) {
    let sum = 0.0;
    let wsum = 0.0;
    for (let i = 0; i < values.length; i += 1) {
        const w = weights[i];
        if (Number.isFinite(values[i]) && Number.isFinite(w) && w > 0.0) {
            sum += values[i] * w;
            wsum += w;
        }
    }
    return wsum > 0.0 ? sum / wsum : Number.NaN;
}

// 纵轴上限提示：按未平滑的逐物件赋值取 p99.5 对应的 raw，再留 8% 余量，
// 避免平滑后峰值变矮导致锚线/刻度被挤出可视区域。
function sequenceYMaxHint(values, toRaw) {
    if (!values || values.length === 0 || typeof toRaw !== 'function') {
        return 0.0;
    }
    const sorted = Float64Array.from(values).sort();
    const q = quantileLinear(sorted, 0.995);
    const raw = toRaw(q);
    return Number.isFinite(raw) && raw > 0.0 ? raw * 1.08 : 0.0;
}

// 难度-进度曲线的平滑：在 [t−W/2, t+W/2) 窗口内对逐物件赋值做加权均值，
// 每 STEP 毫秒采样一次；窗口内没有合格物件时该采样点显示值直接为 0。
// 均值在「value 尺度」上计算，再经 toRaw 反算回原始值（y 轴仍是原始值）。
// windowMs/stepMs 可覆盖：RC 用默认 400ms，LN 用更大的 1000ms 窗口。
// toRaw 传 null 时只返回 times/value（用于先统计曲线分位、再决定锚点平移）。
function smoothAssignmentSequence(times, values, weights, tStart, tEnd, toRaw,
    windowMs = CHART_SMOOTH_WINDOW_MS, stepMs = CHART_SMOOTH_STEP_MS) {
    const n = times.length;
    const out = { times: [], value: [], raw: [], hasData: [] };
    if (n === 0) {
        return out;
    }
    const left = Math.max(tStart, times[0]);
    const right = Math.max(tEnd, times[n - 1]);
    const step = Math.max(stepMs, 1.0);
    const half = Math.max(windowMs, 0.0) * 0.5;
    const samples = Math.max(1, Math.floor((right - left) / step) + 1);
    const zeroRaw = toRaw ? toRaw(0.0) : 0.0;
    let lo = 0;
    let hi = 0;
    for (let s = 0; s < samples; s += 1) {
        const t = left + s * step;
        while (lo < n && times[lo] < t - half) {
            lo += 1;
        }
        while (hi < n && times[hi] <= t + half) {
            hi += 1;
        }
        let sum = 0.0;
        let wsum = 0.0;
        for (let i = lo; i < hi; i += 1) {
            const w = weights ? weights[i] : 1.0;
            if (!(w > 0.0)) {
                continue;
            }
            sum += values[i] * w;
            wsum += w;
        }
        const v = wsum > 0.0 ? sum / wsum : 0.0;
        out.times.push(t);
        out.value.push(v);
        out.hasData.push(wsum > 0.0);
        out.raw.push(toRaw ? (wsum > 0.0 ? toRaw(v) : zeroRaw) : 0.0);
    }
    return out;
}

// 求 n 使 mean(min(1, n/a_i)) == targetAvg（排序 + 1/a 前缀和 + 二分，分位数作上界初值）。
function findNQuantile(values, targetAvg) {
    const a = values instanceof Float64Array ? values : Float64Array.from(values);
    const n = a.length;
    if (n === 0) {
        throw new Error("赋值列表为空，无法计算 n");
    }
    if (targetAvg <= 0.0) {
        return 0.0;
    }
    a.sort();
    if (targetAvg >= 1.0) {
        return a[n - 1];
    }
    const prefixInv = new Float64Array(n + 1);
    for (let i = 0; i < n; i += 1) {
        prefixInv[i + 1] = prefixInv[i] + (a[i] > 0.0 ? 1.0 / a[i] : 0.0);
    }
    const totalInv = prefixInv[n];
    const targetSum = targetAvg * n;
    const partialSum = (value) => {
        const k = upperBound(a, value);
        return k + value * (totalInv - prefixInv[k]);
    };
    let lo = 0.0;
    let hi = quantileLinear(a, targetAvg);
    if (!(hi > 0.0) || partialSum(hi) < targetSum) {
        hi = a[n - 1];
    }
    // 二分求解。注意：hi/lo 收敛到 1 ULP 后 (lo+hi)*0.5 会取整回端点而不再推进，
    // 此时 hi-lo ≈ ULP 可能仍大于绝对容差 N_PRECISION（大 n 值时 ULP ≈ 3.6e-12），
    // 仅按绝对容差判停会死循环（超高物量谱面即触发）。故加“无进展即停”保护 + 迭代上限。
    for (let iter = 0; iter < 200; iter += 1) {
        if (hi - lo <= N_PRECISION) {
            break;
        }
        const mid = (lo + hi) * 0.5;
        if (mid === lo || mid === hi) {
            break;
        }
        if (partialSum(mid) < targetSum) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return (lo + hi) * 0.5;
}

function softplus(z) {
    if (z > 0.0) {
        return z + Math.log1p(Math.exp(-z));
    }
    return Math.log1p(Math.exp(z));
}

function sigmoid(z) {
    if (z > 0.0) {
        return 1.0 / (1.0 + Math.exp(-z));
    }
    const ez = Math.exp(z);
    return ez / (1.0 + ez);
}

// ---- 映射层 profile：4K 使用 beta_1.0.5 冻结曲线，6K 使用 b1.0.5.1 专用曲线 ----
// 结构相同：g(v) = c + Σ σ_j·w_j(v)，w 为 knots 处的 softplus 差分基，σ_j ≥ 0 时 g 单调递增。
// 缓冲随 profile 存放，避免每次调用新建数组；4K 的数值路径与此前逐位一致。
function makeMapProfile(knots, coef) {
    const K = knots.length;
    return {
        K,
        knots,
        coef,
        phi: new Float64Array(K),
        sig: new Float64Array(K),
        w: new Float64Array(K + 1),
        dw: new Float64Array(K + 1),
    };
}
const MAP_PROFILE_4K = makeMapProfile(MAP_KNOTS, MAP_COEF);
const MAP_PROFILE_6K = makeMapProfile(MAP6_KNOTS, MAP6_COEF);
const MAP_PROFILE_7K = makeMapProfile(MAP7_KNOTS, MAP7_COEF);
let MAP_ACTIVE = MAP_PROFILE_4K; // 每次 calculateFixedDensity 内按键数切换

// ---- 统一映射层（任意键数）----
// value = scale_K · S(a_K·g + b_K)，S 为全键共享的单调折线（z 空间的 12 个节点）。
// 每条键数只有 (a, b, scale) 三个标定常数——「算法与参数一致，只有难度标定不同」。
// 4/6/7K 由三分键 benchmark 联合拟合：RMSE 0.4314 / 0.3018 / 0.8009
// （对应旧三套独立映射 0.4365 / 0.3056 / 0.8053）。
// 5/8/9/10K：内容移植标定（同一谱面移植到目标键数后与 7K 数值对齐），粗标定。
const UMAP_KNOTS = [3.623611, 4.163574, 5.406860, 6.085749, 6.956064, 7.298338, 7.959245, 8.133186, 8.551638, 9.297130, 10.099199, 10.639488];
const UMAP_HEIGHTS = [-0.984899, -0.054499, 1.194019, 2.198958, 3.634215, 4.890067, 6.350227, 7.294950, 9.183649, 12.023020, 13.486152, 14.693033];
const UMAP_KEYS = {
    1: { a: 0.884056, b: 3.600612, scale: 1.030920 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    2: { a: 1.297584, b: 1.253190, scale: 1.349976 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    3: { a: 1.341735, b: 1.091961, scale: 1.331232 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    4: { a: 2.374768, b: -0.507805, scale: 1.111704 }, // 1.1.2 完全校准（连续行速率 + chordAlt 0.22）
    5: { a: 1.663421, b: 0.979979, scale: 1.078905 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    6: { a: 2.152658, b: -0.143555, scale: 1.619499 }, // 1.1.2 完全校准（连续行速率 + chordAlt 0.22）
    7: { a: 3.056075, b: -1.998326, scale: 0.824269 }, // 1.1.2 完全校准（连续行速率 + chordAlt 0.22）
    8: { a: 2.032005, b: 1.196105, scale: 1.144815 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    9: { a: 2.198881, b: 1.641738, scale: 0.878277 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
    10: { a: 2.353678, b: 1.413461, scale: 0.921618 }, // see README 0.5 (joint fit for 4/6/7; content-port fit otherwise)
};
let UMAP_ACTIVE = null;

// 未标定键数：在相邻已标定键数之间按 (a, b, scale) 线性插值（端点外沿用最近端点）。
function umapFor(kTotal) {
    const entry = UMAP_KEYS[kTotal];
    if (entry) {
        return entry;
    }
    const ks = Object.keys(UMAP_KEYS).map(Number).sort((x, y) => x - y);
    if (ks.length === 0) return null;
    if (kTotal <= ks[0]) return UMAP_KEYS[ks[0]];
    if (kTotal >= ks[ks.length - 1]) return UMAP_KEYS[ks[ks.length - 1]];
    let idx = 0;
    while (idx < ks.length - 1 && ks[idx + 1] <= kTotal) idx += 1;
    const k0 = ks[idx];
    const k1 = ks[idx + 1];
    const f = (kTotal - k0) / (k1 - k0);
    const p0 = UMAP_KEYS[k0];
    const p1 = UMAP_KEYS[k1];
    return {
        a: p0.a + f * (p1.a - p0.a),
        b: p0.b + f * (p1.b - p0.b),
        scale: p0.scale + f * (p1.scale - p0.scale),
    };
}

// 共享单调折线（线性外推）
function umapShape(z) {
    const xs = UMAP_KNOTS;
    const ys = UMAP_HEIGHTS;
    const n = xs.length;
    if (z <= xs[0]) {
        return ys[0] + (z - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0]);
    }
    if (z >= xs[n - 1]) {
        return ys[n - 1] + (z - xs[n - 1]) * (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]);
    }
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (xs[m] <= z) lo = m;
        else hi = m;
    }
    const f = (z - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + f * (ys[hi] - ys[lo]);
}

function umapShapeSlope(z) {
    const xs = UMAP_KNOTS;
    const ys = UMAP_HEIGHTS;
    const n = xs.length;
    if (z <= xs[0]) return (ys[1] - ys[0]) / (xs[1] - xs[0]);
    if (z >= xs[n - 1]) return (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]);
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (xs[m] <= z) lo = m;
        else hi = m;
    }
    return (ys[hi] - ys[lo]) / (xs[hi] - xs[lo]);
}

function umapShapeInv(y) {
    const xs = UMAP_KNOTS;
    const ys = UMAP_HEIGHTS;
    const n = xs.length;
    if (y <= ys[0]) return xs[0] + (y - ys[0]) * (xs[1] - xs[0]) / (ys[1] - ys[0]);
    if (y >= ys[n - 1]) return xs[n - 1] + (y - ys[n - 1]) * (xs[n - 1] - xs[n - 2]) / (ys[n - 1] - ys[n - 2]);
    let lo = 0;
    let hi = n - 1;
    while (hi - lo > 1) {
        const m = (lo + hi) >> 1;
        if (ys[m] <= y) lo = m;
        else hi = m;
    }
    const f = (y - ys[lo]) / (ys[hi] - ys[lo]);
    return xs[lo] + f * (xs[hi] - xs[lo]);
}

// mapInvert: g → value（映射值空间 → 最终难度值）
function umapInvert(g) {
    const p = UMAP_ACTIVE;
    return p.scale * umapShape(p.a * g + p.b);
}

// mapForward: value → g（反函数，供刻度 / 曲线还原使用）
function umapForward(v) {
    const p = UMAP_ACTIVE;
    return (umapShapeInv(v / p.scale) - p.b) / p.a;
}


// 一次遍历同时给出映射值与导数（g'(v)），供 Newton 求逆使用。
function mapForwardDeriv(v) {
    if (UMAP_ACTIVE) {
        const p = UMAP_ACTIVE;
        const z = p.a * v + p.b;
        return [p.scale * umapShape(z), p.scale * umapShapeSlope(z) * p.a];
    }
    const P = MAP_ACTIVE;
    const K = P.K;
    for (let i = 0; i < K; i += 1) {
        const v0 = P.knots[i][0];
        const s = P.knots[i][1];
        const z = (v - v0) / s;
        if (z > 0.0) {
            P.phi[i] = s * (z + Math.log1p(Math.exp(-z)));
            P.sig[i] = 1.0 / (1.0 + Math.exp(-z));
        } else {
            P.phi[i] = s * Math.log1p(Math.exp(z));
            const ez = Math.exp(z);
            P.sig[i] = ez / (1.0 + ez);
        }
    }
    P.w[0] = v - P.phi[0];
    P.dw[0] = 1.0 - P.sig[0];
    for (let j = 1; j < K; j += 1) {
        P.w[j] = P.phi[j - 1] - P.phi[j];
        P.dw[j] = P.sig[j - 1] - P.sig[j];
    }
    P.w[K] = P.phi[K - 1];
    P.dw[K] = P.sig[K - 1];
    let sum = P.coef[0];
    let deriv = 0.0;
    for (let i = 0; i < P.w.length; i += 1) {
        sum += P.coef[i + 1] * P.w[i];
        deriv += P.coef[i + 1] * P.dw[i];
    }
    return [sum, deriv];
}

function mapBasis(v) {
    const P = MAP_ACTIVE;
    const K = P.K;
    const phi = new Array(K);
    for (let i = 0; i < K; i += 1) {
        const [v0, s] = P.knots[i];
        phi[i] = s * softplus((v - v0) / s);
    }
    const w = new Array(K + 1);
    w[0] = v - phi[0];
    for (let j = 1; j < K; j += 1) {
        w[j] = phi[j - 1] - phi[j];
    }
    w[K] = phi[K - 1];
    return w;
}

function mapForwardBase(v) {
    if (UMAP_ACTIVE) {
        return umapForward(v);
    }
    const P = MAP_ACTIVE;
    const w = mapBasis(v);
    let sum = P.coef[0];
    for (let i = 0; i < w.length; i += 1) {
        sum += P.coef[i + 1] * w[i];
    }
    return sum;
}

function mapForward(v) {
    return mapForwardBase(v);
}

// 解 g(v) = target（g 单调 → Newton，失败时退回二分）。序列求逆时相邻 target 连续，
// 用上一次解热启动可把典型迭代次数压到 1–2 次。
let MAP_INVERT_LAST = 0.0;
function mapInvert(target) {
    if (UMAP_ACTIVE) {
        return umapInvert(target);
    }
    const scale = Math.max(1.0, Math.abs(target));
    const tol = 1e-13 * scale;
    let v = Number.isFinite(MAP_INVERT_LAST) ? MAP_INVERT_LAST : 0.0;
    let [g, dg] = mapForwardDeriv(v);
    for (let i = 0; i < 16; i += 1) {
        const diff = g - target;
        if (Math.abs(diff) <= tol) {
            MAP_INVERT_LAST = v;
            return v;
        }
        if (!(dg > 1e-9) || !Number.isFinite(dg)) {
            break;
        }
        let step = diff / dg;
        if (step > 10.0) {
            step = 10.0;
        } else if (step < -10.0) {
            step = -10.0;
        }
        v -= step;
        [g, dg] = mapForwardDeriv(v);
    }
    // 兜底：二分（慢但稳），保证与旧实现一致的精度。
    let lo = -60.0;
    let hi = 120.0;
    for (let i = 0; i < 60; i += 1) {
        const mid = (lo + hi) * 0.5;
        if (mapForwardBase(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const out = (lo + hi) * 0.5;
    MAP_INVERT_LAST = out;
    return out;
}

// ---- LN 连续斜率映射层（与 RC 的 mapForward/mapInvert 同构）----
// u = ln(1 + LN_B·k·n)；slope(u) = LN_MAP_S0 + Σ Δ_i·σ((u − u_i)/β)，
// value(u) = LN_MAP_C + LN_MAP_S0·u + Σ Δ_i·β·softplus((u − u_i)/β)。
// 全式连续可导（softplus/sigmoid），无分段裁剪；Δ 可正可负，靠斜率下限约束保证单调。
const LN_MAP_BETA = 0.362931;
const LN_MAP_S0 = 9.253036;
const LN_MAP_C = -0.592936;
const LN_MAP_KNOTS = [0.10, 0.25, 0.42, 0.60, 0.80, 1.02, 1.28, 1.60];
const LN_MAP_DELTA = [6.000000,6.000000,6.000000,6.000000,6.000000,6.000000,-5.999999,-6.000000];

// 复用缓冲：K = LN_MAP_KNOTS.length 固定。
const LN_MAP_PHI = new Float64Array(LN_MAP_KNOTS.length);
const LN_MAP_SIG = new Float64Array(LN_MAP_KNOTS.length);
const LN_MAP_W = new Float64Array(LN_MAP_KNOTS.length + 1);
const LN_MAP_DW = new Float64Array(LN_MAP_KNOTS.length + 1);

// 一次遍历同时给出映射值与导数（供 Newton 求逆）。基函数与 RC 的 mapForwardDeriv 同构：
// w[0] = u − φ_0；w[j] = φ_{j−1} − φ_j；w[K] = φ_{K−1}，其中 φ_i = β·softplus((u−u_i)/β)。
// value = C + S0·u + Σ_{j=1..K} Δ_{j−1}·β·w[j]
function lnMapForwardDeriv(u) {
    const K = LN_MAP_KNOTS.length;
    for (let i = 0; i < K; i += 1) {
        const z = (u - LN_MAP_KNOTS[i]) / LN_MAP_BETA;
        LN_MAP_PHI[i] = LN_MAP_BETA * softplus(z);
        LN_MAP_SIG[i] = sigmoid(z);
    }
    LN_MAP_W[0] = u - LN_MAP_PHI[0];
    LN_MAP_DW[0] = 1.0 - LN_MAP_SIG[0];
    for (let j = 1; j < K; j += 1) {
        LN_MAP_W[j] = LN_MAP_PHI[j - 1] - LN_MAP_PHI[j];
        LN_MAP_DW[j] = LN_MAP_SIG[j - 1] - LN_MAP_SIG[j];
    }
    LN_MAP_W[K] = LN_MAP_PHI[K - 1];
    LN_MAP_DW[K] = LN_MAP_SIG[K - 1];
    let v = LN_MAP_C + LN_MAP_S0 * u;
    let d = LN_MAP_S0;
    for (let j = 1; j <= K; j += 1) {
        const coef = LN_MAP_DELTA[j - 1] * LN_MAP_BETA;
        v += coef * LN_MAP_W[j];
        d += coef * LN_MAP_DW[j];
    }
    return [v, d];
}

function lnMapForward(u) {
    return lnMapForwardDeriv(u)[0];
}

let LN_MAP_INVERT_LAST = 0.8;
function lnMapInvert(target) {
    const scale = Math.max(1.0, Math.abs(target));
    const tol = 1e-13 * scale;
    let u = Number.isFinite(LN_MAP_INVERT_LAST) ? LN_MAP_INVERT_LAST : 0.8;
    let [g, dg] = lnMapForwardDeriv(u);
    for (let i = 0; i < 20; i += 1) {
        const diff = g - target;
        if (Math.abs(diff) <= tol) {
            LN_MAP_INVERT_LAST = u;
            return u;
        }
        if (!(dg > 1e-9) || !Number.isFinite(dg)) {
            break;
        }
        let step = diff / dg;
        if (step > 10.0) {
            step = 10.0;
        } else if (step < -10.0) {
            step = -10.0;
        }
        u -= step;
        [g, dg] = lnMapForwardDeriv(u);
    }
    // 兜底：二分（u ∈ [−1, 6] 覆盖 LN 0–40 对应的全部 k·n 范围）。
    let lo = -1.0;
    let hi = 6.0;
    for (let i = 0; i < 90; i += 1) {
        const mid = (lo + hi) * 0.5;
        if (lnMapForward(mid) < target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const out = (lo + hi) * 0.5;
    LN_MAP_INVERT_LAST = out;
    return out;
}

// raw（= 逐事件/整谱的 k·n 或 raw·n）→ LN 等级；raw ≤ 0 视为无 LN 需求。
function lnFromRawMapped(raw, nForLn, add) {
    if (!(nForLn > 0.0) || !(raw > 0.0)) {
        return 0.0;
    }
    const extra = Number.isFinite(add) ? add : 0.0;
    return lnMapForward(Math.log1p(LN_B * raw * nForLn + extra));
}

// LN 等级 → raw（图表刻度/锚点反算）。
function lnRawFromValue(value, nForLn, add) {
    if (!(nForLn > 0.0) || !Number.isFinite(value)) {
        return Number.NaN;
    }
    const extra = Number.isFinite(add) ? add : 0.0;
    const inner = Math.expm1(lnMapInvert(value)) - extra;
    return inner / (LN_B * nForLn);
}

// 滑窗 KDE 交叉熵：返回每窗熵与起始下标。
/**
 * RC v7 结构特征（与 _analysis/features_v13.js 完全一致，已验证 max|diff| = 0）。
 * 全部连续、无量纲、与总时长无关；只做线性扫描 + 两次排序（O(N log N)）。
 * 手 = 相邻两轨（4K：L=0/1，R=2/3；奇数键时最后一手为单轨）。
 */
const RC_V8_RATE_NORM = 8.0;
function rcV8Features(laneTimes, noteCount, onsetCount) {
    const K = laneTimes.length;
    const all = [];
    for (let k = 0; k < K; k += 1) {
        const lt = laneTimes[k];
        for (let i = 0; i < lt.length; i += 1) all.push(lt[i]);
    }
    all.sort((a, b) => a - b);
    const rows = [];
    for (let i = 0; i < all.length;) {
        let j = i + 1;
        while (j < all.length && Math.abs(all[j] - all[i]) < TIME_EPS) j += 1;
        rows.push([all[i], j - i]);
        i = j;
    }
    const M = rows.length;
    const t0 = M > 0 ? rows[0][0] : 0.0;
    const t1 = M > 0 ? rows[M - 1][0] : 0.0;
    const dsec = Math.max((t1 - t0) / 1000.0, 1e-3);

    const handIdx = [];
    for (let h = 0; h < K; h += 2) handIdx.push(h + 1 < K ? [h, h + 1] : [h]);
    const hands = [];
    for (let hi = 0; hi < handIdx.length; hi += 1) {
        const idx = handIdx[hi];
        const ev = [];
        for (let q = 0; q < idx.length; q += 1) {
            const lt = laneTimes[idx[q]];
            for (let i = 0; i < lt.length; i += 1) ev.push([lt[i], idx[q]]);
        }
        ev.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
        const hr = [];
        for (let i = 0; i < ev.length;) {
            let j = i + 1;
            while (j < ev.length && Math.abs(ev[j][0] - ev[i][0]) < TIME_EPS) j += 1;
            const lanesInRow = [];
            for (let q = i; q < j; q += 1) lanesInRow.push(ev[q][1]);
            lanesInRow.sort((a, b) => a - b);
            hr.push([ev[i][0], j - i, lanesInRow]);
            i = j;
        }
        hands.push(hr);
    }
    let nL = 0;
    let nR = 0;
    for (let q = 0; q < handIdx.length; q += 1) {
        let cnt = 0;
        for (let i = 0; i < handIdx[q].length; i += 1) cnt += laneTimes[handIdx[q][i]].length;
        if (q === 0) nL = cnt;
        else if (q === 1) nR = cnt;
    }

    const handSet = hands.map((hr) => {
        const s = new Set();
        for (let i = 0; i < hr.length; i += 1) s.add(hr[i][0]);
        return s;
    });
    let both = 0;
    let alt = 0;
    let prevAct = null;
    for (let i = 0; i < rows.length; i += 1) {
        const t = rows[i][0];
        const a0 = handSet.length > 0 ? handSet[0].has(t) : false;
        const a1 = handSet.length > 1 ? handSet[1].has(t) : false;
        const nAct = (a0 ? 1 : 0) + (a1 ? 1 : 0);
        if (nAct === 2) both += 1;
        if (prevAct !== null) {
            const pn = (prevAct[0] ? 1 : 0) + (prevAct[1] ? 1 : 0);
            if (pn === 1 && nAct === 1 && prevAct[0] !== a0) alt += 1;
        }
        prevAct = [a0, a1];
    }
    const fBoth = both / dsec / RC_V8_RATE_NORM;
    const fAlt = alt / dsec / RC_V8_RATE_NORM;

    const oscH = [];
    const laneAltH = [];
    const jackH = [];
    const dblH = [];
    for (let hi = 0; hi < hands.length; hi += 1) {
        const hr = hands[hi];
        if (hr.length === 0) {
            oscH.push(0.0); laneAltH.push(0.0); jackH.push(0.0); dblH.push(0.0);
            continue;
        }
        let cSum = 0;
        let diffSum = 0;
        let dblCnt = 0;
        let chg = 0;
        let same = 0;
        let prevSingle = null;
        for (let i = 0; i < hr.length; i += 1) {
            const c = hr[i][1];
            cSum += c;
            if (i > 0) diffSum += Math.abs(c - hr[i - 1][1]);
            if (c >= 2) dblCnt += 1;
            if (c === 1) {
                if (prevSingle !== null) {
                    const a = hr[i][2];
                    const b = prevSingle;
                    let isSame = a.length === b.length;
                    for (let q = 0; isSame && q < a.length; q += 1) if (a[q] !== b[q]) isSame = false;
                    if (isSame) same += 1; else chg += 1;
                }
                prevSingle = hr[i][2];
            }
        }
        const mc = cSum / hr.length;
        oscH.push(hr.length > 1 && mc > 0.0 ? diffSum / (hr.length - 1) / mc : 0.0);
        dblH.push(dblCnt / dsec / RC_V8_RATE_NORM);
        laneAltH.push(chg / dsec / RC_V8_RATE_NORM);
        jackH.push(same / dsec / RC_V8_RATE_NORM);
    }
    const meanOf = (a) => {
        if (a.length === 0) return 0.0;
        let s = 0.0;
        for (let i = 0; i < a.length; i += 1) s += a[i];
        return s / a.length;
    };
    const tot = nL + nR;
    return {
        f_both: fBoth,
        f_alt: fAlt,
        f_osc: meanOf(oscH),
        f_lanealt: meanOf(laneAltH),
        f_jack: meanOf(jackH),
        f_dbl: meanOf(dblH),
        f_bal: tot > 0.0 ? (2.0 * Math.min(nL, nR)) / tot : 0.0,
        f_chord: noteCount / Math.max(M, 1),
        f_notes: noteCount / dsec / RC_V8_RATE_NORM,
        f_handmax: Math.max(nL, nR) / dsec / RC_V8_RATE_NORM,
        f_bothrow: M > 0 ? both / M : 0.0,
    };
}

function windowEntropies(lg, window, stride) {
    const n = lg.length;
    if (n < window) {
        return { ents: new Float64Array(0), starts: new Int32Array(0) };
    }
    const startList = [];
    for (let s = 0; s + window <= n; s += stride) {
        startList.push(s);
    }
    const ents = new Float64Array(startList.length);
    const z = new Float64Array(window);
    for (let si = 0; si < startList.length; si += 1) {
        const s = startList[si];
        let lo = Infinity;
        let hi = -Infinity;
        for (let k = 0; k < window; k += 1) {
            const x = lg[s + k];
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        const rng = hi - lo;
        if (rng > 1e-12) {
            const inv = 1.0 / rng;
            for (let k = 0; k < window; k += 1) {
                z[k] = (lg[s + k] - lo) * inv;
            }
        } else {
            z.fill(0.0);
        }
        let sumLogQ = 0.0;
        for (let i = 0; i < window; i += 1) {
            const zi = z[i];
            let acc = 0.0;
            for (let j = 0; j < window; j += 1) {
                const d = zi - z[j];
                acc += Math.exp(IRR_ENT_COEF * (d * d));
            }
            sumLogQ += Math.log(acc / window);
        }
        ents[si] = IRR_LOG_DEN - sumLogQ / window;
    }
    return { ents, starts: Int32Array.from(startList) };
}

// 逐起音的局部邻域熵（窗口 16、步长 4，档间线性插值回逐键）。
function localIrregularity(onsets) {
    const size = onsets.length;
    const out = new Float64Array(size);
    if (size < 3) {
        return out;
    }
    const gaps = [];
    for (let i = 1; i < size; i += 1) {
        const d = onsets[i] - onsets[i - 1];
        if (d > TIME_EPS) {
            gaps.push(d);
        }
    }
    if (gaps.length < IRR_LOCAL_WINDOW) {
        return out;
    }
    const lg = Float64Array.from(gaps, (g) => Math.log2(g));
    const { ents, starts } = windowEntropies(lg, IRR_LOCAL_WINDOW, IRR_LOCAL_STRIDE);
    if (starts.length === 0) {
        return out;
    }
    const half = IRR_LOCAL_WINDOW >> 1;
    const centers = new Float64Array(starts.length);
    for (let i = 0; i < starts.length; i += 1) {
        centers[i] = starts[i] + half;
    }
    const last = centers.length - 1;
    for (let x = 0; x < size; x += 1) {
        if (x <= centers[0]) {
            out[x] = ents[0];
        } else if (x >= centers[last]) {
            out[x] = ents[last];
        } else {
            let lo = 0;
            let hi = last;
            while (hi - lo > 1) {
                const mid = (lo + hi) >> 1;
                if (centers[mid] <= x) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            const x0 = centers[lo];
            const x1 = centers[hi];
            const t = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
            out[x] = ents[lo] + (ents[hi] - ents[lo]) * t;
        }
    }
    return out;
}

// 在给定时刻上取逐键局部熵（精确命中，容忍 TIME_EPS；未命中取 0）。
function entropyAt(onsets, ent, times) {
    const out = new Float64Array(times.length);
    if (onsets.length === 0 || times.length === 0) {
        return out;
    }
    for (let i = 0; i < times.length; i += 1) {
        const t = times[i];
        const k0 = lowerBound(onsets, t);
        if (k0 < onsets.length && Math.abs(onsets[k0] - t) < TIME_EPS) {
            out[i] = ent[k0];
        }
    }
    return out;
}

// 沿起音序列做局部窗口平均（与 localIrregularity 同窗口/插值口径）。
function localWindowMean(values, window, stride) {
    const n = values.length;
    if (n === 0 || window <= 1) {
        return Float64Array.from(values);
    }
    window = Math.max(1, Math.floor(window));
    stride = Math.max(1, Math.floor(stride));
    if (n <= window) {
        const half = window >> 1;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i += 1) {
            const lo = Math.max(0, i - half);
            const hi = Math.min(n, i + half + 1);
            let sum = 0.0;
            for (let j = lo; j < hi; j += 1) {
                sum += values[j];
            }
            out[i] = sum / (hi - lo);
        }
        return out;
    }
    const starts = [];
    for (let s = 0; s + window <= n; s += stride) {
        starts.push(s);
    }
    if (starts.length === 0) {
        return Float64Array.from(values);
    }
    const means = new Float64Array(starts.length);
    for (let si = 0; si < starts.length; si += 1) {
        let sum = 0.0;
        const s = starts[si];
        for (let j = 0; j < window; j += 1) {
            sum += values[s + j];
        }
        means[si] = sum / window;
    }
    const centers = new Float64Array(starts.length);
    for (let i = 0; i < starts.length; i += 1) {
        centers[i] = starts[i] + (window >> 1);
    }
    const out = new Float64Array(n);
    const last = centers.length - 1;
    for (let x = 0; x < n; x += 1) {
        if (x <= centers[0]) {
            out[x] = means[0];
        } else if (x >= centers[last]) {
            out[x] = means[last];
        } else {
            let lo = 0;
            let hi = last;
            while (hi - lo > 1) {
                const mid = (lo + hi) >> 1;
                if (centers[mid] <= x) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            const x0 = centers[lo];
            const x1 = centers[hi];
            const t = x1 > x0 ? (x - x0) / (x1 - x0) : 0.0;
            out[x] = means[lo] + (means[hi] - means[lo]) * t;
        }
    }
    return out;
}

// 双手合并层：左右手同刻 chord/协调权重 s(t) ∈ [0,1]。
//   c_eff = max(L,R) + alpha·min(L,R)
//   c_bar = 局部窗口平均(c_eff)
//   s     = 1 − exp(−(c_bar−1)/scale)
//   s    *= (1 − f_bar)^density（f_bar = 局部窗口内 c_eff>1 的比例）
function crossHandWeight(lanes, onsets) {
    const n = onsets.length;
    const out = new Float64Array(n);
    if (n === 0) {
        return out;
    }
    const handCount = Math.max(1, Math.ceil(lanes.length / 2));
    const laneMaps = lanes.map((lane) => {
        const m = new Map();
        for (const note of lane) {
            m.set(note.t, (m.get(note.t) || 0) + 1);
        }
        return m;
    });
    const cEff = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
        const t = onsets[i];
        let total = 0;
        let maxCount = 0;
        for (let h = 0; h < handCount; h += 1) {
            let c = 0;
            const end = Math.min((h + 1) * 2, lanes.length);
            for (let l = h * 2; l < end; l += 1) {
                c += laneMaps[l].get(t) || 0;
            }
            total += c;
            if (c > maxCount) {
                maxCount = c;
            }
        }
        cEff[i] = maxCount + IRR_CROSS_ALPHA * (total - maxCount);
    }
    const cBar = localWindowMean(cEff, IRR_CROSS_WINDOW, IRR_LOCAL_STRIDE);
    const dense = IRR_CROSS_DENSITY > 0.0;
    const fBar = dense
        ? localWindowMean(Float64Array.from(cEff, (v) => (v > 1.0 ? 1.0 : 0.0)),
            IRR_CROSS_WINDOW, IRR_LOCAL_STRIDE)
        : null;
    for (let i = 0; i < n; i += 1) {
        let s = 1.0 - Math.exp(-(cBar[i] - 1.0) / IRR_CROSS_SCALE);
        if (dense) {
            s *= Math.pow(Math.max(0.0, Math.min(1.0, 1.0 - fBar[i])), IRR_CROSS_DENSITY);
        }
        out[i] = Math.max(0.0, Math.min(1.0, s));
    }
    return out;
}

// ---- LN 参考难度：OD 判定窗 / 长度有效性门 ----

// 该 OD 下 LN 头/尾判定的半宽 (W_head, W_tail)，单位 ms。
function lnOdWindows(od) {
    const wHead = Math.max((LN_HEAD_WINDOW_A - LN_HEAD_WINDOW_B * od) / 2.0, 0.0);
    return [wHead, wHead * LN_TAIL_WINDOW_RATIO];
}

// 长度有效性门的临界区间 (lo, hi)，单位 ms：ℓ ≤ lo 视为单键、ℓ ≥ hi 视为完整长条。
function lnValidBand(od) {
    const [wHead, wTail] = lnOdWindows(od);
    const lo = LN_MIN_HOLD_MS + LN_VALID_OD_GAIN * (wHead + wTail);
    return [lo, lo * LN_VALID_RAMP];
}

// 批量指数核和：times / queries 均升序，返回 K(q) = Σ exp(-|times[i] - q| / tau)。
// 只累加 |ti - q| ≤ 34·tau 的项（e^-34 < 2e-15，低于双精度有效位），每个查询点
// 从 hi 向两侧各扫一个固定半径窗口；原实现每个查询点都重扫全部 times（O(n·m)）。
// 不用「跨步乘衰减」的增量写法：间隔大时 1/decay 会放大舍入误差，实测会漂移到 1e-1 量级。
function kernelSumSeries(times, queries, tau) {
    const n = times.length;
    const m = queries.length;
    const out = new Float64Array(m);
    if (n === 0 || m === 0) {
        return out;
    }
    const invTau = 1.0 / tau;
    const reach = 34.0 * tau;
    let hi = 0;
    for (let q = 0; q < m; q += 1) {
        const t = queries[q];
        while (hi < n && times[hi] <= t) {
            hi += 1;
        }
        let sum = 0.0;
        for (let i = hi - 1; i >= 0; i -= 1) {
            const delta = t - times[i];
            if (delta > reach) {
                break;
            }
            sum += Math.exp(-delta * invTau);
        }
        for (let i = hi; i < n; i += 1) {
            const delta = times[i] - t;
            if (delta > reach) {
                break;
            }
            sum += Math.exp(-delta * invTau);
        }
        out[q] = sum;
    }
    return out;
}

// 单手两轨左右不均：signed kernel 差 → 软门控 → 时间衰减序列（返回 time → 累积权重 Map）。
function handImbalancePairSeq(leftTimes, rightTimes, allNoteTimes) {
    const leftCol = leftTimes || [];
    const rightCol = rightTimes || [];
    // 合并两个已升序的数组并去重（不用 Set/展开，减少分配）。
    const union = [];
    let leftIdx = 0;
    let rightIdx = 0;
    while (leftIdx < leftCol.length || rightIdx < rightCol.length) {
        const aTime = leftIdx < leftCol.length ? leftCol[leftIdx] : Infinity;
        const bTime = rightIdx < rightCol.length ? rightCol[rightIdx] : Infinity;
        if (aTime < bTime) {
            union.push(aTime);
            leftIdx += 1;
        } else if (bTime < aTime) {
            union.push(bTime);
            rightIdx += 1;
        } else {
            union.push(aTime);
            leftIdx += 1;
            rightIdx += 1;
        }
    }
    // 三个 kernel 和一次性批量算出（O(n+m)），避免每个查询点重扫。
    const leftK = kernelSumSeries(leftCol, union, HAND_IMB_TAU_MS);
    const rightK = kernelSumSeries(rightCol, union, HAND_IMB_TAU_MS);
    const allK = kernelSumSeries(allNoteTimes, union, HAND_IMB_TAU_MS);
    const seq = new Map();
    let s = 0.0;
    let prev = null;
    for (let u = 0; u < union.length; u += 1) {
        const t = union[u];
        const L = leftK[u];
        const R = rightK[u];
        const M = L + R;
        const D = L - R;
        const I = Math.sqrt(D * D + HAND_IMB_EPS * HAND_IMB_EPS) - HAND_IMB_EPS;
        const gate = 1.0 / (1.0 + Math.exp(-(M - 1.5) / 0.25));
        const G = allK[u];
        const dens = (M / (M + HAND_IMB_M0)) * (G / (G + HAND_IMB_NPS0));
        const raw = gate * I / (M + HAND_IMB_EPS) * dens;
        const weakMass = Math.min(L, R);
        const qWeak = weakMass / (weakMass + WEAK_GATE_Q0);
        const imb = I / (M + HAND_IMB_EPS);
        const weakGate = qWeak / (1.0 + Math.exp(-(imb - WEAK_GATE_IMB_B) / WEAK_GATE_IMB_S));
        if (prev !== null) {
            s *= Math.pow(0.5, (t - prev) / HAND_IMB_HALF_LIFE_MS);
        }
        s += raw * weakGate;
        seq.set(t, s);
        prev = t;
    }
    return seq;
}

// 单手不均权重：4K 每手两轨 = 单一配对（与原实现逐位一致）；
// 6K 每手三轨 = 12/23 两个重叠配对，单轨取包含它的各配对权重的算术平均。
function handImbalanceWeights(laneGroups, laneGroupTimes, allNoteTimes) {
    const weights = laneGroupTimes.map((tt) => new Float64Array(tt.length));
    for (const hand of handLayoutOf(laneGroups.length)) {
        if (hand.length < 2) {
            continue;
        }
        const pairSeqs = [];
        for (let r = 0; r + 1 < hand.length; r += 1) {
            pairSeqs.push(handImbalancePairSeq(
                laneGroupTimes[hand[r]] || [],
                laneGroupTimes[hand[r + 1]] || [],
                allNoteTimes,
            ));
        }
        for (let r = 0; r < hand.length; r += 1) {
            const lane = hand[r];
            const members = [];
            if (r - 1 >= 0) {
                members.push(pairSeqs[r - 1]);
            }
            if (r + 1 < hand.length) {
                members.push(pairSeqs[r]);
            }
            if (members.length === 0) {
                continue;
            }
            const tt = laneGroupTimes[lane];
            for (let k = 0; k < tt.length; k += 1) {
                let sum = 0.0;
                for (const seq of members) {
                    sum += seq.get(tt[k]) || 0.0;
                }
                weights[lane][k] = sum / members.length;
            }
        }
    }
    return weights;
}

function smoothstep(x, lo, hi) {
    if (hi - lo <= TIME_EPS) {
        return x >= hi ? 1.0 : 0.0;
    }
    let t = (x - lo) / (hi - lo);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return t * t * (3.0 - 2.0 * t);
}

function median(values) {
    const n = values.length;
    if (n === 0) {
        return 0.0;
    }
    const a = Float64Array.from(values);
    a.sort();
    const mid = n >> 1;
    return n % 2 === 1 ? a[mid] : (a[mid - 1] + a[mid]) * 0.5;
}

// 单根长条的长度权重 u = smoothstep(ℓ; lo, hi) ∈ [0,1]：过短视为单键、足够长视为完整长条。
function lnLengthWeight(lengthMs, od) {
    const [lo, hi] = lnValidBand(od);
    return smoothstep(lengthMs, lo, hi);
}

// 长度权重施加上增益：w = 1 − clip(gain)·(1 − u)；gain ≤ 0 时恒为 1。
function lnLenWeight(lengthMs, od) {
    if (LN_LEN_WEIGHT_GAIN <= 0.0) {
        return 1.0;
    }
    const gain = Math.min(Math.max(LN_LEN_WEIGHT_GAIN, 0.0), 1.0);
    return 1.0 - gain * (1.0 - lnLengthWeight(lengthMs, od));
}

// 全谱 LN 时长有效性 v = smoothstep(ℓ_med; lo, hi) ∈ [0,1]（无长条时为 0）：
// 长条短到判定上等同单键时整层结构量清零。
function lnHoldValidity(lanes, od) {
    const lens = [];
    for (const lane of lanes) {
        for (const n of lane) {
            if (n.len > 0) {
                lens.push(n.len);
            }
        }
    }
    if (lens.length === 0) {
        return 0.0;
    }
    const [lo, hi] = lnValidBand(od);
    return smoothstep(median(lens), lo, hi);
}

// ---- LN 参考难度：结构得分与 ln 倍率 ----
//
// 参与 k 的是 S_rel（同轨相邻松手间距倒数，按松手分组）、S_jack（松手→同轨下一次按下）、
// S_cross（保持期间配对轨超出本地基线的动作速率）与 S_hold（保持时长 / 本地物件平均间隔）。
// 逐灰长度权重 w/u 与长度因子 h 见 lnEventScores；整谱折扣 w(吸收) / v(有效性) 见 lnStructureScores。

function invGap(d) {
    return d > TIME_EPS ? INV_DIST_SCALE / d : 0.0;
}

// 局部密度下界 → 时间尺度 T_loc（ms）：窗口内 RC contribution 的几何均值
// （exp(mean(ln(c+b))) − b）作为“密度下界”，其倒数即局部间隔下界。
// 与 OD 判定窗相加得到 S_jack 的临界间隙，决定“大规模增益”何时开始。
// 在 LN 锚层插入「Chordstream 关注」的实验项（默认关闭，__V11 覆盖）。
//
// 1) 用与 RC 涂色条完全一致的规则把谱面切成 Jacks / Stream / Chordstream 区间；
// 2) 统计 Chordstream 区间的音符密度（notes/s）与占比；
// 3) lnAnchorMode:
//      0 = 现状（RC 分位 + occupancy + 局部音符密度）
//      1 = nRc 分位换成 Chordstream 密度
//      2 = 局部密度换成「Chordstream 音符的局部密度」
//      3 = 两者都换
// 旋钮：lnAnchorMode、lnAnchorMode{K}、lnCsW、lnCsWin。
function lnChordstreamMetrics(rawLanes) {
    const rows = [];
    for (const lane of rawLanes) {
        for (const n of lane) rows.push({ t: n.t, lane });
    }
    if (rows.length < 4) return { csDensity: 0.0, csShare: 0.0, csTimes: [], span: 0.0 };
    rows.sort((a, b) => a.t - b.t);
    // 同刻合并
    const merged = [];
    for (const r of rows) {
        const last = merged[merged.length - 1];
        if (last && Math.abs(last.t - r.t) < 1e-6) {
            if (!last.lanes.includes(r.lane)) last.lanes.push(r.lane);
        } else {
            merged.push({ t: r.t, lanes: [r.lane] });
        }
    }
    const MERGE_GAP = 15.0;
    const shareLane = (a, b) => a.some((x) => b.includes(x));
    // 同 rcintervals：间距 <15ms、无同轨、且小于上下间距最大值的 1/4 时合并（迭代取最优候选）
    for (;;) {
        const gaps = [];
        for (let i = 0; i + 1 < merged.length; i += 1) gaps.push(merged[i + 1].t - merged[i].t);
        let pick = -1;
        let pickRatio = Infinity;
        for (let i = 0; i + 1 < merged.length; i += 1) {
            const g = gaps[i];
            if (g >= MERGE_GAP) continue;
            if (shareLane(merged[i].lanes, merged[i + 1].lanes)) continue;
            const above = i > 0 ? gaps[i - 1] : Infinity;
            const below = i + 2 < merged.length ? gaps[i + 1] : Infinity;
            const ref = Math.max(Number.isFinite(above) ? above : -Infinity, Number.isFinite(below) ? below : -Infinity);
            const limit = Number.isFinite(ref) && ref > 0 ? ref * 0.25 : Infinity;
            if (!(g < limit)) continue;
            const ratio = g / (limit || 1);
            if (ratio < pickRatio) { pick = i; pickRatio = ratio; }
        }
        if (pick < 0) break;
        const a = merged[pick];
        const b = merged[pick + 1];
        const na = a.lanes.length;
        const nb = b.lanes.length;
        merged.splice(pick, 2, {
            t: (a.t * na + b.t * nb) / (na + nb),
            lanes: [...new Set([...a.lanes, ...b.lanes])].sort((x, y) => x - y),
        });
    }
    if (merged.length < 3) return { csDensity: 0.0, csShare: 0.0, csTimes: [], span: 0.0 };
    const segs = [];
    for (let i = 0; i + 1 < merged.length; i += 1) {
        const A = merged[i];
        const B = merged[i + 1];
        segs.push({ type: shareLane(A.lanes, B.lanes) ? "Jacks" : "Stream" });
    }
    for (let i = 0; i + 1 < segs.length; i += 1) {
        if (segs[i].type === "Jacks" || segs[i + 1].type === "Jacks") continue;
        if (!shareLane(merged[i].lanes, merged[i + 2].lanes)) continue;
        segs[i].type = "Chordstream";
        segs[i + 1].type = "Chordstream";
    }
    let csNotes = 0;
    let allNotes = 0;
    const csTimes = [];
    for (let i = 0; i < merged.length; i += 1) {
        const type = (segs[i] || segs[i - 1] || { type: "Stream" }).type;
        const n = merged[i].lanes.length;
        allNotes += n;
        if (type === "Chordstream") {
            csNotes += n;
            csTimes.push(merged[i].t);
        }
    }
    const span = Math.max(1e-6, (merged[merged.length - 1].t - merged[0].t) / 1000.0);
    return { csDensity: csNotes / span, csShare: allNotes > 0 ? csNotes / allNotes : 0.0, csTimes, span };
}

function lnLocalDensityTime(tauScaled, density) {
    if (!density || !density.times || density.times.length === 0) {
        return 90.0;
    }
    const times = density.times;
    const values = density.values;
    const n = times.length;
    const half = LN_TLOC_WINDOW >> 1;
    const idx = lowerBound(times, tauScaled);
    const lo = Math.min(Math.max(idx - half, 0), Math.max(n - LN_TLOC_WINDOW, 0));
    const hi = Math.min(n, lo + LN_TLOC_WINDOW);
    let sumLn = 0.0;
    let cnt = 0;
    for (let i = lo; i < hi; i += 1) {
        const v = values[i] + LN_TLOC_B;
        if (v > 0.0) {
            sumLn += Math.log(v);
            cnt += 1;
        }
    }
    if (cnt === 0) {
        return LN_TLOC_MAX_MS;
    }
    const gm = Math.exp(sumLn / cnt) - LN_TLOC_B;
    if (!(gm > 1e-9)) {
        return LN_TLOC_MAX_MS;
    }
    return Math.min(Math.max(INV_DIST_SCALE / gm, LN_TLOC_MIN_MS), LN_TLOC_MAX_MS);
}

// 与 RC 通道同款的分位聚合（findNQuantile）：把逐事件结构量收成一个“n 尺度”的标量，
// 而不是算术平均；分位目标见 LN_AGG_QUANTILE。
function lnAggregate(values) {
    if (!values || values.length === 0) {
        return 0.0;
    }
    return findNQuantile(Float64Array.from(values), v11Cfg("lnAggQ", LN_AGG_QUANTILE));
}

// 指定分位的聚合（目前只给 S_rel 用，便于单独扫分位）。
function lnAggregateAt(values, q) {
    if (!values || values.length === 0) {
        return 0.0;
    }
    return findNQuantile(Float64Array.from(values), q);
}

// 单轨 LN 事件的三项结构量；该轨无 LN 时返回 null。
// 逐根长度权重 w = lnLenWeight(ℓ, od)；S_rel / S_jack 再乘长度因子 h = ℓ/(ℓ+ℓ₀)
// （恢复“有效长条中越长越难”的单调性），S_hold 只乘 w。
function lnEventScores(col, partnerCol, od, invRate = 1.0, density = null, otherPresses = null) {
    const holds = col.filter((n) => n.len > 0);
    if (holds.length === 0) {
        return null;
    }
    const tt = uniqueSorted(holds.map((n) => n.t + n.len).sort((a, b) => a - b));
    const presses = col.map((n) => n.t).sort((a, b) => a - b);

    // 每个松手时刻的平均长度 → (长度权重 w, 长度因子 h)
    const lenAcc = new Map();
    for (const n of holds) {
        const key = n.t + n.len;
        const e = lenAcc.get(key);
        if (e) {
            e[0] += n.len;
            e[1] += 1;
        } else {
            lenAcc.set(key, [n.len, 1]);
        }
    }
    const w = new Float64Array(tt.length);
    const h = new Float64Array(tt.length);
    for (let i = 0; i < tt.length; i += 1) {
        const e = lenAcc.get(tt[i]);
        const m = e ? e[0] / e[1] : 0.0;
        w[i] = lnLenWeight(m, od);
        h[i] = LN_HOLD_REF_MS <= 0.0 ? 1.0 : m / (m + LN_HOLD_REF_MS);
    }

    // 松手节奏均匀度折扣（LN v5）：完全均匀的等间距排列 → 几乎无锁手/松手需求。
    const uni = new Float64Array(tt.length);
    uni.fill(1.0);
    if (tt.length >= 3) {
        let meanGap = 0.0;
        for (let i = 1; i < tt.length; i += 1) {
            meanGap += tt[i] - tt[i - 1];
        }
        meanGap /= (tt.length - 1);
        let varSum = 0.0;
        for (let i = 1; i < tt.length; i += 1) {
            const d = (tt[i] - tt[i - 1]) - meanGap;
            varSum += d * d;
        }
        const sd = Math.sqrt(varSum / (tt.length - 1));
        // 不均性口径：默认相对 CV；lnUniT0>0 时用“相对时间”口径 sd/(meanGap+T0)，
        // 让短间隔的绝对离散不被当成同等不规整。
        const uniCv0 = v11Cfg("lnUniCv0", LN_UNI_CV0);
        const uniFloor = v11Cfg("lnUniFloor", LN_UNI_FLOOR);
        const uniPower = v11Cfg("lnUniPower", LN_UNI_POWER);
        const uniT0 = v11Cfg("lnUniT0", LN_UNI_T0);
        const uniCap = v11Cfg("lnUniCap", 1.0);
        const cv = meanGap + uniT0 > TIME_EPS ? sd / (meanGap + uniT0) : 0.0;
        const t = Math.min(1.0, Math.max(0.0, cv / uniCv0));
        const uniVal = uniFloor + (1.0 - uniFloor) * Math.pow(t, uniPower);
        uni.fill(Math.min(uniCap, uniVal));
    }

    // S_rel：同轨相邻松手间距倒数的算术平均（v5：L2 → 均值，消除均匀流的 √2 双计）。
    // 实验模式（__V11.lnRelMode）：
    //   0 = 现状（尾–尾间隔 × 自身长度权重）
    //   1 = 面尾 → 同轨下一次按下（含短面头/单点）
    //   2 = 尾–尾间隔，但按 min(w_i, w_neighbor) 配对权重（长面不再被短面拖高）
    //   3 = 同轨 press/release 翻转率（每秒切换次数，滑窗）
    //   4 = 弃用 sRel：改为 sShort（短面自身）+ sRePress（尾→按下，饱和）
    const relMode = v11Cfg("lnRelMode", 6.0);
    const sRel = new Float64Array(tt.length);
    const sShort = new Float64Array(tt.length);
    const sRePress = new Float64Array(tt.length);
    const tailGapNext = new Float64Array(tt.length);
    tailGapNext.fill(-1.0);
    for (let i = 0; i < tt.length; i += 1) {
        let lo = 0;
        let hi = presses.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (presses[mid] <= tt[i]) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        tailGapNext[i] = lo < presses.length ? presses[lo] - tt[i] : -1.0;
    }
    const lenAvg = new Float64Array(tt.length);
    for (let i = 0; i < tt.length; i += 1) {
        const e = lenAcc.get(tt[i]);
        lenAvg[i] = e ? e[0] / e[1] : 0.0;
    }
    const sig = (x) => 1.0 / (1.0 + Math.exp(-x));
    if (relMode >= 3.5 && relMode < 4.5) {
        const shortL0 = v11Cfg("lnShortL0", 200.0);
        const shortTau = v11Cfg("lnShortTau", 60.0);
        const repD0 = v11Cfg("lnRepD0", 80.0);
        const repTau = v11Cfg("lnRepTau", 25.0);
        for (let i = 0; i < tt.length; i += 1) {
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sRel[i] = 0.0;
            sShort[i] = wEff * sig((shortL0 - lenAvg[i]) / shortTau);
            const d = tailGapNext[i];
            sRePress[i] = d >= 0.0 ? wEff * sig((repD0 - d) / repTau) : 0.0;
        }
    } else if (relMode >= 2.5 && relMode < 3.5) {
        // 翻转率：把该轨 press / release 合并成时间序列，统计尾附近每秒翻转次数。
        const evT = [];
        const evR = [];
        for (const p of presses) { evT.push(p); evR.push(1); }
        for (const t of tt) { evT.push(t); evR.push(0); }
        const order = evT.map((t, idx) => idx).sort((a, b) => evT[a] - evT[b]);
        const W = v11Cfg("lnFlipWin", 250.0);
        for (let i = 0; i < tt.length; i += 1) {
            let flips = 0;
            let lastType = -1;
            for (const idx of order) {
                const t = evT[idx];
                if (t < tt[i] - W) continue;
                if (t > tt[i] + W) break;
                const type = evR[idx];
                if (lastType >= 0 && type !== lastType) flips += 1;
                lastType = type;
            }
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sRel[i] = (flips / (2.0 * W / 1000.0)) * wEff;
        }
    } else if (relMode >= 1.5 && relMode < 2.5) {
        for (let i = 0; i < tt.length; i += 1) {
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            const wPrev = i > 0 ? Math.min(wEff, Math.max(w[i - 1], LN_REL_W_MIN)) : 0.0;
            const wNext = i + 1 < tt.length ? Math.min(wEff, Math.max(w[i + 1], LN_REL_W_MIN)) : 0.0;
            const pd = i > 0 ? tt[i] - tt[i - 1] : -1.0;
            const nd = i + 1 < tt.length ? tt[i + 1] - tt[i] : -1.0;
            sRel[i] = 0.5 * (invGap(pd) * wPrev + invGap(nd) * wNext) * uni[i];
        }
    } else if (relMode >= 6.5 || relMode >= 5.5) {
        // mode 6：锚点 = LN 头尾中点（单点仍用自身时间）；LN 前后的单点同样作为锚点参与取距。
        // 这样「长面 + 短面」与「长面 + 单点」的锚点间距接近，不再把长面的尾间距算得过近。
        const holdLenAt = new Map();
        for (const n of holds) {
            const prev = holdLenAt.get(n.t) || 0;
            if (n.len > prev) {
                holdLenAt.set(n.t, n.len);
            }
        }
        const anc = [];
        for (const p of presses) {
            const len = holdLenAt.get(p) || 0;
            anc.push({ t: p + len * 0.5, isLn: len > 0, len });
        }
        anc.sort((a, b) => a.t - b.t);
        // 锚点间差距的 CV → 均匀度折扣（等间距墙仍然打折）
        const uniAnchor = new Array(anc.length).fill(1.0);
        if (anc.length >= 3) {
            const gapsA = [];
            for (let i = 1; i < anc.length; i += 1) gapsA.push(anc[i].t - anc[i - 1].t);
            let mg = 0.0;
            for (const g of gapsA) mg += g;
            mg /= gapsA.length;
            let vv = 0.0;
            for (const g of gapsA) vv += (g - mg) * (g - mg);
            const uCv0 = v11Cfg("lnUniCv0", LN_UNI_CV0);
            const uFloor = v11Cfg("lnUniFloor", LN_UNI_FLOOR);
            const uPower = v11Cfg("lnUniPower", LN_UNI_POWER);
            const uT0 = v11Cfg("lnUniT0", LN_UNI_T0);
            const uCap = v11Cfg("lnUniCap", 1.0);
            const cvA = mg + uT0 > TIME_EPS ? Math.sqrt(vv / gapsA.length) / (mg + uT0) : 0.0;
            const tt2 = Math.min(1.0, Math.max(0.0, cvA / uCv0));
            const uu1 = Math.min(uCap, uFloor + (1.0 - uFloor) * Math.pow(tt2, uPower));
            // 周期 2 的规则性：交替间距也算“规律”（长+短/长+米 的锚点序列常是 2 周期）
            let uu2 = 1.0;
            if (gapsA.length >= 4) {
                const g2 = [];
                for (let j = 1; j + 2 < gapsA.length; j += 1) {
                    g2.push(Math.abs(gapsA[j] - gapsA[j + 2]) / Math.max(TIME_EPS, Math.max(gapsA[j], gapsA[j + 2])));
                }
                if (g2.length > 0) {
                    let mg2 = 0.0;
                    for (const x of g2) mg2 += x;
                    mg2 /= g2.length;
                    const tt3 = Math.min(1.0, Math.max(0.0, mg2 / uCv0));
                    uu2 = Math.min(uCap, uFloor + (1.0 - uFloor) * Math.pow(tt3, uPower));
                }
            }
            // 周期 2 规则性折扣的权重（lnUniPeriodW=1 现状；0 表示完全不看周期 2 一致性）
            const periodW = v11Cfg("lnUniPeriodW", 1.0);
            const uu = uu1 + (Math.min(uu1, uu2) - uu1) * periodW;
            uniAnchor.fill(uu);
        }
        const byT = new Map();
        for (let i = 0; i < anc.length; i += 1) {
            byT.set(anc[i].t, i);
        }
        for (let i = 0; i < tt.length; i += 1) {
            const a = tt[i] - lenAvg[i] * 0.5;
            // 找到该锚点下标（浮点用邻近比较）
            let idx = -1;
            if (byT.has(a)) {
                idx = byT.get(a);
            } else {
                let bestD = Infinity;
                for (let j = 0; j < anc.length; j += 1) {
                    const d = Math.abs(anc[j].t - a);
                    if (d < bestD) { bestD = d; idx = j; }
                }
            }
            const pd = idx > 0 ? anc[idx].t - anc[idx - 1].t : -1.0;
            const nd = idx + 1 < anc.length ? anc[idx + 1].t - anc[idx].t : -1.0;
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            const uniUse = relMode >= 6.5 ? uni[i] : uniAnchor[idx];
            const lnRelShortC = v11Cfg("lnRelShort", 0.0);
            const lnRelShortIrreg = v11Cfg("lnRelShortIrreg", 0.0);
            let shortGate = 1.0;
            if (lnRelShortC !== 0.0) {
                shortGate = 1.0 - lnRelShortC / (1.0 + Math.exp(-(200.0 - lenAvg[i]) / 60.0));
            }
            if (lnRelShortIrreg !== 0.0) {
                const sigS = 1.0 / (1.0 + Math.exp(-(200.0 - lenAvg[i]) / 60.0));
                shortGate *= 1.0 - lnRelShortIrreg * sigS * uniUse;
            }
            sRel[i] = 0.5 * (invGap(pd) + invGap(nd)) * wEff * uniUse * shortGate
                * Math.pow(h[i], v11Cfg("lnRelH", 0.0));
        }
    } else if (relMode >= 4.5) {
        // mode 5：短面不参与「尾–尾」间隔链（短面的难度由 sShort 承担），
        // 长面之间按长面自己的尾序列算间隔 → 长面+短面 与 长面+单点 分值接近。
        const longL0 = v11Cfg("lnRelSkipLen", 200.0);
        const idxLong = [];
        for (let i = 0; i < tt.length; i += 1) {
            if (lenAvg[i] >= longL0) {
                idxLong.push(i);
            }
        }
        for (let i = 0; i < tt.length; i += 1) {
            sRel[i] = -1.0;
        }
        for (let j = 0; j < idxLong.length; j += 1) {
            const i = idxLong[j];
            const pd = j > 0 ? tt[i] - tt[idxLong[j - 1]] : -1.0;
            const nd = j + 1 < idxLong.length ? tt[idxLong[j + 1]] - tt[i] : -1.0;
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sRel[i] = 0.5 * (invGap(pd) + invGap(nd)) * wEff * uni[i]
                * Math.pow(h[i], v11Cfg("lnRelH", 0.0));
        }
    } else if (relMode >= 0.5) {
        const relD0 = v11Cfg("lnRelD0", 30.0);
        for (let i = 0; i < tt.length; i += 1) {
            const d = tailGapNext[i];
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sRel[i] = d >= 0.0 ? (INV_DIST_SCALE / (d + relD0)) * wEff : 0.0;
        }
    } else {
        for (let i = 0; i < tt.length; i += 1) {
            const pd = i > 0 ? tt[i] - tt[i - 1] : -1.0;
            const nd = i + 1 < tt.length ? tt[i + 1] - tt[i] : -1.0;
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sRel[i] = 0.5 * (invGap(pd) + invGap(nd)) * wEff * uni[i]
                * Math.pow(h[i], v11Cfg("lnRelH", 0.0));
        }
    }
    if (relMode < 3.5 || relMode >= 4.5) {
        const shortL0 = v11Cfg("lnShortL0", 200.0);
        const shortTau = v11Cfg("lnShortTau", 60.0);
        const repD0 = v11Cfg("lnRepD0", 80.0);
        const repTau = v11Cfg("lnRepTau", 25.0);
        for (let i = 0; i < tt.length; i += 1) {
            const wEff = Math.max(w[i], LN_REL_W_MIN);
            sShort[i] = wEff * sig((shortL0 - lenAvg[i]) / shortTau);
            const d = tailGapNext[i];
            sRePress[i] = d >= 0.0 ? wEff * sig((repD0 - d) / repTau) : 0.0;
        }
    }

    // S_jack（v6）：50ms 面缝归零的连续门控（见 LN_JACK_ZERO_MS 注释）。
    const [jackWHead, jackWTail] = lnOdWindows(od);
    const wOd = jackWHead + jackWTail;
    const groupStarts = new Map();
    for (const n of holds) {
        const key = n.t + n.len;
        if (!groupStarts.has(key)) {
            groupStarts.set(key, []);
        }
        groupStarts.get(key).push(n.t);
    }
    const sJack = new Float64Array(tt.length);
    const gaps = new Float64Array(tt.length);
    gaps.fill(-1.0e9);
    for (let i = 0; i < tt.length; i += 1) {
        let gsum = 0.0;
        let gcnt = 0;
        for (const startT of groupStarts.get(tt[i])) {
            const k = upperBound(presses, startT);
            if (k < presses.length) {
                gsum += presses[k] - tt[i];
                gcnt += 1;
            }
        }
        if (gcnt === 0) {
            sJack[i] = 0.0;
            continue;
        }
        const g = gsum / gcnt;
        gaps[i] = g;
        const gate = 1.0 / (1.0 + Math.exp((g - LN_JACK_G50_MS) / LN_JACK_SLOPE_MS));
        sJack[i] = w[i] * h[i] * gate;
    }

    // S_adj（v6）：头尾相邻性。逐松手事件取「到其它轨最近按下」的距离 d，
    // adj = 1/(1+(2d/W_od)^4) × max(w, LN_REL_W_MIN)：松手与另一轨按下越贴（异步锁手/交接越多）越大；
    // 四轨同时按下/松开的整齐排列 d ≈ W_od → adj ≈ 0（整齐排列不额外加成）。
    const sAdj = new Float64Array(tt.length);
    if (otherPresses && otherPresses.length > 0) {
        const wDen = wOd > 1e-6 ? wOd : 1.0;
        for (let i = 0; i < tt.length; i += 1) {
            const te = tt[i];
            let lo = 0;
            let hi = otherPresses.length;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (otherPresses[mid] < te) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let d = Number.POSITIVE_INFINITY;
            if (lo < otherPresses.length) {
                d = otherPresses[lo] - te;
            }
            if (lo > 0) {
                d = Math.min(d, te - otherPresses[lo - 1]);
            }
            if (!Number.isFinite(d)) {
                sAdj[i] = 0.0;
                continue;
            }
            const q = (2.0 * d) / wDen;
            const q2 = q * q;
            const close = softClip01(1.0 - d / wDen);
            sAdj[i] = close * Math.max(w[i], LN_REL_W_MIN);
        }
    }

    // S_hold：保持时长 / 该轨物件平均间隔（无量纲，一根长条跨越多少个本地节拍）× w
    const lens = holds.map((n) => n.len);
    let baseGap = 0.0;
    if (presses.length > 1) {
        let sum = 0.0;
        for (let i = 1; i < presses.length; i += 1) {
            sum += presses[i] - presses[i - 1];
        }
        baseGap = sum / (presses.length - 1);
    }
    const sHold = (baseGap > TIME_EPS && lens.length > 0)
        ? Float64Array.from(lens, (l) => (l / baseGap) * lnLenWeight(l, od))
        : new Float64Array(lens.length);

    // 逐事件的平均 S_hold（每个松手事件一个长条赋值，供难度-进度图序列使用）
    const holdGroup = new Float64Array(tt.length);
    if (baseGap > TIME_EPS && holds.length > 0) {
        const endIdx = new Map();
        for (let i = 0; i < tt.length; i += 1) {
            endIdx.set(tt[i], i);
        }
        const holdCnt = new Float64Array(tt.length);
        for (let j = 0; j < holds.length; j += 1) {
            const idx = endIdx.get(holds[j].t + holds[j].len);
            if (idx !== undefined) {
                holdGroup[idx] += sHold[j];
                holdCnt[idx] += 1.0;
            }
        }
        for (let i = 0; i < tt.length; i += 1) {
            if (holdCnt[i] > 0.0) {
                holdGroup[i] /= holdCnt[i];
            }
        }
    }

    // S_cross：保持期间配对轨动作数超出本地基线期望的部分，换算成速率。
    const sCross = new Float64Array(tt.length);
    if (partnerCol && partnerCol.length > 0) {
        const pacts = [];
        for (const n of partnerCol) {
            pacts.push(n.t);
            if (n.len > 0) {
                pacts.push(n.t + n.len);
            }
        }
        pacts.sort((a, b) => a - b);
        const hs = holds.map((n) => n.t);
        const he = holds.map((n) => n.t + n.len);
        const lnW = holds.map((n) => lnLenWeight(n.len, od));
        const endIndex = new Map();
        for (let i = 0; i < tt.length; i += 1) {
            endIndex.set(tt[i], i);
        }
        const sums = new Float64Array(tt.length);
        const counts = new Float64Array(tt.length);
        for (let j = 0; j < holds.length; j += 1) {
            const kb = upperBound(pacts, hs[j]) - 1;
            const ka = lowerBound(pacts, he[j]);
            const nEv = Math.max(ka - kb - 1, 0);
            const hasB = kb >= 0;
            const hasA = ka < pacts.length;
            const gapB = hasB ? hs[j] - pacts[kb] : 0.0;
            const gapA = hasA ? pacts[ka] - he[j] : 0.0;
            const goodB = hasB && gapB > TIME_EPS;
            const goodA = hasA && gapA > TIME_EPS;
            let rate = 0.0;
            if (goodB && goodA) {
                rate = 2.0 / (gapB + gapA);
            } else if (goodB) {
                rate = 1.0 / gapB;
            } else if (goodA) {
                rate = 1.0 / gapA;
            }
            const length = he[j] - hs[j];
            if (rate > 0.0 && length > TIME_EPS) {
                const cross = (INV_DIST_SCALE * Math.max(nEv - rate * length, 0.0)
                    / length) * lnW[j];
                const idx = endIndex.get(he[j]);
                if (idx !== undefined) {
                    sums[idx] += cross;
                    counts[idx] += 1.0;
                }
            }
        }
        for (let i = 0; i < tt.length; i += 1) {
            sCross[i] = counts[i] > 0.0 ? sums[i] / counts[i] : 0.0;
        }
    }

    let tailGapMed = 0.0;
    let tailCv = 0.0;
    if (tt.length >= 3) {
        const gs = [];
        for (let i = 1; i < tt.length; i += 1) gs.push(tt[i] - tt[i - 1]);
        gs.sort((a, b) => a - b);
        tailGapMed = gs[gs.length >> 1];
        let mg = 0; for (const x of gs) mg += x; mg /= gs.length;
        let vv = 0; for (const x of gs) vv += (x - mg) * (x - mg);
        tailCv = mg > TIME_EPS ? Math.sqrt(vv / gs.length) / mg : 0.0;
    }
    return { sRel, sJack, sAdj, sCross, sHold, sShort, sRePress, times: tt, holdGroup, w, gaps, tailGapMed, tailCv };
}

// 全谱 LN 事件上的分位聚合（RC 同款 findNQuantile），并施以整谱折扣：
// S_rel / S_jack × w(松手吸收) × v(时长有效性)，S_cross / S_hold × v。
// 衩放手（fork release）逐事件分：对同手相邻三轨 (a, b, c)，b 轨每次松手时
// 查 a、c 是否仍在保持（LN 区间覆盖该时刻 ± tol）或被按下；两轨都有 → 1，单轨 → 0.5。
// 再乘事件长度权重 w（短条的放手压力更小），与 sAdj 的口径一致。
function lnForkScores(lanes, laneIndex, got, tol) {
    const nEv = got.times.length;
    const out = new Float64Array(nEv);
    const handOf = [];
    let start = 0;
    for (const hand of handLayoutOf(lanes.length)) {
        for (const lane of hand) {
            handOf[lane] = { hand, idx: hand.indexOf(lane) };
        }
        start += hand.length;
    }
    const info = handOf[laneIndex];
    if (!info || info.hand.length < 3) {
        return out;
    }
    const { hand, idx } = info;
    if (idx === 0 || idx === hand.length - 1) {
        return out;
    }
    const a = hand[idx - 1];
    const c = hand[idx + 1];

    const holdsA = [];
    const tapsA = [];
    for (const n of lanes[a] || []) {
        if (n.len > 0) holdsA.push([n.t, n.t + n.len]);
        else tapsA.push(n.t);
    }
    const holdsC = [];
    const tapsC = [];
    for (const n of lanes[c] || []) {
        if (n.len > 0) holdsC.push([n.t, n.t + n.len]);
        else tapsC.push(n.t);
    }
    const active = (holds, taps, t) => {
        for (let i = 0; i < holds.length; i += 1) {
            if (holds[i][0] - tol <= t && t <= holds[i][1] + tol) {
                return true;
            }
        }
        for (let i = 0; i < taps.length; i += 1) {
            if (Math.abs(taps[i] - t) <= tol) {
                return true;
            }
        }
        return false;
    };

    const wEff = new Float64Array(nEv);
    for (let j = 0; j < nEv; j += 1) {
        wEff[j] = Math.max(got.w[j], LN_REL_W_MIN);
        const t = got.times[j];
        const na = active(holdsA, tapsA, t) ? 1 : 0;
        const nc = active(holdsC, tapsC, t) ? 1 : 0;
        const f = na + nc;
        if (f > 0) {
            out[j] = (f === 2 ? 1.0 : 0.5) * wEff[j];
        }
    }
    return out;
}

function lnStructureScores(lanes, od, invRate = 1.0, density = null) {
    const rels = [];
    const jacks = [];
    const shorts = [];
    const repress = [];
    const jackEvents = [];
    const adjs = [];
    const crosses = [];
    const holdsArr = [];
    const seamSamples = [];
    const forks = [];
    const forkDenoms = [];
    const relSolo = [];
    const relLongGate = [];
    const events = [];
    const handOthers = handOthersOf(lanes.length);
    for (let i = 0; i < lanes.length; i += 1) {
        const partner = lnPartnerCol(lanes, handOthers[i]);
        let otherPresses = null;
        if (lanes.length > 1) {
            otherPresses = [];
            for (let j = 0; j < lanes.length; j += 1) {
                if (j === i) {
                    continue;
                }
                for (const n of lanes[j]) {
                    otherPresses.push(n.t);
                }
            }
            otherPresses.sort((a, b) => a - b);
        }
        const got = lnEventScores(lanes[i], partner, od, invRate, density, otherPresses);
        if (got) {
            const forkScores = lnForkScores(lanes, i, got, LN_FORK_TOL_MS);
            for (const v of forkScores) {
                forks.push(v);
            }
            for (let j = 0; j < got.times.length; j += 1) {
                forkDenoms.push(Math.max(got.w[j], LN_REL_W_MIN));
            }
            const endLen = new Map();
            for (const n of lanes[i]) {
                if (n.len > 0) {
                    const key = n.t + n.len;
                    const prev = endLen.get(key) || 0;
                    if (n.len > prev) endLen.set(key, n.len);
                }
            }
            for (let j = 0; j < got.times.length; j += 1) {
                const tRel = got.times[j];
                relSolo.push(tRel);
                relLongGate.push((endLen.get(tRel) || 0) >= LN_REL_LONG_MS ? 1.0 : 0.0);
            }
            for (const v of got.sRel) {
                if (v >= 0.0) {
                    rels.push(v);
                }
            }
            for (const v of got.sJack) {
                jacks.push(v);
                jackEvents.push(v);
            }
            for (const v of got.sShort) {
                shorts.push(v);
            }
            for (const v of got.sRePress) {
                repress.push(v);
            }
            for (const v of got.sAdj) {
                adjs.push(v);
            }
            for (const v of got.sCross) {
                crosses.push(v);
            }
            for (let j = 0; j < got.gaps.length; j += 1) {
                const gv = got.gaps[j];
                if (gv > -1.0e8) {
                    seamSamples.push(softClip01(1.0 - Math.abs(gv - LN_SEAM_GAP_MS) / LN_SEAM_TOL_MS));
                }
            }
            for (const v of got.sHold) {
                holdsArr.push(v);
            }
            for (let j = 0; j < got.times.length; j += 1) {
                events.push({
                    t: got.times[j],
                    rel: got.sRel[j],
                    fork: forkScores[j],
                    jack: got.sJack[j],
                    adj: got.sAdj[j],
                    cross: got.sCross[j],
                    hold: got.holdGroup[j],
                    w: got.w[j],
                });
            }
        }
    }
    // 盾（头侧）：每个 LN 头之前同轨单点越近 → 头判时机越难；长条权重 w，0.75 分位聚合。
    let sShieldHead = 0.0;
    {
        const vals = [];
        for (const col of lanes) {
            if (col.length === 0) continue;
            const taps = [];
            for (const n of col) if (!(n.len > 0.0)) taps.push(n.t);
            if (taps.length === 0) continue;
            taps.sort((a, b) => a - b);
            for (const h of col) {
                if (!(h.len > 0.0)) continue;
                // 头之前最近的单点（二分）
                let lo = 0, hi = taps.length;
                while (lo < hi) { const mid = (lo + hi) >> 1; if (taps[mid] <= h.t) lo = mid + 1; else hi = mid; }
                if (lo === 0) continue;
                const d = h.t - taps[lo - 1];
                const head = Math.exp(-d / LN_SHIELD_HEAD_TAU);
                if (head > 0.0) vals.push(head * Math.max(lnLenWeight(h.len, od), LN_REL_W_MIN));
            }
        }
        if (vals.length > 0) sShieldHead = lnAggregate(Float64Array.from(vals));
    }

    // 松手难度（带符号，连续、任意键数）：
    //   noAlign ：尾与其它轨的头/尾都不对齐 → 缺少节奏参照，↑
    //   adjHold ：松开时同手邻近轨仍未松 → 需要选择性松手，↑
    //   headPen ：尾接近其它轨的头（含反键/交错） → ↓，避免与 S_jack 重复增益
    // 优化：用排序数组 + 二分（最近邻）替代逐事件全表扫描；保持区间用时间扫掠维护 active 集。
    let sRelease = 0.0;
    {
        const vals = [];
        const knob = (d, tau) => Math.exp(-Math.max(0.0, Math.abs(d)) / tau);
        const nearestDist = (arr, x) => {
            let lo = 0, hi = arr.length;
            while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
            let best = Infinity;
            if (lo < arr.length) best = Math.min(best, Math.abs(arr[lo] - x));
            if (lo > 0) best = Math.min(best, Math.abs(arr[lo - 1] - x));
            return best;
        };
        const hands = handLayoutOf(lanes.length);
        for (const hand of hands) {
            for (const lane of hand) {
                const col = lanes[lane] || [];
                if (col.length === 0) continue;
                const oTimes = [];
                const oEnds = [];
                const lnHeads = [];
                const ivStart = [];
                const ivEnd = [];
                for (const l2 of hand) {
                    if (l2 === lane) continue;
                    for (const n of lanes[l2] || []) {
                        const end = n.len > 0.0 ? n.t + n.len : n.t;
                        oTimes.push(n.t);
                        oEnds.push(end);
                        if (n.len > 0.0) { lnHeads.push(n.t); ivStart.push(n.t); ivEnd.push(end); }
                    }
                }
                oTimes.sort((a, b) => a - b);
                oEnds.sort((a, b) => a - b);
                lnHeads.sort((a, b) => a - b);
                const order = ivStart.map((_, idx) => idx).sort((a, b) => ivStart[a] - ivStart[b]);
                const lnCount = ivStart.length;
                let ptr = 0;
                let activeEnds = [];
                for (const h of col) {
                    if (!(h.len > 0.0)) continue;
                    const te = h.t + h.len;
                    while (ptr < order.length && ivStart[order[ptr]] <= te) { activeEnds.push(ivEnd[order[ptr]]); ptr += 1; }
                    if (activeEnds.length) {
                        let w = 0;
                        for (const e of activeEnds) if (e >= te) activeEnds[w++] = e;
                        activeEnds.length = w;
                    }
                    let align = 0.0;
                    let headPen = 0.0;
                    if (oTimes.length) {
                        const dT = nearestDist(oTimes, te);
                        const dE = nearestDist(oEnds, te);
                        align = Math.max(knob(dT, LN_REL_ALIGN_TAU), knob(dE, LN_REL_ALIGN_TAU));
                    }
                    if (lnHeads.length) headPen = knob(nearestDist(lnHeads, te), LN_REL_HEAD_TAU);
                    let holdSum = 0.0;
                    for (const e of activeEnds) holdSum += 1.0 - knob(e - te, LN_REL_HOLD_TAU);
                    const adjHold = lnCount > 0 ? holdSum / lnCount : 0.0;
                    const noAlign = 1.0 - align;
                    const floorW = v11Cfg("lnRelFloor", LN_REL_FLOOR);
                    const raw = floorW + (1.0 - floorW) * noAlign + adjHold - LN_REL_HEAD_PEN * headPen;
                    vals.push(Math.max(0.0, raw) * Math.max(lnLenWeight(h.len, od), LN_REL_W_MIN));
                }
            }
        }
        if (vals.length > 0) sRelease = Math.max(0.0, lnAggregate(Float64Array.from(vals)));
    }
    // 释放难度层：对每个释放事件，取与「其它键起音」的最近距离；
    // 长条释放且距离 > LN_REL_SOLO_MS 视为孤立释放（占比）。
    let sRelLayer = 0.0;
    if (relSolo.length > 0 && lnHandMaxLen(lanes) >= 3) {
        const onsets = [];
        for (const lane of lanes) {
            for (const n of lane) {
                onsets.push(n.t);
            }
        }
        onsets.sort((a, b) => a - b);
        const holdsAll = [];
        for (const lane of lanes) {
            for (const n of lane) {
                if (n.len > 0) holdsAll.push(n);
            }
        }
        let acc = 0.0;
        let cnt = 0;
        for (let idx = 0; idx < relSolo.length; idx += 1) {
            const t = relSolo[idx];
            // 最近起音距离（忽略恰好等于 t 的本轨释放，取严格相邻的两个起音点）
            let lo = 0;
            let hi = onsets.length - 1;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (onsets[mid] < t) lo = mid + 1;
                else hi = mid;
            }
            let d = Infinity;
            for (let sh = -2; sh <= 2; sh += 1) {
                const kk = lo + sh;
                if (kk < 0 || kk >= onsets.length) continue;
                const dd = Math.abs(onsets[kk] - t);
                if (dd > 1e-9 && dd < d) d = dd;
            }
            const longGate = idx < relLongGate.length ? relLongGate[idx] : 0;
            if (Number.isFinite(d)) {
                acc += (d > LN_REL_SOLO_MS ? 1.0 : 0.0) * longGate;
                cnt += 1;
            }
        }
        sRelLayer = cnt > 0 ? acc / cnt : 0.0;
    }
    // 盾键（LN × RC 交集）：LN 保持期间同手相邻轨出现单点（严格位于保持区间内部），
    // 按长条长度权重 w 加权得到「带盾长条占比」。4K 相邻轨即配对轨；6K/7K 为手内相邻键位。
    let sShieldSum = 0.0;
    {
        const hands = handLayoutOf(lanes.length);
        let num = 0.0;
        let den = 0.0;
        for (const hand of hands) {
            if (hand.length < 2) continue;
            for (let hp = 0; hp < hand.length; hp += 1) {
                const laneHolds = (lanes[hand[hp]] || []).filter((n) => n.len > 0.0);
                if (laneHolds.length === 0) continue;
                const adjTaps = [];
                for (const dl of [-1, 1]) {
                    const np = hp + dl;
                    if (np < 0 || np >= hand.length) continue;
                    for (const n of lanes[hand[np]] || []) {
                        if (!(n.len > 0.0)) adjTaps.push(n.t);
                    }
                }
                if (adjTaps.length === 0) continue;
                for (const h of laneHolds) {
                    const wgt = Math.max(lnLenWeight(h.len, od), LN_REL_W_MIN);
                    den += wgt;
                    const inLo = h.t + LN_SHIELD_TOL_MS;
                    const inHi = h.t + h.len - LN_SHIELD_TOL_MS;
                    let hit = false;
                    for (const tapT of adjTaps) {
                        if (tapT > inLo && tapT < inHi) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit) num += wgt;
                }
            }
        }
        sShieldSum = den > 0.0 ? num / den : 0.0;
    }
    const absorb = lnReleaseAbsorb(lanes, v11Cfg("lnAbsorbGain", LN_ABSORB_GAIN));
    const lnValidForce = v11Cfg("lnValidForce", -1.0);
    const valid = lnValidForce >= 0.0 ? lnValidForce : lnHoldValidity(lanes, od);
    return {
        sRel: lnAggregateAt(rels, v11Cfg("lnRelQ", LN_AGG_QUANTILE)) * absorb * valid,
        sShort: lnAggregate(Float64Array.from(shorts)) * valid,
        sRePress: lnAggregate(Float64Array.from(repress)) * valid,
        sJack: lnAggregate(jacks) * absorb * valid,
        sJackEvents: jackEvents,
        sAdj: lnAggregate(adjs) * absorb * valid,
        sCross: lnAggregate(crosses) * valid,
        sHold: lnAggregate(holdsArr) * valid,
        // 衩（fork）松手结构分：逐事件 fork 权重（1/0.5 × w）的 0.75 分位。
        // 稀疏项用分位聚合与其它 S 项同口径；只有较大比例事件是衩时该项才明显抬升。
        sFork: lnAggregate(forks) * valid,
        sRelLayer: sRelLayer * valid,
        sShield: sShieldSum * valid,
        sShieldHead: sShieldHead * valid,
        sRelease: sRelease * valid,
        seamReg: lnAggregate(seamSamples) * valid,
        events,
        absorb,
        valid,
    };
}

// LN 通道的“自家分位 n”（实验）：逐松手事件取 γ 加权结构量，再用 findNQuantile 收成 n。
// LN_OWN_N_MODE = 1 时替代 RC n；会破坏现有 A/B 标定（102 张基准 RMSE ~1.8），需重新拟合。
function lnOwnQuantileN(lanes, od, invRate = 1.0, density = null) {
    const vals = [];
    const handOthers = handOthersOf(lanes.length);
    for (let i = 0; i < lanes.length; i += 1) {
        const partner = lnPartnerCol(lanes, handOthers[i]);
        const got = lnEventScores(lanes[i], partner, od, invRate, density);
        if (!got) {
            continue;
        }
        for (let j = 0; j < got.sRel.length; j += 1) {
            vals.push(LN_MULT_REL * got.sRel[j] + v11Cfg("lnJack", LN_MULT_JACK) * got.sJack[j]
                + LN_MULT_ADJ * got.sAdj[j] + LN_MULT_CROSS * got.sCross[j]);
        }
    }
    return vals.length === 0 ? 0.0 : findNQuantile(Float64Array.from(vals), LN_AGG_QUANTILE);
}

// 全谱松手吸收折扣 w = 1 − λ·ā ∈ [0,1]（λ ≤ 0 时恒为 1）。
// 单次松手吸收度 a = clip(1 − d/(0.5·g), 0, 1)：d 为松手时刻到最近“其它起音”的距离，
// g 为本地节拍（全谱正间隔中位数），ā 为全谱松手吸收度均值。
function lnReleaseAbsorb(lanes, gain) {
    if (gain <= 0.0) {
        return 1.0;
    }
    const onsetSet = new Set();
    for (const lane of lanes) {
        for (const n of lane) {
            onsetSet.add(n.t);
        }
    }
    const onsets = Array.from(onsetSet).sort((a, b) => a - b);
    if (onsets.length === 0) {
        return 1.0;
    }
    const diffs = [];
    for (let i = 1; i < onsets.length; i += 1) {
        const d = onsets[i] - onsets[i - 1];
        if (d > TIME_EPS) {
            diffs.push(d);
        }
    }
    if (diffs.length === 0) {
        return 1.0;
    }
    diffs.sort((a, b) => a - b);
    const mid = diffs.length >> 1;
    const gap = diffs.length % 2 === 1 ? diffs[mid] : (diffs[mid - 1] + diffs[mid]) * 0.5;
    if (gap <= TIME_EPS) {
        return 1.0;
    }
    const rels = [];
    for (const lane of lanes) {
        for (const n of lane) {
            if (n.len > 0) {
                rels.push(n.t + n.len);
            }
        }
    }
    if (rels.length === 0) {
        return 1.0;
    }
    const scale = ABSORB_DIST_SCALE * gap;
    let sum = 0.0;
    for (const r of rels) {
        const k0 = lowerBound(onsets, r);
        let d = Infinity;
        for (let sh = -1; sh <= 0; sh += 1) {
            const kk = Math.min(Math.max(k0 + sh, 0), onsets.length - 1);
            const dd = Math.abs(onsets[kk] - r);
            if (dd < d) {
                d = dd;
            }
        }
        sum += Math.min(Math.max(1.0 - d / scale, 0.0), 1.0);
    }
    const aBar = sum / rels.length;
    return 1.0 - Math.min(Math.max(gain, 0.0), 1.0) * aBar;
}

// ρ 饱和曲线 F(ρ) = lin·ρ + (1−lin)·(1+c^h)·ρ^h/(ρ^h+c^h)。
// 纯 Hill（lin=0）在 ρ≪c 段按 ρ^h 塌向 0；叠一条线性 toe（lin>0）后低 ρ 段近似线性，
// 少量长条的谱面不再被归零。仍满足 F(0)=0、F(1)=1、单调不减。
function lnRhoCurve(rho, h, c, lin) {
    if (rho <= 0.0) {
        return 0.0;
    }
    const k = Math.min(Math.max(lin, 0.0), 1.0);
    const rh = Math.pow(rho, h);
    const ch = Math.pow(c, h);
    const hill = (1.0 + ch) * rh / (rh + ch);
    return k * rho + (1.0 - k) * hill;
}

// 长条占比 ρ = Σ 长度权重 / 物件总数（无物件时为 0）。
function lnHoldRatio(lanes, od) {
    let total = 0;
    let holds = 0.0;
    for (const lane of lanes) {
        total += lane.length;
        for (const n of lane) {
            if (n.len > 0) {
                holds += lnLenWeight(n.len, od);
            }
        }
    }
    return total === 0 ? 0.0 : holds / total;
}

// ln 倍率 k = F(ρ)·(γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold)
// S_rel / S_jack 已含整谱折扣；rate > 1（DT）时这两项与物件时间同倍缩放 → 乘 rate。
// density 提供 RC 逐物件 contribution（时间/值），用于 S_jack 的局部密度下界 T_loc。
function lnMultiplier(lanes, od, rate, density = null) {
    const rho = lnHoldRatio(lanes, od);
    if (rho <= 0.0) {
        return { k: 0.0, events: [], factor: 0.0, absorb: 0.0, valid: 0.0 };
    }
    const invRate = 1.0 / rate;
    const structure = lnStructureScores(lanes, od, invRate, density);
    const { sRel, sJack, sAdj, sCross, sHold, sFork, events, absorb, valid } = structure;
    const gammaFork = v11Cfg("lnFork", LN_MULT_FORK);
    const gammaJack = v11Cfg(`lnJack${lanes.length}`, v11Cfg("lnJack", LN_MULT_JACK_BY_KEY[lanes.length] ?? LN_MULT_JACK));
    const gammaRel = v11Cfg(`lnRel${lanes.length}`, v11Cfg("lnRel", LN_MULT_REL_BY_KEY[lanes.length] ?? LN_MULT_REL));
    const gammaHold = v11Cfg("lnHold", LN_MULT_HOLD);
    const gammaAdj = v11Cfg("lnAdj", LN_MULT_ADJ);
    const gammaShort = v11Cfg("lnShort", 0.0);
    const gammaRepress = v11Cfg("lnRepress", 0.0);
    const gammaRelLayer = v11Cfg("relLayer", 3.0) * (lnHandMaxLen(lanes) >= 3 ? 1.0 : 0.0);
    const gammaShield = v11Cfg("lnShield", LN_SHIELD_W[lanes.length] || 0.0);
    const total = gammaRel * (sRel * rate)
        + gammaJack * (sJack * rate)
        + gammaAdj * (sAdj * rate)
        + LN_MULT_CROSS * sCross
        + gammaHold * sHold
        + gammaFork * sFork
        + gammaRelLayer * (structure.sRelLayer || 0.0)
        + gammaShield * (structure.sShield || 0.0)
        + gammaShort * (structure.sShort || 0.0)
        + gammaRepress * (structure.sRePress || 0.0);
    const factor = lnRhoCurve(rho, LN_MULT_EXP, LN_MULT_RHO_C, LN_MULT_RHO_LIN);
    const seamPenalty = 1.0 - LN_SEAM_DISCOUNT * structure.seamReg;
    if (typeof globalThis !== 'undefined' && globalThis.__RC_DEBUG_LN && typeof globalThis.__RC_DEBUG_LN === 'object') {
        globalThis.__RC_DEBUG_LN.sFork = sFork;
        globalThis.__RC_DEBUG_LN.sRelLayer = structure.sRelLayer || 0.0;
        globalThis.__RC_DEBUG_LN.forkTotal = total;
    }
    const totalSeam = total * seamPenalty;
    return {
        k: totalSeam < 0.0 ? 0.0 : factor * totalSeam,
        events,
        factor,
        absorb,
        valid,
        parts: {
            rho, factor, sRel, sJack, sAdj, sCross, sHold, sFork,
            sRelLayer: structure.sRelLayer || 0.0, sShield: structure.sShield || 0.0,
            sJackEvents: structure.sJackEvents || [],
            sShieldHead: structure.sShieldHead || 0.0, sRelease: structure.sRelease || 0.0,
            sShort: structure.sShort || 0.0, sRePress: structure.sRePress || 0.0,
            seamPenalty, gammaFork, gammaRelLayer, gammaShield, gammaJack, gammaRel, gammaHold, gammaAdj,
        },
    };
}

function collectLanes(parsed) {
    const keyCount = Number(parsed?.columnCount) || 0;
    const columns = Array.isArray(parsed?.columns) ? parsed.columns : [];
    const starts = Array.isArray(parsed?.noteStarts) ? parsed.noteStarts : [];
    const ends = Array.isArray(parsed?.noteEnds) ? parsed.noteEnds : [];
    const types = Array.isArray(parsed?.noteTypes) ? parsed.noteTypes : [];
    const lanes = Array.from({ length: keyCount }, () => []);
    for (let i = 0; i < columns.length; i += 1) {
        const key = Number(columns[i]);
        const t = Number(starts[i]);
        if (!Number.isFinite(key) || key < 0 || key >= keyCount || !Number.isFinite(t)) {
            continue;
        }
        let len = 0;
        if ((Number(types[i]) & TYPE_HOLD) !== 0) {
            const end = Number(ends[i]);
            if (Number.isFinite(end) && end > t) {
                len = end - t;
            }
        }
        lanes[key].push({ t, len });
    }
    for (const lane of lanes) {
        lane.sort((a, b) => a.t - b.t);
    }
    return lanes;
}

function resolveParsed(source, cvtFlag = null) {
    const flag = String(cvtFlag || "").toUpperCase();
    const wantIN = flag.includes("IN");
    const wantHO = flag.includes("HO");
    const canConvert = wantIN || wantHO;

    let parsed = null;
    let osuText = null;
    if (source && typeof source === "object" && typeof source.getParsedData === "function") {
        if (typeof source.osuText === "string") {
            osuText = source.osuText;
        }
        parsed = source.getParsedData();
    } else if (typeof source === "string") {
        osuText = source;
    } else {
        throw new Error("Unsupported Fixed Density source.");
    }

    // lazer IN/HO：在独立 parser 上应用转换后再取解析数据，避免污染调用方共享 parser。
    // 转换失败（异常 / 0 物件）时回退未转换结果：图表与胶囊不应因 cvt 失败而整块变 NaN。
    if (canConvert) {
        if (typeof osuText !== "string") {
            console.warn("[FD] cvtFlag requires .osu text source; using unconverted parse.");
        } else {
            try {
                const converter = new OsuFileParser(osuText);
                converter.process();
                if (wantIN) converter.modIN();
                if (wantHO) converter.modHO();
                if (converter.columns.length > 0) {
                    return converter.getParsedData();
                }
                console.warn(`[FD] cvtFlag=${flag} produced 0 objects; falling back to unconverted parse.`);
            } catch (error) {
                console.warn(`[FD] cvtFlag=${flag} conversion failed: ${error?.message || error}; falling back.`);
            }
        }
    }

    if (parsed) {
        return parsed;
    }
    const parser = new OsuFileParser(osuText);
    parser.process();
    return parser.getParsedData();
}

function normalizeRate(rate) {
    const value = Number(rate);
    if (!Number.isFinite(value) || value <= 0) {
        return 1.0;
    }
    return value;
}

/**
 * 计算 Fixed Density v1 的 rc_value 与 ln_value。
 *
 * @param {string|object} source .osu 文本 或 已 process() 的 OsuFileParser 实例
 * @param {number} [rate=1] 变速倍率（DT=1.5、HT=0.75、lazer 任意正速）；时间戳按 1/rate 缩放
 * @param {object} [options] 可选；`cvtFlag`（"IN"/"HO"，lazer Invert / Hold Off）会在
 *   独立 parser 上应用对应转换后再计算，语义与 lazer 一致。
 * @returns {{ rcValue: number, lnValue: number }} 无有效物件时两者均为 NaN
 */

// ---- RC v10：局部物理特征乘子（仅 kTotal===4，逐事件连续因子）----
// 7 个维度全部来自谱面自身（行内押数、双手同押行、换手、同轨纵连、手内不平衡、
// 行轨集重复、间隔不规则），沿时间 500ms 高斯核平滑后取 m = exp(clip(INTERCEPT + Σβz)).
// 不使用任何键型标签；6K/7K 不启用本层，保持 b1.0.5.1 行为。
const RC10_SMOOTH_MS = 500.0;
const RC10_JACK_MS = 250.0;
const RC10_EXP_CLIP = 2.0;
const RC10_INTERCEPT = -0.3170;
const RC10_BETA = [0.0000, 0.0000, 0.0000, -0.0375, 0.0000, 0.0000, 0.0000];
// 逐键 β（可用环境变量覆盖：MM_RC10_BETA_6K='[...]'；浏览器环境无 process，自动跳过）
const RC10_BETA_BY_KEY = (() => {
    const out = {};
    if (typeof process !== "undefined" && process.env) {
        for (const kk of [4, 5, 6, 7, 8, 9, 10]) {
            const raw = process.env[`MM_RC10_BETA_${kk}K`];
            if (!raw) continue;
            try {
                const v = JSON.parse(raw);
                if (Array.isArray(v) && v.length === 7) out[kk] = v;
            } catch (e) { /* ignore */ }
        }
    }
    globalThis.__RC10_BETA_BY_KEY = out;
    return out;
})();
const RC10_FEAT_MEAN = [0.398, 0.281, 0.732, 0.833, 0.753, 0.142, 0.163];
const RC10_FEAT_SCALE = [0.144, 0.260, 0.216, 0.245, 0.224, 0.226, 0.210];

const RC10_BETA_6K_CAL = [0.000, 0.064, 0.000, 0.032, 0.064, -0.120, 0.000];
const RC10_GATE_6K = [0.30, 0.65];
const RC10_BETA_7K_CAL = [0.000, 0.064, 0.000, 0.032, 0.064, -0.120, 0.000];
const RC10_GATE_7K = [0.30, 0.65];
const RC_END_BETA = (() => {
    const out = { 4: 0.0, 6: 0.0, 7: 0.0 };
    if (typeof process !== "undefined" && process.env && process.env.MM_RC_END_BETA) {
        try {
            const v = JSON.parse(process.env.MM_RC_END_BETA);
            for (const k of Object.keys(v)) out[k] = Number(v[k]) || 0.0;
        } catch (e) { /* ignore */ }
    }
    return out;
})(); // 耐力层逐键系数（0 = 关闭）
const RC_END_GBETA = (() => {
    const out = { 4: 0.35, 6: 0.3, 7: 0.5 };
    if (typeof process !== "undefined" && process.env && process.env.MM_RC_END_GBETA) {
        try { const v = JSON.parse(process.env.MM_RC_END_GBETA); for (const k of Object.keys(v)) out[k] = Number(v[k]) || 0.0; } catch (e) { /* ignore */ }
    }
    return out;
})(); // 谱面级耐力项：gPred += beta·ln(1+E_q75)
const RC_END_SIGMA = (() => {
    const env = (typeof process !== "undefined" && process.env && process.env.MM_RC_END_SIGMA) ? Number(process.env.MM_RC_END_SIGMA) : NaN;
    return Number.isFinite(env) && env > 0 ? env : 1.0;
})(); // 对数难度差 σ（相对变化 ~35% 视为不同难度段）
const RC_END_RESET_K = 3.0;    // 难度突变时的指数衰减系数（越大越"骤降"）
const RC_END_TAU_MS = 600.0;   // 断档后的衰减时间常数
const RC_END_GAP_MS = 200.0;   // 正常行间隔（≤ 该值不额外衰减）
const RC_END_DT_CAP_MS = 600.0;// 单行最多计入的时长
const RC_END_WIN_MS = 2000.0; // 轨内 nps 统计窗口
const RC_END_EMA_MS = (() => {
    const env = (typeof process !== "undefined" && process.env && process.env.MM_RC_END_EMA) ? Number(process.env.MM_RC_END_EMA) : NaN;
    return Number.isFinite(env) && env > 0 ? env : 3000.0;
})(); // 难度基线 EMA 时间常数（消除行间抖动，只对真实段落变化重置）
const RC_END_T0_MS = (() => {
    const env = (typeof process !== "undefined" && process.env && process.env.MM_RC_END_T0) ? Number(process.env.MM_RC_END_T0) : NaN;
    return Number.isFinite(env) && env > 0 ? env : 30000.0;
})(); // 累积时间尺度（默认 30s：E 表示"连续相近难度的等效秒数/30s"）
let RC10_LAST_K = 0;
function rc10LocalFeatures(lanes, kTotal, mergedTimes) {
    const nOut = mergedTimes.length;
    RC10_LAST_K = kTotal;
    if (!lanes || lanes.length === 0 || nOut === 0) return null;
    if (kTotal < 4) return null;
    const onsets = [];
    for (let c = 0; c < kTotal; c += 1) {
        const lane = lanes[c];
        if (!lane) continue;
        for (let i = 0; i < lane.length; i += 1) onsets.push([lane[i].t, c]);
    }
    if (onsets.length < 2) return null;
    onsets.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const rowTimes = [];
    const rowCols = [];
    let curT = onsets[0][0];
    let curCols = [onsets[0][1]];
    for (let i = 1; i < onsets.length; i += 1) {
        const t = onsets[i][0];
        const c = onsets[i][1];
        if (Math.abs(t - curT) <= 1e-6) curCols.push(c);
        else { rowTimes.push(curT); rowCols.push(curCols); curT = t; curCols = [c]; }
    }
    rowTimes.push(curT);
    rowCols.push(curCols);
    const nRows = rowTimes.length;
    if (nRows < 2) return null;

    const leftCount = handLayoutOf(kTotal)[0].length;
    const lastLaneT = new Float64Array(kTotal).fill(-1e9);
    const raw = [];
    let prevMask = 0;
    let prevSet = null;
    for (let i = 0; i < nRows; i += 1) {
        const t = rowTimes[i];
        const cols = rowCols[i];
        const set = new Set(cols);
        let mask = 0;
        let left = 0;
        let right = 0;
        let jack = 0;
        for (const c of cols) {
            if (c < leftCount) { left += 1; mask |= 1; } else { right += 1; mask |= 2; }
            if (t - lastLaneT[c] <= RC10_JACK_MS) jack += 1;
        }
        for (const c of cols) lastLaneT[c] = t;
        const total = left + right;
        let inter = 0;
        if (prevSet) for (const c of set) if (prevSet.has(c)) inter += 1;
        const denom = Math.sqrt(total * (prevSet ? prevSet.size : 1));
        const f = new Array(7);
        f[0] = total / kTotal;
        f[1] = (mask === 3) ? 1.0 : 0.0;
        f[2] = (i > 0 && mask !== prevMask) ? 1.0 : 0.0;
        f[3] = total > 0 ? jack / total : 0.0;
        f[4] = total > 0 ? Math.abs(left - right) / total : 0.0;
        f[5] = (prevSet && denom > 0) ? inter / denom : 0.0;
        raw.push(f);
        prevMask = mask;
        prevSet = set;
    }
    const diffs = [];
    for (let i = 1; i < nRows; i += 1) diffs.push(rowTimes[i] - rowTimes[i - 1]);
    const sorted = diffs.slice().sort((a, b) => a - b);
    const med = sorted.length > 0 ? Math.max(sorted[Math.floor(sorted.length / 2)], 1e-3) : 1.0;
    for (let i = 0; i < nRows; i += 1) {
        if (i === 0) { raw[i][6] = 0.0; continue; }
        const d = Math.max(rowTimes[i] - rowTimes[i - 1], 1e-3);
        raw[i][6] = Math.min(1.0, Math.abs(Math.log(d / med)) / Math.log(4.0));
    }

    const sigma = v11Cfg("smooth", RC10_SMOOTH_MS);
    const reach = 3.0 * sigma;
    const smooth = [];
    for (let i = 0; i < nRows; i += 1) {
        const ti = rowTimes[i];
        const acc = new Array(7).fill(0.0);
        let wsum = 0.0;
        for (let j = i; j >= 0 && ti - rowTimes[j] <= reach; j -= 1) {
            const dt = ti - rowTimes[j];
            const w = Math.exp(-0.5 * (dt / sigma) * (dt / sigma));
            const fj = raw[j];
            for (let q = 0; q < 7; q += 1) acc[q] += w * fj[q];
            wsum += w;
        }
        for (let j = i + 1; j < nRows && rowTimes[j] - ti <= reach; j += 1) {
            const dt = rowTimes[j] - ti;
            const w = Math.exp(-0.5 * (dt / sigma) * (dt / sigma));
            const fj = raw[j];
            for (let q = 0; q < 7; q += 1) acc[q] += w * fj[q];
            wsum += w;
        }
        if (wsum > 0) for (let q = 0; q < 7; q += 1) acc[q] /= wsum;
        smooth.push(acc);
    }

    const out = [];
    for (let q = 0; q < 7; q += 1) out.push(new Float64Array(nOut));
    let ptr = 0;
    for (let k = 0; k < nOut; k += 1) {
        const t = mergedTimes[k];
        while (ptr + 1 < nRows && Math.abs(rowTimes[ptr + 1] - t) <= Math.abs(rowTimes[ptr] - t)) ptr += 1;
        const row = smooth[ptr];
        for (let q = 0; q < 7; q += 1) out[q][k] = row[q];
    }
    return out;
}

function rc10LocalMultiplier(feat) {
    if (!feat) return null;
    const n = feat[0].length;
    const m = new Float64Array(n);
    for (let k = 0; k < n; k += 1) {
        let z = v11Cfg("intercept", RC10_INTERCEPT);
        for (let q = 0; q < 7; q += 1) {
            const kNow = RC10_LAST_K;
            const betaArr = RC10_BETA_BY_KEY[kNow] || (kNow === 6 ? RC10_BETA_6K_CAL : (kNow === 7 ? RC10_BETA_7K_CAL : RC10_BETA));
            const betaQ = v11Cfg(`rc10Beta${kNow}_${q}`, v11Cfg(`rc10Beta${q}`, betaArr[q]));
            if (betaQ !== 0.0) {
                const gateDef = (kNow === 6) ? RC10_GATE_6K : (kNow === 7 ? RC10_GATE_7K : [-1.0, -1.0]);
                const gLo = v11Cfg("rc10GateLo", gateDef[0]);
                const gHi = v11Cfg("rc10GateHi", gateDef[1]);
                let gate = 1.0;
                const gateFeats = v11Cfg("rc10GateFeats", ((kNow === 6 || kNow === 7) ? "0,5" : ""));
                if (gLo >= 0.0 && String(gateFeats).split(",").includes(String(q))) {
                    const ov = feat[5][k];
                    const tt = Math.max(0.0, Math.min(1.0, (ov - gLo) / Math.max(1e-6, gHi - gLo)));
                    gate = tt * tt * (3.0 - 2.0 * tt);
                }
                const scale = Math.max(RC10_FEAT_SCALE[q], 1e-6);
                z += gate * betaQ * (feat[q][k] - RC10_FEAT_MEAN[q]) / scale;
            }
        }
        const zc = Math.max(-RC10_EXP_CLIP, Math.min(RC10_EXP_CLIP, z));
        m[k] = Math.exp(zc);
    }
    return m;
}

// ---- RC v13：高密度轮换折叠（roll -> 对拍）----
// 物理背景：单键轮换（1234 循环）的行速率超过人手单点极限后，实际演奏会被
// 成对合并成同手双押（1234 的 (0,1)/(2,3) 对 = 对拍；1324 的 (0,2)/(1,3) 对 = 对切），
// 因此其难度应向「同音符数的对拍」收敛。本层只吃三个物理量：
//   1) 局部单键行速率（nps，40ms 级间隔 → 40+ nps 才饱和）；
//   2) 四轨轮换度（窗口内不同轨数，4 轨才满）；
//   3) 两种相位下的同手相邻对占比取大者（1234/1243/2134 = 1，1324 = 0，用于把对拍型和对切型分开）。
// 满足条件的行按 exp(-delta * score) 逐事件打折，等价于「这些键会被并排按下」。
// 不引用任何键型标签；仅 4K 启用，6K/7K 保持 b1.0.6 行为。
const RC_ROT_LO_NPS = 20.0;
const RC_ROT_HI_NPS = 40.0;
const RC_ROT_DELTA = 0.50;
const RC_ROT_REACH_MS = 180.0;

function rcRotSmoothstep(t) {
    const x = Math.max(0.0, Math.min(1.0, t));
    return x * x * (3.0 - 2.0 * x);
}

function rcRotDiscount(lanes, kTotal, mergedTimes) {
    if (kTotal < 4 || !lanes || lanes.length === 0 || mergedTimes.length === 0) return null;
    const onsets = [];
    for (let c = 0; c < kTotal; c += 1) {
        const lane = lanes[c];
        if (!lane) continue;
        for (let i = 0; i < lane.length; i += 1) onsets.push([lane[i].t, c]);
    }
    if (onsets.length < 8) return null;
    onsets.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const rowTimes = [];
    const rowCols = [];
    let curT = onsets[0][0];
    let curCols = [onsets[0][1]];
    for (let i = 1; i < onsets.length; i += 1) {
        const t = onsets[i][0];
        const c = onsets[i][1];
        if (Math.abs(t - curT) <= 1e-6) curCols.push(c);
        else { rowTimes.push(curT); rowCols.push(curCols); curT = t; curCols = [c]; }
    }
    rowTimes.push(curT);
    rowCols.push(curCols);
    const singleIdx = [];
    for (let i = 0; i < rowTimes.length; i += 1) {
        if (rowCols[i].length === 1) singleIdx.push(i);
    }
    const nS = singleIdx.length;
    if (nS < 5) return null;
    const leftCount = handLayoutOf(kTotal)[0].length;
    const handOf = (c) => (c < leftCount ? 0 : 1);
    const loNps = v11Cfg("rotLo", RC_ROT_LO_NPS);
    const hiNps = v11Cfg("rotHi", RC_ROT_HI_NPS);
    const delta = v11Cfg("rotDelta", RC_ROT_DELTA);
    const reach = v11Cfg("rotReach", RC_ROT_REACH_MS);
    const raw = new Float64Array(nS);
    const WIN = 8;
    const wLen = Math.min(WIN, nS);
    for (let k = 0; k < nS; k += 1) {
        const w0 = Math.max(0, Math.min(nS - wLen, k - (wLen >> 1)));
        const ts = [];
        const ls = [];
        for (let q = 0; q < wLen; q += 1) {
            const ri = singleIdx[w0 + q];
            ts.push(rowTimes[ri]);
            ls.push(rowCols[ri][0]);
        }
        const span = ts[wLen - 1] - ts[0];
        if (!(span > 1e-6)) continue;
        const rate = (wLen - 1) / (span / 1000.0);
        const distinct = new Set(ls).size;
        // 成对相位：同一手相邻两轨的「对拍」配对占比（取两种相位中较大者）。
        let same0 = 0;
        let c0 = 0;
        for (let q = 0; q + 1 < wLen; q += 2) {
            c0 += 1;
            if (handOf(ls[q]) === handOf(ls[q + 1]) && Math.abs(ls[q] - ls[q + 1]) === 1) same0 += 1;
        }
        let same1 = 0;
        let c1 = 0;
        for (let q = 1; q + 1 < wLen; q += 2) {
            c1 += 1;
            if (handOf(ls[q]) === handOf(ls[q + 1]) && Math.abs(ls[q] - ls[q + 1]) === 1) same1 += 1;
        }
        const pairFrac = Math.max(c0 > 0 ? same0 / c0 : 0.0, c1 > 0 ? same1 / c1 : 0.0);
        const sRot = Math.max(0.0, Math.min(1.0, (distinct - 2) / 2.0));
        const sShape = Math.max(0.0, Math.min(1.0, (pairFrac - 0.34) / 0.66));
        const sRate = rcRotSmoothstep((rate - loNps) / Math.max(1e-6, hiNps - loNps));
        raw[k] = delta * sRate * sShape * sRot;
    }
    // 把单键行得分摊到合并行（同刻/邻刻加权，避免边界硬跳变）
    const out = new Float64Array(mergedTimes.length);
    let ptr = 0;
    for (let j = 0; j < mergedTimes.length; j += 1) {
        const t = mergedTimes[j];
        while (ptr + 1 < nS && Math.abs(rowTimes[singleIdx[ptr + 1]] - t) <= Math.abs(rowTimes[singleIdx[ptr]] - t)) ptr += 1;
        let wsum = 0.0;
        let acc = 0.0;
        for (let k = ptr; k >= 0; k -= 1) {
            const dt = t - rowTimes[singleIdx[k]];
            if (dt > reach) break;
            const w = Math.exp(-0.5 * (dt / reach) * (dt / reach));
            acc += w * raw[k];
            wsum += w;
        }
        for (let k = ptr + 1; k < nS; k += 1) {
            const dt = rowTimes[singleIdx[k]] - t;
            if (dt > reach) break;
            const w = Math.exp(-0.5 * (dt / reach) * (dt / reach));
            acc += w * raw[k];
            wsum += w;
        }
        out[j] = wsum > 0.0 ? Math.exp(-acc / wsum) : 1.0;
    }
    return out;
}

// ---- RC v14：叠内挖空（抬指/换指）修正（4K/6K/7K 通用，连续量）----
// 物理背景：满叠/中叠谱里「平拍」与「挖空」相邻时，音轨集会变化，手指必须抬起再按下
// （或换到相邻轨）；快速叠流里这比纯平拍更费手。旧模型只看行内押数，于是
//   4343（4↔3 交替）、4242（1234↔14）、3333（124↔134）都会被判低。
// 本层不判键型，只用连续物理量：
//   1) 相邻两行的「换指量」z = |A_{k−1} Δ A_k| / kTotal（对称差 / 键数）：
//      平拍 0；4343（同一轨抬落）1/k；4242 或 3333（两轨换位）2/k；按比例连续。
//   2) 满叠度 fill = (c_{k−1} + c_k) / (2·kTotal)：只有接近满叠的叠流里，换指才意味着
//      「抬指再重按」；普通 jumpstream 的换轨被连续门控压低。
//   3) 局部行速率（行/秒）与满叠度都用连续 smoothstep，无分段硬阈值。
// 行值乘 exp(delta · sRate · sFill · z)，沿时间高斯平滑。
// 4K/6K/7K 共用同一公式与常量（z 归一化到 kTotal）；LN 通道不经过本层。
const RC_HOLLOW_LO_RPS = 5.0568; // 行速率上升段起点（行/秒）
const RC_HOLLOW_HI_RPS = 9.1800; // 行速率上升段终点（峰值区开始）
const RC_HOLLOW_DOWN_LO_RPS = 11.7600; // 行速率衰减段起点（密度再高时抬指被判定窗吞掉）
const RC_HOLLOW_DOWN_HI_RPS = 16.8642; // 行速率衰减段终点（此后不再修正）
const RC_HOLLOW_DELTA = 0.2910; // 标定：_analysis/hollow_calib3.py（4K 映射层同步重拟合） // 乘在换指量 z 上的增益（log 空间；标定见 _analysis/hollow_calib2.py）
const RC_HOLLOW_FILL_LO = 0.5610; // 满叠度门控下界
const RC_HOLLOW_FILL_HI = 1.0000; // 满叠度门控上界
const RC_HOLLOW_WIN = 4; // 行速率统计窗口（±行数）
const RC_HOLLOW_REACH_MS = 47.0755; // 沿时间平滑半径
const RC_HOLLOW_BAL_MIX = 1.0000; // 缺轨轮换（非同轨挖空）相对固定缺轨的削减比例

function rcHollowMultFn(lanes, kTotal, mergedTimes) {
    if (!(kTotal >= 4) || !lanes || lanes.length === 0 || mergedTimes.length === 0) return null;
    const onsets = [];
    for (let c = 0; c < kTotal; c += 1) {
        const lane = lanes[c];
        if (!lane) continue;
        for (let i = 0; i < lane.length; i += 1) onsets.push([lane[i].t, c]);
    }
    if (onsets.length < 6) return null;
    onsets.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const rowTimes = [];
    const rowCols = [];
    let curT = onsets[0][0];
    let curCols = [onsets[0][1]];
    for (let i = 1; i < onsets.length; i += 1) {
        const t = onsets[i][0];
        const c = onsets[i][1];
        if (Math.abs(t - curT) <= 1e-6) curCols.push(c);
        else { rowTimes.push(curT); rowCols.push(curCols); curT = t; curCols = [c]; }
    }
    rowTimes.push(curT);
    rowCols.push(curCols);
    const nRows = rowTimes.length;
    if (nRows < 2) return null;

    const loRps = v11Cfg("hollowLo", RC_HOLLOW_LO_RPS);
    const hiRps = v11Cfg("hollowHi", RC_HOLLOW_HI_RPS);
    const delta = v11Cfg("hollow", RC_HOLLOW_DELTA);
    const fillLo = v11Cfg("hollowFillLo", RC_HOLLOW_FILL_LO);
    const fillHi = v11Cfg("hollowFillHi", RC_HOLLOW_FILL_HI);
    const win = Math.max(1, Math.round(v11Cfg("hollowWin", RC_HOLLOW_WIN)));
    const reach = v11Cfg("hollowReach", RC_HOLLOW_REACH_MS);
    // 逐行「音轨集变化量」d（对称差 / 键数，连续、与键数无关）
    const dArr = new Float64Array(nRows);
    for (let k = 1; k < nRows; k += 1) {
        const a = rowCols[k - 1];
        const b = rowCols[k];
        let inter = 0;
        for (const c of b) if (a.includes(c)) inter += 1;
        dArr[k] = (a.length + b.length - 2 * inter) / kTotal;
    }
    const raw = new Float64Array(nRows);
    for (let k = 1; k < nRows; k += 1) {
        const a = rowCols[k - 1];
        const b = rowCols[k];
        // 换指量本身（对称差 / 键数），连续、与键数无关；持续性由窗口占比门控
        const z = dArr[k];
        // 满叠度：两行里较大押数 / 键数（4242 需要 4 键行参与才算「叠」）
        const fill = Math.max(a.length, b.length) / kTotal;
        const sFill = rcRotSmoothstep((fill - fillLo) / Math.max(1e-6, fillHi - fillLo));
        // 局部行速率（±win 行窗口）
        const b0 = Math.max(0, k - win);
        const b1 = Math.min(nRows - 1, k + win);
        const span = rowTimes[b1] - rowTimes[b0];
        const rps = (b1 > b0 && span > 1e-6) ? ((b1 - b0) * 1000.0) / span : 0.0;
        const dlo = v11Cfg("hollowDownLo", RC_HOLLOW_DOWN_LO_RPS);
        const dhi = v11Cfg("hollowDownHi", RC_HOLLOW_DOWN_HI_RPS);
        const sRateUp = rcRotSmoothstep((rps - loRps) / Math.max(1e-6, hiRps - loRps));
        const sRateDown = 1.0 - rcRotSmoothstep((rps - dlo) / Math.max(1e-6, dhi - dlo));
        const sRate = sRateUp * sRateDown;
        // 持续性：窗口内「周期 2 的挖空/换指」占比（连续）：
        //   本排押数与两排前相同（4→3→4 / 3→3 / 4→2→4 都满足），且音轨集相对上一排在变。
        //   偶发缺一轨、或和弦升级/降级的一次性变化都只贡献很小占比。
        let altShare = 0.0;
        let altCnt = 0;
        for (let j = Math.max(1, k - win); j <= Math.min(nRows - 1, k + win); j += 1) {
            altCnt += 1;
            if (j >= 2) {
                const cj = rowCols[j].length;
                const cj2 = rowCols[j - 2].length;
                const sizeSame = Math.max(0.0, 1.0 - Math.abs(cj - cj2) / Math.max(1, cj2));
                altShare += sizeSame * Math.min(1.0, dArr[j] / 0.25);
            }
        }
        altShare = altCnt > 0 ? altShare / altCnt : 0.0;
        const altLo = v11Cfg("hollowAltLo", 0.9500);
        const altHi = v11Cfg("hollowAltHi", 1.0000);
        const sAlt = rcRotSmoothstep((altShare - altLo) / Math.max(1e-6, altHi - altLo));
        // 同轨挖空平衡：缺轨在相邻两次挖空之间是否相同。
        // 同一轨反复抬落（如 (1234)(123) 固定缺 4 轨）最难；缺轨轮换（(1234)(123)(1234)(124)）
        // 手指可在两轨间交替，实际更简单 —— 用缺轨集对称差连续加权。
        let bal = 1.0;
        if (k >= 2) {
            let holeSame = 0;
            let holeDiff = 0;
            for (let c = 0; c < kTotal; c += 1) {
                const inA = b.includes(c);
                const inB = rowCols[k - 2].includes(c);
                if (!inA && !inB) holeSame += 1;
                else if (inA !== inB) holeDiff += 1;
            }
            const sameHole = holeDiff + holeSame > 0 ? holeSame / (holeSame + holeDiff) : 1.0;
            // 行速率越高，抬指越可能被判定窗吞掉 → 轮换缺轨的「更简单」优势衰减，甚至反转
            const balLo = v11Cfg("hollowBalLo", 11.0);
            const balHi = v11Cfg("hollowBalHi", 15.0);
            const balGain = v11Cfg("hollowBalGain", 3.0);
            const downFrac = rcRotSmoothstep((rps - balLo) / Math.max(1e-6, balHi - balLo));
            const mixEff = v11Cfg("hollowBalMix", RC_HOLLOW_BAL_MIX) * (1.0 - balGain * downFrac);
            bal = 1.0 - mixEff * (1.0 - sameHole);
        }
        raw[k] = delta * sRate * sFill * sAlt * bal * z;
    }
    // 沿时间高斯平滑并映射到合并行（同刻/邻刻加权，避免边界硬跳变）
    const out = new Float64Array(mergedTimes.length);
    let ptr = 0;
    for (let j = 0; j < mergedTimes.length; j += 1) {
        const t = mergedTimes[j];
        while (ptr + 1 < nRows && Math.abs(rowTimes[ptr + 1] - t) <= Math.abs(rowTimes[ptr] - t)) ptr += 1;
        let wsum = 0.0;
        let acc = 0.0;
        for (let k = ptr; k >= 0; k -= 1) {
            const dt = t - rowTimes[k];
            if (dt > reach) break;
            const w = Math.exp(-0.5 * (dt / reach) * (dt / reach));
            acc += w * raw[k];
            wsum += w;
        }
        for (let k = ptr + 1; k < nRows; k += 1) {
            const dt = rowTimes[k] - t;
            if (dt > reach) break;
            const w = Math.exp(-0.5 * (dt / reach) * (dt / reach));
            acc += w * raw[k];
            wsum += w;
        }
        out[j] = wsum > 0.0 ? Math.exp(acc / wsum) : 1.0;
    }
    if (globalThis.__RC_HOLLOW_DEBUG) globalThis.__RC_DEBUG_HOLLOW = out;
    return out;
}

export function calculateFixedDensity(source, rate = 1.0, options = {}) {
    const parsed = resolveParsed(source, options?.cvtFlag ?? null);
    if (parsed?.status === "NotMania") {
        throw new Error("Beatmap mode is not mania");
    }
    if (parsed?.status === "Fail") {
        throw new Error("Beatmap parse failed");
    }

    const rawLanes = collectLanes(parsed);
    const kTotal = rawLanes.length;
    if (kTotal < 1) {
        return { rcValue: Number.NaN, lnValue: Number.NaN, rcSequence: null, lnSequence: null, rcTicks: [], lnTicks: [], overlaps: { rc: [], ln: [], total: 0 } };
    }

    // LN 长度门 / 判定窗依赖 OD；缺失（<0）时回退 8.0（与 benchmark 一致）。
    const odRaw = Number(parsed?.od);
    const od = (Number.isFinite(odRaw) && odRaw >= 0.0) ? odRaw : 8.0;

    // 变速：时间戳按 1/rate 缩放（rate=1 时不变）。
    const rateNorm = normalizeRate(rate);
    const invRate = 1.0 / rateNorm;
    const lanes = invRate === 1.0
        ? rawLanes
        : rawLanes.map((lane) => lane.map((n) => ({ t: n.t * invRate, len: n.len * invRate })));

    const onsetSet = new Set();
    for (const lane of lanes) {
        for (const n of lane) {
            onsetSet.add(n.t);
        }
    }
    const allOnsets = Array.from(onsetSet).sort((a, b) => a - b);
    const onsetCount = allOnsets.length;

    const laneTimes = lanes.map((lane) => lane.map((n) => n.t));
    const laneGroups = laneTimes.map(groupLane);
    // 重叠键（同轨同时间戳多个物件）统一按「一个物件」计入总数：
    // 每轨的物件数 = 该轨的时间分组数。
    let noteCount = 0;
    for (const gs of laneGroups) noteCount += gs.length;
    const chord = noteCount / Math.max(onsetCount, 1);
    const lnN = Math.log(Math.max(noteCount, 1));
    const laneGroupTimes = laneGroups.map((gs) => gs.map((g) => g.t));
    // 同轨同时间戳的多个物件按「一个物件」计算（不再做超限替代/溢出累加），
    // 因此组大小统一为 1；重叠情况只用于图上的红线提示（detectOverlaps）。
    const laneGroupSizes = laneGroups.map((gs) => gs.map(() => 1));
    const overlaps = detectOverlaps(lanes);

    // 6K 手内轨道布局（4K = 每手两轨；6K = 每手三轨，含 12/23 重叠配对）
    const hands = handLayoutOf(kTotal);
    const handOfLane = new Array(kTotal).fill(null);
    for (const hand of hands) {
        for (let r = 0; r < hand.length; r += 1) {
            handOfLane[hand[r]] = { hand, rank: r };
        }
    }

    // RC 通道：逐轨贡献（softmax 合并 + 邻域额外 contribution）
    const laneContribsList = [];
    for (let i = 0; i < kTotal; i += 1) {
        const info = handOfLane[i];
        const mateIdx = info ? handMateOf(info.hand, info.rank) : i;
        laneContribsList.push(laneContribs(
            laneGroups[i],
            laneGroupTimes[mateIdx],
            laneGroupSizes[mateIdx],
        ));
    }

    // 单轨局部熵乘子（贡献层）：contribution *= exp(γ_lane · irr_lane(t))
    if (IRR_GAIN_LANE !== 0.0) {
        for (let i = 0; i < kTotal; i += 1) {
            const tt = laneGroupTimes[i];
            if (tt.length === 0) {
                continue;
            }
            const ent = localIrregularity(tt);
            const contribs = laneContribsList[i];
            for (let k = 0; k < tt.length; k += 1) {
                contribs[k] *= Math.exp(IRR_GAIN_LANE * ent[k]);
            }
        }
    }

    // 单手两轨左右不均：时间衰减序列乘子
    if (HAND_IMB_ETA !== 0.0) {
        const allNoteTimes = [];
        for (const lane of lanes) {
            for (const n of lane) {
                allNoteTimes.push(n.t);
            }
        }
        // kernelSumSeries 需要升序输入；allNoteTimes 按轨道拼接，全局未必有序。
        allNoteTimes.sort((a, b) => a - b);
        const handWeights = handImbalanceWeights(laneGroups, laneGroupTimes, allNoteTimes);
        for (let i = 0; i < kTotal; i += 1) {
            const contribs = laneContribsList[i];
            const w = handWeights[i];
            for (let k = 0; k < contribs.length; k += 1) {
                contribs[k] *= 1.0 + HAND_IMB_ETA * w[k] / (1.0 + w[k]);
            }
        }
    }

    // 第一级合并：手内相邻双轨配对（softmax 合并为一条赋值序列）
    //   4K：每手 1 对 → (0,1)、(2,3)
    //   6K：每手 2 对重叠窗口 → 12 = (0,1) / (3,4)，23 = (1,2) / (4,5)
    const pairBandSequence = (start, mate) => {
        const groupsA = laneGroups[start];
        const groupsB = laneGroups[mate];
        if (groupsA.length === 0 && groupsB.length === 0) {
            return [];
        }
        const timesA = laneGroupTimes[start];
        const sizesA = laneGroupSizes[start];

        let unionTimes;
        let counts;
        let curveA;
        let curveB;
        if (mate === start) {
            unionTimes = timesA;
            counts = sizesA;
            curveA = laneCurve(timesA, laneContribsList[start], unionTimes);
            curveB = curveA;
        } else {
            const merged = mergeGroupSizes(timesA, sizesA, laneGroupTimes[mate], laneGroupSizes[mate]);
            unionTimes = merged.times;
            counts = merged.countsA.map((v, idx) => v + merged.countsB[idx]);
            curveA = laneCurve(timesA, laneContribsList[start], unionTimes);
            curveB = laneCurve(laneGroupTimes[mate], laneContribsList[mate], unionTimes);
        }

        const seq = [];
        if (mate === start) {
            for (let k = 0; k < unionTimes.length; k += 1) {
                if (counts[k] > 0) {
                    seq.push([unionTimes[k], curveA[k], counts[k]]);
                }
            }
        } else {
            for (let k = 0; k < unionTimes.length; k += 1) {
                const c = counts[k];
                if (c <= 0) {
                    continue;
                }
                seq.push([unionTimes[k], mergeOp(curveA[k], curveB[k]), c]);
            }
        }
        return seq;
    };

    // 手内合并层（一手 > 2 轨，即 6K 双手 / 7K 右手）：把相邻轨配对序列（band）合并成手内序列。
    //   时刻取手内轨道并集；物件数取手内并集计数（中轨不重复计）。
    //   beta_1.0.7 及以前：v = HAND_NORM_SCALE * (Σ v_i^p)^(1/p)，p = 2（2-范数）。
    //   2-范数会把「冗余的 band 数」当成独立难度叠加（√n 放大），左手 2 条 band、
    //   右手 3 条 band 时右手被系统性抬高。beta_1.1.1 起 RC 改用 hand-size 中性的
    //   算术平均 HAND_MERGE_MODE="mean"（Σv/n，与 band 数无关）。
    //   LN 基准仍按旧 2-范数密度标定，故同时算出旧 p2 序列，专供 LN 密度使用。
    const mergeHandBands = (hand, bands) => {
        let handTimes = laneGroupTimes[hand[0]];
        let handSizes = laneGroupSizes[hand[0]];
        for (let r = 1; r < hand.length; r += 1) {
            const merged = mergeGroupSizes(handTimes, handSizes,
                laneGroupTimes[hand[r]], laneGroupSizes[hand[r]]);
            handTimes = merged.times;
            handSizes = merged.countsA.map((v, idx) => v + merged.countsB[idx]);
        }
        const queryTimes = Float64Array.from(handTimes);
        const bandValues = bands.map((seq) => sampleDecay(seq, queryTimes));
        // 每行「该 band 是否有音符」：用于 active 归一（空轨/未参与该行的 band 不摊薄均值）。
        const bandRowCounts = bands.map((seq) => {
            const m = new Map();
            for (const r of seq) m.set(r[0], (m.get(r[0]) || 0) + r[2]);
            return m;
        });
        // 每个 band 的「两轨同时按下」时刻集合（用于 track-pair 乘积层，排除单轨经过造成的伪激活）。
        const bandBothTimes = bands.map((_, bi) => {
            const a = new Set(laneGroupTimes[hand[bi]].map((t) => t));
            const b = new Set(laneGroupTimes[hand[bi + 1]].map((t) => t));
            const out = new Set();
            for (const t of a) if (b.has(t)) out.add(t);
            return out;
        });
        const nBands = bandValues.length;
        // 序列级 band 难度：默认取“该 band 有音符的行”的均值；
        // pairprod 模式取“两轨同时按下”的行（track-pair 的真实使用强度）。
        const bandLevel = [];
        const useBoth = (typeof globalThis !== 'undefined')
            && (globalThis.__MERGE_MODE === 'pairprod' || globalThis.__MERGE_MODE === 'prodblend'
                || globalThis.__MERGE_MODE === 'prodadd');
        for (let bi = 0; bi < nBands; bi += 1) {
            let acc = 0.0;
            let cnt = 0;
            for (let k = 0; k < queryTimes.length; k += 1) {
                const hit = useBoth
                    ? (bandBothTimes[bi].has(queryTimes[k]) && bandRowCounts[bi].has(queryTimes[k]))
                    : bandRowCounts[bi].has(queryTimes[k]);
                if (hit) {
                    acc += bandValues[bi][k];
                    cnt += 1;
                }
            }
            bandLevel.push(cnt > 0 ? acc / cnt : 0.0);
        }
        // 乘积式手内合并因子（序列级）：
        //   C = Σ_runs s·(Π_{i∈run}(1 + A_i/s) − 1)， runs = 相邻重叠 band 的最大连续段
        //   M = C / Σ A_i  （单 band 时 M = 1）
        let prodFactor = 1.0;
        if (typeof globalThis !== 'undefined') globalThis.__PROD_DBG = { bandLevel: [], factor: 1.0 };
        if (nBands > 1) {
            let sSum = 0.0;
            let sCnt = 0;
            let aSum = 0.0;
            for (let bi = 0; bi < nBands; bi += 1) {
                if (bandLevel[bi] > 0.0) {
                    sSum += bandLevel[bi];
                    sCnt += 1;
                }
                aSum += bandLevel[bi];
            }
            if (sCnt > 1 && aSum > 0.0) {
                const sRef = sSum / sCnt;
                let c = 0.0;
                let runStart = 0;
                for (let q = 1; q <= nBands; q += 1) {
                    if (q === nBands || bandLevel[q] <= 0.0 || bandLevel[q - 1] <= 0.0) {
                        let prod = 1.0;
                        let runSum = 0.0;
                        for (let r = runStart; r < q; r += 1) {
                            prod *= (1.0 + bandLevel[r] / sRef);
                            runSum += bandLevel[r];
                        }
                        c += (runSum > 0.0) ? sRef * (prod - 1.0) : 0.0;
                        runStart = q;
                    }
                }
                prodFactor = c / aSum;
                if (typeof globalThis !== 'undefined' && globalThis.__PROD_DBG) {
                    globalThis.__PROD_DBG.bandLevel = bandLevel.slice();
                    globalThis.__PROD_DBG.factor = prodFactor;
                }
            }
        }
        const g = (typeof globalThis !== 'undefined') ? globalThis : null;
        const mode = (g && typeof g.__MERGE_MODE === 'string') ? g.__MERGE_MODE : HAND_MERGE_MODE;
        const p = (g && Number.isFinite(g.__MERGE_P)) ? g.__MERGE_P : HAND_NORM_P;
        const scale = (g && Number.isFinite(g.__MERGE_SCALE)) ? g.__MERGE_SCALE : HAND_MERGE_SCALE;
        const bothW = (g && Number.isFinite(g.__MERGE_BOTH)) ? g.__MERGE_BOTH : HAND_MERGE_BOTH;
        const invP = 1.0 / p;
        const main = [];
        const p2 = [];
        // 密度越高，和弦尺寸/形状差异越容易被“手速”抹平（与 4K 对拍-平拍在高密度收敛同构）。
        let firstT = 0.0;
        let lastT = 0.0;
        let rowCnt = 0;
        for (let k = 0; k < handSizes.length; k += 1) {
            if (handSizes[k] > 0) {
                const t = queryTimes[k];
                if (rowCnt === 0) firstT = t;
                lastT = t;
                rowCnt += 1;
            }
        }
        const handRps = (rowCnt > 1 && lastT > firstT) ? (rowCnt - 1) / ((lastT - firstT) / 1000.0) : 0.0;
        const altT = Math.min(1.0, Math.max(0.0, (handRps - 8.0) / 12.0));
        const altFade = 1.0 - altT * altT * (3.0 - 2.0 * altT);
        const chordAlt = (Number.isFinite(MERGE_CHORD_ALT) && nBands > 1 && kTotal >= 7) ? MERGE_CHORD_ALT * altFade : 0.0;
        for (let k = 0; k < queryTimes.length; k += 1) {
            const c = handSizes[k];
            if (!(c > 0)) {
                continue;
            }
            const cPrev = (k > 0 && handSizes[k - 1] > 0) ? handSizes[k - 1] : c;
            const dSize = Math.abs(c - cPrev);
            const altW = dSize >= 3 ? 1.0 : (dSize === 2 ? 0.8 : (dSize === 1 ? 0.0 : 0.0));
            const altGain = 1.0 + chordAlt * altW;
            let v = 0.0;
            if (mode === 'pnorm' || mode === 'rms') {
                let acc = 0.0;
                for (const vals of bandValues) {
                    acc += Math.pow(Math.max(vals[k], 0.0), p);
                }
                v = Math.pow(acc, invP) * (mode === 'rms' ? Math.pow(1.0 / nBands, invP) : 1.0);
            } else if (mode === 'mean' || mode === 'prodfac' || mode === 'prodactive' || mode === 'pairprod' || mode === 'prodblend' || mode === 'prodadd') {
                let acc = 0.0;
                if (mode === 'prodactive' || mode === 'pairprod' || mode === 'prodblend') {
                    // 只对本行有键的 band 取均值（消除空轨/手宽稀释），再乘序列级乘积因子。
                    let act = 0;
                    for (let bi = 0; bi < nBands; bi += 1) {
                        if (bandRowCounts[bi].has(queryTimes[k])) {
                            acc += bandValues[bi][k];
                            act += 1;
                        }
                    }
                    v = act > 0 ? (acc / act) * prodFactor : 0.0;
                } else {
                    for (const vals of bandValues) acc += vals[k];
                    v = (acc / nBands) * (mode === 'prodfac' ? prodFactor : 1.0);
                }
                if (mode === 'prodblend' || mode === 'prodadd') {
                    const lam = (typeof globalThis !== 'undefined' && Number.isFinite(globalThis.__MERGE_LAMBDA))
                        ? globalThis.__MERGE_LAMBDA
                        : (Number.isFinite(MERGE_LAMBDA_BY_KEY[kTotal]) ? MERGE_LAMBDA_BY_KEY[kTotal] : MERGE_BLEND_LAMBDA);
                    v *= 1.0 + lam * Math.log(Math.max(prodFactor, 1e-6));
                }
            } else if (mode === 'active' || mode === 'activesum') {
                // 只对「本行实际有键」的 band 求和：消除手宽/空轨带来的 1/nBands 稀释。
                let acc = 0.0;
                let act = 0;
                for (let bi = 0; bi < nBands; bi += 1) {
                    if (bandRowCounts[bi].has(queryTimes[k])) {
                        acc += bandValues[bi][k];
                        act += 1;
                    }
                }
                if (act === 0) {
                    for (const vals of bandValues) acc += vals[k];
                    act = nBands;
                }
                v = (mode === 'activesum' ? acc : acc / act);
                if (mode === 'activesum') v = acc / Math.max(1.0, Math.sqrt(act));
            } else if (mode === 'product') {
                // 乘积式手内合并：把「有键的 band」按相邻重叠关系分组，组内做
                //   v_grp = s·(Π (1 + v_i/s) − 1)  （单 band 时退化为 v_i，天然 ≥1 下界再减 1）
                // 组间相加。相邻两 band（共享一轨，如 12/23）落在同一组 → 相互放大；
                // 相距较远的 band（如 12/34）落在不同组 → 近似线性叠加，稍简单。
                const actIdx = [];
                const actVal = [];
                for (let bi = 0; bi < nBands; bi += 1) {
                    if (bandRowCounts[bi].has(queryTimes[k])) {
                        actIdx.push(bi);
                        actVal.push(bandValues[bi][k]);
                    }
                }
                if (actIdx.length === 0) {
                    for (const vals of bandValues) { v += vals[k]; }
                    v /= nBands;
                } else if (actIdx.length === 1) {
                    v = actVal[0];
                } else {
                    let acc = 0.0;
                    for (const x of actVal) acc += x;
                    const sRef = Math.max(acc / actVal.length, 1e-9);
                    let runStart = 0;
                    for (let q = 1; q <= actIdx.length; q += 1) {
                        if (q === actIdx.length || actIdx[q] !== actIdx[q - 1] + 1) {
                            // [runStart, q) 是一个相邻重叠组
                            let prod = 1.0;
                            for (let r = runStart; r < q; r += 1) prod *= (1.0 + actVal[r] / sRef);
                            v += sRef * (prod - 1.0);
                            runStart = q;
                        }
                    }
                }
            } else if (mode === 'max') {
                let m = 0.0;
                for (const vals of bandValues) m = Math.max(m, vals[k]);
                v = m;
            } else {
                // softmax 链：逐条 band 用 mergeOp 合并，尺度不随 band 数放大
                v = bandValues[0][k];
                for (let bi = 1; bi < nBands; bi += 1) v = mergeOp(v, bandValues[bi][k]);
                if (bothW !== 0.0 && nBands > 1) {
                    let mn = Infinity;
                    for (const vals of bandValues) mn = Math.min(mn, vals[k]);
                    v += bothW * mn;
                }
            }
            main.push([queryTimes[k], scale * v * altGain, c]);
            // 旧 2-范数（LN 密度锚定用；与 1.0.7 的 RC 手内合并同式）
            let acc2 = 0.0;
            for (const vals of bandValues) acc2 += vals[k] * vals[k];
            p2.push([queryTimes[k], Math.sqrt(acc2), c]);
        }
        return { main, p2 };
    };

    const handSequences = [];
    const handSequencesP2 = [];
    for (const hand of hands) {
        const bands = [];
        // 单手只有 1 轨（1K/2K 每手 1 轨、3K 左手 1 轨）：没有相邻对，直接用该轨自身序列。
        if (hand.length === 1) {
            const solo = pairBandSequence(hand[0], hand[0]);
            if (solo.length > 0) {
                handSequences.push(solo);
                handSequencesP2.push(solo);
            }
            continue;
        }
        for (let r = 0; r + 1 < hand.length; r += 1) {
            const seq = pairBandSequence(hand[r], hand[r + 1]);
            if (seq.length > 0) {
                bands.push(seq);
            }
        }
        if (bands.length === 0) {
            continue;
        }
        if (bands.length === 1) {
            handSequences.push(bands[0]);
            handSequencesP2.push(bands[0]);
        } else {
            const mergedHand = mergeHandBands(hand, bands);
            handSequences.push(mergedHand.main);
            handSequencesP2.push(mergedHand.p2);
        }
    }

    // 第二级合并：双手序列视为赋值突变衰减函数，逐时刻 softmax 合并
    RC11_KTOTAL = kTotal;
    let mergedSeq = [];
    for (const seq of handSequences) {
        mergedSeq = mergeDecaySequences(mergedSeq, seq);
    }
    // LN 密度锚定序列：6K/7K 的 LN 基准按旧 2-范数合并标定，这里保留旧口径的
    // 合并结果（仅作为 LN 的密度输入，RC 的数值/曲线不使用）。
    const needP2 = (kTotal === 6 || kTotal === 7)
        && typeof globalThis !== 'undefined' && globalThis.__MERGE_MODE !== 'pnorm'
        && HAND_MERGE_MODE !== 'pnorm';
    let mergedSeqP2 = mergedSeq;
    if (needP2) {
        mergedSeqP2 = [];
        for (const seq of handSequencesP2) {
            mergedSeqP2 = mergeDecaySequences(mergedSeqP2, seq);
        }
    }

    // 全谱局部熵乘子（最终合并层）：v(t) *= exp(γ_global · irr_global(t))
    // 6K/7K 的 LN 密度锚定序列（p2）与 RC 序列时间轴一致，需乘上同一熵增益，
    // 否则 LN 的 n 会和 1.0.7 产生偏差。
    if (IRR_GAIN_GLOBAL !== 0.0 && mergedSeq.length > 0) {
        const mergedTimes = Float64Array.from(mergedSeq, (row) => row[0]);
        const gEnt = localIrregularity(allOnsets);
        const e = entropyAt(allOnsets, gEnt, mergedTimes);
        for (let k = 0; k < mergedSeq.length; k += 1) {
            const gain = Math.exp(IRR_GAIN_GLOBAL * e[k]);
            mergedSeq[k][1] *= gain;
            if (mergedSeqP2 !== mergedSeq && k < mergedSeqP2.length) {
                mergedSeqP2[k][1] *= gain;
            }
        }
    }

    // 双手合并层：left/right 同刻 chord/协调权重（纯单键/stream 的 s=0）
    if (IRR_GAIN_CROSS !== 0.0 && mergedSeq.length > 0) {
        const mergedTimes = Float64Array.from(mergedSeq, (row) => row[0]);
        const crossW = crossHandWeight(lanes, allOnsets);
        const e = entropyAt(allOnsets, crossW, mergedTimes);
        for (let k = 0; k < mergedSeq.length; k += 1) {
            mergedSeq[k][1] *= Math.exp(IRR_GAIN_CROSS * e[k]);
        }
    }

    // RC v10：逐事件局部物理维度乘子（仅 4K）
    const rc10Feat = ((kTotal >= 4) && mergedSeq.length > 0)
        ? rc10LocalFeatures(lanes, kTotal, Float64Array.from(mergedSeq, (row) => row[0]))
        : null;
    const rc10Mult = rc10LocalMultiplier(rc10Feat);
    // RC v13：高密度单键轮换折叠乘子（仅 4K；6K/7K 返回 null，行为不变）
    const rcRotMult = (kTotal >= 4 && mergedSeq.length > 0)
        ? rcRotDiscount(lanes, kTotal, Float64Array.from(mergedSeq, (row) => row[0]))
        : null;
    // RC v14：叠内挖空（抬指/换指）修正（4K/6K/7K 连续量，LN 不经过）
    const rcHollowMult = (mergedSeq.length > 0)
        ? rcHollowMultFn(lanes, kTotal, Float64Array.from(mergedSeq, (row) => row[0]))
        : null;

    // 实验：整手和弦的「一次动作 + 加成」连续模型（rcStackW > 0 时启用）。
    // 逐行：c = 该排押数，o = 与上一排音轨集的 Jaccard 重合度（i=0 时 o=0）。
    //   factor = 1 / (1 + w · (c − 1)^q · o^p)
    // 物理含义：一排同时按 c 键 = 一次整手动作 + 少量加成，不是 c 倍的逐轨密度；
    // o→1（同一组键反复拍）时按一次动作计；o→0（换轨 chordjack）几乎不打折。
    // w=1,q=1 时长叠拍 ≈ 单键拍；连续可调，且由时间平滑层保证不抖动。
    const rcStackW = v11Cfg(`rcStackW${kTotal}`, v11Cfg("rcStackW", 0.0));
    if (rcStackW !== 0.0 && typeof lanes !== 'undefined' && Array.isArray(lanes)) {
        const onS = [];
        for (let c = 0; c < kTotal; c += 1) {
            const lane = lanes[c];
            if (!lane) continue;
            for (let i = 0; i < lane.length; i += 1) onS.push([lane[i].t, c]);
        }
        onS.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
        const sT = []; const sS = [];
        let tCur = onS.length ? onS[0][0] : 0; let setCur = onS.length ? [onS[0][1]] : [];
        for (let i = 1; i < onS.length; i += 1) {
            const t = onS[i][0]; const c = onS[i][1];
            if (Math.abs(t - tCur) <= 1e-6) setCur.push(c);
            else { sT.push(tCur); sS.push(setCur.slice()); tCur = t; setCur = [c]; }
        }
        if (onS.length) { sT.push(tCur); sS.push(setCur.slice()); }
        const q = v11Cfg("rcStackQ", 1.0);
        const p = v11Cfg("rcStackP", 2.0);
        let ptrS = 0;
        for (const row of mergedSeq) {
            const t = row[0];
            while (ptrS + 1 < sT.length && Math.abs(sT[ptrS + 1] - t) <= Math.abs(sT[ptrS] - t)) ptrS += 1;
            const b = sS[ptrS];
            const c = Math.max(1, Math.round(row[2] || 1));
            if (c <= 1) continue;
            let o = 0.0;
            if (ptrS > 0) {
                const a = sS[ptrS - 1];
                let inter = 0;
                for (const x of b) if (a.includes(x)) inter += 1;
                const uni = a.length + b.length - inter;
                o = uni > 0 ? inter / uni : 0.0;
            }
            const sizeTerm = Math.pow(c - 1, q);
            const factor = 1.0 / (1.0 + rcStackW * sizeTerm * Math.pow(o, p));
            if (factor !== 1.0) row[1] *= factor;
        }
    }
    let totalCount = 0;
    for (const row of mergedSeq) {
        totalCount += row[2];
    }
    if (totalCount === 0) {
        return { rcValue: Number.NaN, lnValue: Number.NaN, rcSequence: null, lnSequence: null, rcTicks: [], lnTicks: [], overlaps: { rc: [], ln: [], total: 0 } };
    }

    // ---- 耐力层（在分位之前）：长时间维持相近难度 → 对数累积加成；难度突变/断档 → 快速衰减 ----
    // E_i = E_{i−1}·exp(−(1−sim)·resetK − max(0,dt−gap)/tau) + sim·min(dt,dtCap)/T0
    // sim = exp(−(ln(v_i/v_{i−1})/σ)²)；endMult_i = 1 + β_k·ln(1 + E_i)
    const rcEndBeta = v11Cfg(`rcEnd${kTotal}`, v11Cfg("rcEnd", (RC_END_BETA[kTotal] || 0.0)));
    const rcEndG = v11Cfg(`rcEndG${kTotal}`, v11Cfg("rcEndG", (RC_END_GBETA[kTotal] || 0.0)));
    const endMult = new Float64Array(mergedSeq.length).fill(1.0);
    let rcEndScore = 0.0;
    if (mergedSeq.length > 1) {
        const sigma = Math.max(1e-3, v11Cfg("rcEndSig", RC_END_SIGMA));
        const resetK = v11Cfg("rcEndReset", RC_END_RESET_K);
        const tauMs = Math.max(1.0, v11Cfg("rcEndTau", RC_END_TAU_MS));
        const gapMs = Math.max(0.0, v11Cfg("rcEndGap", RC_END_GAP_MS));
        const dtCap = Math.max(1.0, v11Cfg("rcEndDtCap", RC_END_DT_CAP_MS));
        const t0 = Math.max(1.0, v11Cfg("rcEndT0", RC_END_T0_MS));
        const emaTau = Math.max(1.0, v11Cfg("rcEndEma", RC_END_EMA_MS));
        let E = 0.0;
        let ema = 0.0;
        let prevBase = 0.0;
        const eList = [];
        const lanePtrLo = new Int32Array(laneTimes.length);
        const lanePtrHi = new Int32Array(laneTimes.length);
        for (let i = 0; i < mergedSeq.length; i += 1) {
            const row = mergedSeq[i];
            const base = row[1] * (rc10Mult ? rc10Mult[i] : 1.0)
                * (rcRotMult ? rcRotMult[i] : 1.0)
                * (rcHollowMult ? rcHollowMult[i] : 1.0);
            const tRow = row[0];
            let laneNps = 0.0;
            const winStart = tRow - RC_END_WIN_MS;
            for (let li = 0; li < laneTimes.length; li += 1) {
                const arr = laneTimes[li];
                const len = arr.length;
                if (!len) continue;
                // 行时间单调递增：每个轨维护 [winStart, tRow] 的双指针区间，均摊 O(1)
                let p0 = lanePtrLo[li];
                while (p0 < len && arr[p0] < winStart) p0 += 1;
                lanePtrLo[li] = p0;
                let p1 = lanePtrHi[li];
                if (p1 < p0) p1 = p0;
                while (p1 < len && arr[p1] <= tRow) p1 += 1;
                lanePtrHi[li] = p1;
                const nps = (p1 - p0) / (RC_END_WIN_MS / 1000.0);
                if (nps > laneNps) laneNps = nps;
            }
            const sVal = laneNps;
            if (i === 0) {
                ema = sVal;
            } else {
                const dt = Math.max(0.0, row[0] - mergedSeq[i - 1][0]);
                const alpha = 1.0 - Math.exp(-dt / emaTau);
                const rr = sVal - ema;
                const sim = Math.exp(-(rr / sigma) * (rr / sigma));
                const decay = Math.exp(-(1.0 - sim) * resetK - Math.max(0.0, dt - gapMs) / tauMs);
                E = E * decay + sim * Math.min(dt, dtCap) / t0;
                endMult[i] = 1.0 + rcEndBeta * Math.log(1.0 + E);
                ema += alpha * (sVal - ema);
            }
            prevBase = base;
            eList.push(E);
        }
        if (eList.length) {
            eList.sort((a, b) => a - b);
            rcEndScore = eList[Math.min(eList.length - 1, Math.floor(0.75 * (eList.length - 1)))];
        }
    }
    const mergedValues = new Float64Array(totalCount);
    let idx = 0;
    for (let rowIdx = 0; rowIdx < mergedSeq.length; rowIdx += 1) {
        const row = mergedSeq[rowIdx];
        const rowMul = (rc10Mult ? rc10Mult[rowIdx] : 1.0)
            * (rcRotMult ? rcRotMult[rowIdx] : 1.0)
            * (rcHollowMult ? rcHollowMult[rowIdx] : 1.0);
        const v = row[1] * rowMul * endMult[rowIdx];
        const c = row[2];
        for (let k = 0; k < c; k += 1) {
            mergedValues[idx] = v;
            idx += 1;
        }
    }

    const n = findNQuantile(mergedValues, QUANTILE);
    if (!Number.isFinite(n) || n <= 0.0) {
        return { rcValue: Number.NaN, lnValue: Number.NaN, rcSequence: null, lnSequence: null, rcTicks: [], lnTicks: [], overlaps: { rc: [], ln: [], total: 0 } };
    }
    // LN 通道的密度尺度 n 不吸收 RC 专用的轮换折叠：roll 折扣只影响 RC 的感知
    // 难度，LN 基准（103 张）按 b1.0.6 口径标定，这里用「不含轮换折扣」的同一
    // 分位数，保证 LN 数值与 b1.0.6 逐位一致（6K/7K 无折扣，直接复用 n）。
    let nNoRot = n;
    if (rcRotMult || rcHollowMult) {
        const noRotValues = new Float64Array(totalCount);
        let nr = 0;
        for (let rowIdx = 0; rowIdx < mergedSeq.length; rowIdx += 1) {
            const row = mergedSeq[rowIdx];
            const v = row[1] * (rc10Mult ? rc10Mult[rowIdx] : 1.0);
            for (let k = 0; k < row[2]; k += 1) {
                noRotValues[nr] = v;
                nr += 1;
            }
        }
        const nQ = findNQuantile(noRotValues, QUANTILE);
        if (Number.isFinite(nQ) && nQ > 0.0) nNoRot = nQ;
        // 实验：LN 通道使用「新参数」的 RC 分位（不同分位点），等价于给 LN 单独一套分位参数。
        const lnAnchorQ = v11Cfg(`lnAnchorQ${kTotal}`, v11Cfg("lnAnchorQ", QUANTILE));
        if (Math.abs(lnAnchorQ - QUANTILE) > 1e-6) {
            const nQ2 = findNQuantile(noRotValues, Math.min(1.0, Math.max(0.05, lnAnchorQ)));
            if (Number.isFinite(nQ2) && nQ2 > 0.0) nNoRot = nQ2;
        }
    } else {
        const lnAnchorQ = v11Cfg(`lnAnchorQ${kTotal}`, v11Cfg("lnAnchorQ", QUANTILE));
        if (Math.abs(lnAnchorQ - QUANTILE) > 1e-6) {
            const nQ2 = findNQuantile(mergedValues, Math.min(1.0, Math.max(0.05, lnAnchorQ)));
            if (Number.isFinite(nQ2) && nQ2 > 0.0) nNoRot = nQ2;
        }
    }

    // 6K：使用 b1.0.5.1 专用映射层；其余键数沿用 beta_1.0.5 的 4K 映射曲线
    MAP_ACTIVE = (kTotal === 6) ? MAP_PROFILE_6K
        : (kTotal === 7) ? MAP_PROFILE_7K
        : MAP_PROFILE_4K;
    UMAP_ACTIVE = umapFor(kTotal);
    // RC v8：结构特征修正层（连续、无量纲；κ 全 0 时与 beta_1.0.5 逐位一致）
    const rcFeat = rcV8Features(laneTimes, noteCount, onsetCount);
    // 交互项（连续、无量纲；与 _analysis/tune_rc_v8e.py 的 basis() 一致）
    rcFeat.ix_both_lanealt = rcFeat.f_both * rcFeat.f_lanealt;
    rcFeat.ix_both_bothrow = rcFeat.f_both * rcFeat.f_bothrow;
    rcFeat.ix_both_osc = rcFeat.f_both * rcFeat.f_osc;
    rcFeat.ix_both_bal = rcFeat.f_both * rcFeat.f_bal;
    rcFeat.ix_bal_notes = rcFeat.f_bal * rcFeat.f_notes;
    rcFeat.ix_jack_notes = rcFeat.f_jack * rcFeat.f_notes;
    rcFeat.ix_jack_hmax = rcFeat.f_jack * rcFeat.f_handmax;
    rcFeat.ix_dbl_hmax = rcFeat.f_dbl * rcFeat.f_handmax;
    rcFeat.ix_chord_notes = rcFeat.f_chord * rcFeat.f_notes;
    rcFeat.ix_hmax_chord = rcFeat.f_handmax * rcFeat.f_chord;
    rcFeat.ix_osc_notes = rcFeat.f_osc * rcFeat.f_notes;
    rcFeat.ix_lanealt_notes = rcFeat.f_lanealt * rcFeat.f_notes;
    rcFeat.ix_notes_bothrow = rcFeat.f_notes * rcFeat.f_bothrow;
    rcFeat.ix_jack_chord = rcFeat.f_jack * rcFeat.f_chord;
    rcFeat.ix_jack_osc = rcFeat.f_jack * rcFeat.f_osc;
    rcFeat.ix_lanealt_jack = rcFeat.f_lanealt * rcFeat.f_jack;
    rcFeat.ix_dbl_chord = rcFeat.f_dbl * rcFeat.f_chord;
    rcFeat.ix_both_notes = rcFeat.f_both * rcFeat.f_notes;
    // v10：废弃 29 项全局标量层（键型拟合产物），改由逐事件局部物理维度乘子承担。
    const rcFeatDelta = 0.0;
    // chord 项已删除（E=0 的死路径；chord = noteCount/onsetCount 是逐行离散量，若启用会引入跳变）。
        // 行速率（行/秒）：音数密度 n 之外的另一维——同一 nps 下“行更密”（和弦更小）更难。
    // 连续口径：多押中单个物件轻微偏移（远小于平均行距）不应算作两行，
    // 用高斯核把时间上几乎重合的行合并成“有效行数”(1 ≤ N_eff ≤ 行数)。
    let rowRate = 0.0;
    let rowRateDiscrete = 0.0;
    let effRows = 0.0;
    if (mergedSeq.length > 1) {
        const t0 = mergedSeq[0][0];
        const t1 = mergedSeq[mergedSeq.length - 1][0];
        const span = (t1 - t0) / 1000.0;
        if (span > 1e-3) {
            rowRateDiscrete = (mergedSeq.length - 1) / span;
            const sig = ROW_RATE_MERGE_MS;
            const s2 = 2.0 * sig * sig;
            const reach = 4.0 * sig;
            const m = mergedSeq.length;
            for (let k = 0; k < m; k += 1) {
                const tk = mergedSeq[k][0];
                let acc = 0.0;
                for (let j = k - 1; j >= 0; j -= 1) {
                    const dt = tk - mergedSeq[j][0];
                    if (dt > reach) break;
                    acc += Math.exp(-(dt * dt) / s2);
                }
                for (let j = k + 1; j < m; j += 1) {
                    const dt = mergedSeq[j][0] - tk;
                    if (dt > reach) break;
                    acc += Math.exp(-(dt * dt) / s2);
                }
                effRows += 1.0 / (1.0 + acc);
            }
            rowRate = effRows > 1.0 ? (effRows - 1.0) / span : 0.0;
        }
    }
    const rowBeta = (typeof ROW_RATE_BETA[kTotal] === 'number') ? ROW_RATE_BETA[kTotal] : 0.0;
    const rowTerm = (rowRate > 1e-3 && rowBeta !== 0.0) ? rowBeta * Math.log(rowRate / ROW_RATE_REF) : 0.0;
    const enduranceTerm = rcEndG !== 0.0 ? rcEndG * Math.log(1.0 + rcEndScore) : 0.0;
    const gPred = Math.log(n) - FIT_D_LNN * lnN + rcFeatDelta + rowTerm + enduranceTerm;
    if (typeof globalThis !== "undefined" && globalThis.__RC_DEBUG_MODE) {
        globalThis.__RC_DEBUG = {
            n, chord, lnN, noteCount, onsetCount, kTotal, laneTimes, rcFeat,
            gPred, rcFeatDelta, rowRate, rowRateDiscrete, effRows, rowTerm,
            rcEndScore, enduranceTerm, endMult,
            mergedTimes: Float64Array.from(mergedSeq, (row) => row[0]),
            mergedRaw: Float64Array.from(mergedSeq, (row) => row[1]),
            mergedCounts: Float64Array.from(mergedSeq, (row) => row[2]),
            rc10Feat,
            rcRotMult,
            mapInvertFn: mapInvert,
            mapForwardFn: mapForward,
            chord: chord,
            fitDLnn: FIT_D_LNN,
            nQuantile: QUANTILE,
        };
    }
    // 映射层输出（rcRawValue）是 4K/6K/7K 各自的映射尺度；6K/7K 再经段位对齐
    // 折线换算成「段位 = 整数」的显示尺度（4K 已是该结构，identity）。
    const rcRawValue = mapInvert(gPred);
    const rcValue = rcRawValue; // 统一映射层输出已是段位尺度，无需再对齐
    const rcDanTier = fdDanTier(kTotal);

    // ---- 难度-进度图序列（RC / LN 逐物件赋值）----
    // RC：合并序列的逐物件贡献（时间/值）。每个物件映射到 value 尺度：
    //   value = mapInvert(ln(raw) − (E·chord + D·lnN))，与 rc_value 同一映射。
    const rcValueConst = FIT_D_LNN * lnN - rcFeatDelta;
    const rcSeqTimes = [];
    const rcSeqRaw = [];
    const rcSeqValue = [];
    const rcSeqWeights = [];
    for (let rowIdx = 0; rowIdx < mergedSeq.length; rowIdx += 1) {
        const row = mergedSeq[rowIdx];
        const rowMul = (rc10Mult ? rc10Mult[rowIdx] : 1.0)
            * (rcRotMult ? rcRotMult[rowIdx] : 1.0)
            * (rcHollowMult ? rcHollowMult[rowIdx] : 1.0);
        const raw = row[1] * rowMul;
        const count = row[2];
        const val = raw > 0.0 ? mapInvert(Math.log(raw) - rcValueConst) : -60.0;
        const weight = raw > 0.0 ? 1.0 : 0.0;
        for (let k = 0; k < count; k += 1) {
            rcSeqTimes.push(row[0]);
            rcSeqRaw.push(raw);
            rcSeqValue.push(val);
            rcSeqWeights.push(weight);
        }
    }
    const rcTicks = [];
    const rcSeqValueSorted = Float64Array.from(rcSeqValue).sort();
    const rcP75 = rcSeqValueSorted.length > 0 ? quantileLinear(rcSeqValueSorted, CHART_ANCHOR_Q_RC) : rcValue;
    const rcMean = weightedMean(rcSeqValue, rcSeqWeights);
    // 先按未平移值算出 400ms 平滑曲线的采样值（只用 value），再决定锚点：
    // 默认把曲线自身的 rcCurveAnchorQuantile 分位（0.875）平移成整谱 rc_value，
    // 标线落在曲线 q90 附近，曲线整体被削减，仅顶部约 10% 的采样越过标线。
    const rcGridStart = rcSeqTimes.length > 0 ? rcSeqTimes[0] : 0.0;
    const rcGridEnd = rcSeqTimes.length > 0 ? rcSeqTimes[rcSeqTimes.length - 1] : 0.0;
    const rcCurveUnshifted = smoothAssignmentSequence(
        rcSeqTimes, rcSeqValue, rcSeqWeights, rcGridStart, rcGridEnd, null,
    );
    const rcCurveAnchorQ = Number.isFinite(CHART_CFG.rcCurveAnchorQuantile)
        ? Math.min(1.0, Math.max(0.0, CHART_CFG.rcCurveAnchorQuantile))
        : 0.875;
    const rcCurveSorted = Float64Array.from(rcCurveUnshifted.value).sort();
    const rcCurveQ = rcCurveSorted.length > 0 ? quantileLinear(rcCurveSorted, rcCurveAnchorQ) : Number.NaN;
    // 值对齐：默认把曲线分位平移成整谱 rc_value；也可在 config.js 把 anchorMode
    // 设为 "quantile" 退回旧分位对齐（逐物件分位）。
    const rcAnchorBase = CHART_ANCHOR_MODE === "mean" && Number.isFinite(rcCurveQ)
        ? rcCurveQ
        : (Number.isFinite(rcMean) ? rcMean : rcP75);
    const rcDelta = rcRawValue - rcAnchorBase;
    if (Number.isFinite(rcDelta)) {
        for (let i = 0; i < rcSeqValue.length; i += 1) {
            rcSeqValue[i] += rcDelta;
        }
    }
    if (rcDanTier) {
        // 6K/7K：刻度直接取整数段位 k，换算回映射层尺度再求 raw（与 4K 的
        // 整数 value 刻度同构；">Finish++" 等哨兵档不画刻度）。
        const tickMax = rcDanTier.names.length - 2;
        for (let k = 0; k <= tickMax; k += 1) {
            const raw = Math.exp(
                mapForward(k - rcDelta) + rcValueConst, // 统一映射层输出即段位尺度：整数 k 直接对应刻度
            );
            if (Number.isFinite(raw) && raw > 0.0) {
                rcTicks.push({ value: k, raw });
            }
        }
    } else {
        for (let v = 0; v <= 34; v += 1) {
            // 平移后的 value 尺度：整数刻度 v 对应原始 value = v - rcDelta
            const raw = Math.exp(mapForward(v - rcDelta) + rcValueConst);
            if (Number.isFinite(raw) && raw > 0.0) {
                rcTicks.push({ value: v, raw });
            }
        }
    }
    const rcAnchor = {
        value: rcValue,
        raw: Number.isFinite(rcAnchorBase) ? Math.exp(mapForward(rcAnchorBase) + rcValueConst) : Number.NaN,
        label: fixedDensityRcLabel(rcValue, kTotal),
    };
    // 平滑：400ms 窗口内所有物件（等权）的加权均值，每 100ms 采样，空窗口 = 0。
    const rcToRaw = (v) => Math.exp(mapForward(v - rcDelta) + rcValueConst);
    const rcSmoothed = {
        times: rcCurveUnshifted.times,
        value: rcCurveUnshifted.value.map((v, i) => (rcCurveUnshifted.hasData[i] ? v + rcDelta : 0.0)),
        raw: rcCurveUnshifted.value.map((v, i) => (rcCurveUnshifted.hasData[i] ? rcToRaw(v + rcDelta) : rcToRaw(0.0))),
    };
    const rcYMaxHint = sequenceYMaxHint(rcSeqValue, rcToRaw);

    // LN 通道的局部密度输入：RC 合并序列的逐物件 contribution（时间/值）。
    // 时间轴与 RC 一致（已按 1/rate 缩放）；查询时用 释手时刻 × 1/rate。
    let p2Count = 0;
    for (const row of mergedSeqP2) p2Count += row[2];
    const lnDensityTimes = new Float64Array(p2Count);
    const lnDensityValues = new Float64Array(p2Count);
    let densityIdx = 0;
    for (const row of mergedSeqP2) {
        const c = row[2];
        for (let k = 0; k < c; k += 1) {
            lnDensityTimes[densityIdx] = row[0];
            lnDensityValues[densityIdx] = row[1];
            densityIdx += 1;
        }
    }
    const lnDensity = { times: lnDensityTimes, values: lnDensityValues };

    const lnMult = lnMultiplier(rawLanes, od, rateNorm, lnDensity);
    const k = lnMult.k;
    // 统一 LN 锚（4/6/7K 同一公式，逐键常数）：nForLn = wg·n_rc + wo·occupancy + wl·n_local
    // 权重由 6K/7K/4K LN benchmark 蒸馏标定（见 params_beta_1.1.3.json 的 ln.unified_local_anchor）。
    const LN_ANCHOR_W = (kTotal === 7)
        ? { wg: 0.25, wo: 1.00, wl: 0.50 }
        : ((kTotal === 6) ? { wg: 0.75, wo: 2.00, wl: 0.25 } : { wg: 0.75, wo: 0.50, wl: 0.25 });
    const lnRcW = v11Cfg("lnRcW", LN_ANCHOR_W.wg);
    let nForLn = nNoRot;
    const lnAnchorP2 = v11Cfg(`lnAnchorP2${kTotal}`, v11Cfg("lnAnchorP2", -1));
    if (lnAnchorP2 === 0) nForLn = nNoRot; // 强制使用 RC 均值合并密度（忽略 2-范数）
    if (mergedSeqP2 !== mergedSeq) {
        // 旧口径（2-范数手内合并）的密度分位数，保证 6K/7K LN 与 1.0.7 逐位一致
        const p2Values = new Float64Array(p2Count);
        let pi = 0;
        for (let rowIdx = 0; rowIdx < mergedSeqP2.length; rowIdx += 1) {
            const row = mergedSeqP2[rowIdx];
            const m = rc10Mult ? rc10Mult[rowIdx] : 1.0;
            for (let j = 0; j < row[2]; j += 1) p2Values[pi++] = row[1] * m;
        }
        const nP2 = findNQuantile(p2Values, QUANTILE);
        if (Number.isFinite(nP2) && nP2 > 0.0 && lnAnchorP2 !== 0) nForLn = nP2;
    }
    if (LN_OWN_N_MODE !== 0) {
        const nLn = lnOwnQuantileN(rawLanes, od, 1.0 / rateNorm, lnDensity);
        if (Number.isFinite(nLn) && nLn > 0.0) {
            nForLn = nLn;
        }
    }
    // LN 密度锚定修正（实验，默认关闭）：RC 密度只反映"按下"的密度，LN 长条占用并不进入，
    // 导致「少按键、多长条、多叠持」的 LN 谱被系统性低估。用平均同时保持数（occupancy）
    // 作为 LN 自身负载补进密度锚。
    // 仅 6K/7K（4K LN 已经标定好，不动）；实验时用 __V11.lnOccK=0 放开到 4K。
    const lnOccW = v11Cfg("lnOcc", LN_ANCHOR_W.wo);
    const lnSat = (x, s) => (s > 0.0 ? x / (1.0 + x / s) : x);
    const lnRcSat = v11Cfg("lnRcSat", 0.0);
    // LN Speed（短面密集）时 RC 密度与 LN 局部密度重复计数 → 按短面占比衰减 RC 侧锚。
    const lnRcShort = v11Cfg(`lnRcShort${kTotal}`, v11Cfg("lnRcShort", 0.0));
    let lnShortShareChart = 0.0;
    {
        let shortCnt = 0.0;
        let holdCnt = 0.0;
        for (const lane of rawLanes) {
            for (const n of lane) {
                if (n.len > 0) {
                    holdCnt += 1.0;
                    if (n.len < 200.0) shortCnt += 1.0;
                }
            }
        }
        if (holdCnt > 0.0) lnShortShareChart = shortCnt / holdCnt;
    }
    let shortShareAnchor = 0.0;
    if (lnRcShort !== 0.0) {
        let shortCnt = 0.0;
        let holdCnt = 0.0;
        for (const lane of rawLanes) {
            for (const n of lane) {
                if (n.len > 0) {
                    holdCnt += 1.0;
                    if (n.len < 200.0) shortCnt += 1.0;
                }
            }
        }
        if (holdCnt > 0.0) shortShareAnchor = shortCnt / holdCnt;
    }
    const rcShortGate = lnRcShort !== 0.0
        ? 1.0 - lnRcShort / (1.0 + Math.exp(-(shortShareAnchor - 0.55) / 0.12))
        : 1.0;
    const lnAnchorMode = v11Cfg(`lnAnchorMode${kTotal}`, v11Cfg("lnAnchorMode", 0));
    const lnCsW = v11Cfg("lnCsW", 1.0);
    const lnCsNeed = lnAnchorMode !== 0
        || v11Cfg("lnCsAdd", 0.0) !== 0.0
        || v11Cfg(`lnCsAdd${kTotal}`, LN_CS_ADD_BY_KEY[kTotal] || 0.0) !== 0.0
        || v11Cfg("lnCsMix", 0.0) !== 0.0
        || v11Cfg(`lnCsMix${kTotal}`, LN_CS_MIX_BY_KEY[kTotal] || 0.0) !== 0.0;
    const lnCs = lnCsNeed ? lnChordstreamMetrics(rawLanes) : null;
    let nRcPart = lnSat(nForLn * lnRcW, lnRcSat) * rcShortGate;
    if (lnCs && (lnAnchorMode === 1 || lnAnchorMode === 3)) {
        nRcPart = lnCs.csDensity * lnCsW;
    }
    let lnOccupancy = 0.0;
    if (lnOccW > 0.0) {
        let holdMs = 0.0;
        let tMin = Infinity;
        let tMax = -Infinity;
        for (const lane of rawLanes) {
            for (const n of lane) {
                const len = n.len > 0 ? n.len : 0.0;
                holdMs += len;
                if (n.t < tMin) tMin = n.t;
                const end = n.t + len;
                if (end > tMax) tMax = end;
            }
        }
        const span = tMax - tMin;
        if (span > TIME_EPS) {
            lnOccupancy = holdMs / span;
            nForLn = nRcPart + lnOccW * lnOccupancy;
        }
    }
    if (lnOccW <= 0.0) {
        nForLn = nRcPart;
    }
    // LN 局部密度锚（统一公式）：只在 LN 事件位置采样 ±lnLocWin ms 内的音符密度，
    // 非 LN 段的 RC/音符变化不会进入该项。
    const lnLocW = v11Cfg("lnLocW", LN_ANCHOR_W.wl);
    const lnLocWin = v11Cfg(`lnLocWin${kTotal}`, v11Cfg("lnLocWin", 500.0));
    let nLocal = 0.0;
    if (lnLocW > 0.0) {
        const lnEnds = [];
        const allTimes = [];
        const useCsLocal = Boolean(lnCs) && (lnAnchorMode === 2 || lnAnchorMode === 3);
        const lnLocMinLen = v11Cfg("lnLocMinLen", 0.0);
        for (const lane of rawLanes) {
            for (const n of lane) {
                allTimes.push(n.t);
                if (n.len > 0 && n.len >= lnLocMinLen) lnEnds.push(n.t + n.len);
            }
        }
        if (useCsLocal && lnCs && lnCs.csTimes.length > 0) {
            allTimes.length = 0;
            for (const t of lnCs.csTimes) allTimes.push(t);
        }
        if (lnEnds.length > 0 && allTimes.length > 0) {
            allTimes.sort((a, b) => a - b);
            let acc = 0.0;
            for (const te of lnEnds) {
                let lo = 0;
                let hi = allTimes.length;
                const targetA = te - lnLocWin;
                while (lo < hi) {
                    const mid = (lo + hi) >> 1;
                    if (allTimes[mid] < targetA) lo = mid + 1;
                    else hi = mid;
                }
                let cnt = 0;
                for (let j = lo; j < allTimes.length && allTimes[j] <= te + lnLocWin; j += 1) {
                    cnt += 1;
                }
                acc += cnt;
            }
            nLocal = acc / lnEnds.length / (2.0 * lnLocWin / 1000.0);
            // 实验：局部锚中混入「Chordstream 局部密度」的比例（默认 0）
            const lnCsMix = v11Cfg(`lnCsMix${kTotal}`, v11Cfg("lnCsMix", LN_CS_MIX_BY_KEY[kTotal] ?? 0.0));
            if (lnCsMix > 0.0 && lnCs && lnCs.csTimes.length > 0) {
                let accCs = 0.0;
                for (const te of lnEnds) {
                    let lo2 = 0;
                    let hi2 = lnCs.csTimes.length;
                    const ta = te - lnLocWin;
                    while (lo2 < hi2) {
                        const mid = (lo2 + hi2) >> 1;
                        if (lnCs.csTimes[mid] < ta) lo2 = mid + 1;
                        else hi2 = mid;
                    }
                    let c2 = 0;
                    for (let j = lo2; j < lnCs.csTimes.length && lnCs.csTimes[j] <= te + lnLocWin; j += 1) c2 += 1;
                    accCs += c2;
                }
                const nLocalCs = accCs / lnEnds.length / (2.0 * lnLocWin / 1000.0);
                nLocal = (1.0 - lnCsMix) * nLocal + lnCsMix * nLocalCs;
            }
            // 短面密集（LN Speed）时局部密度与 RC 密度重复计数 → 按短面占比衰减 lnLoc。
            const LN_LOC_SHORT = (kTotal === 7) ? 0.15 : ((kTotal === 6) ? 0.40 : 0.40);
            const lnLocShort = v11Cfg(`lnLocShort${kTotal}`, v11Cfg("lnLocShort", LN_LOC_SHORT));
            if (lnLocShort !== 0.0) {
                let shortCnt = 0.0;
                let holdCnt = 0.0;
                for (const lane of rawLanes) {
                    for (const n of lane) {
                        if (n.len > 0) {
                            holdCnt += 1.0;
                            if (n.len < 200.0) shortCnt += 1.0;
                        }
                    }
                }
                if (holdCnt > 0.0) {
                    const shortShare = shortCnt / holdCnt;
                    const gate = 1.0 / (1.0 + Math.exp(-(shortShare - 0.55) / 0.12));
                    nLocal *= (1.0 - lnLocShort * gate);
                }
            }
            const lnLocSat = v11Cfg("lnLocSat", 0.0);
            nForLn = nForLn + lnLocW * lnSat(nLocal, lnLocSat);
        }
    }
    // 实验：独立「Chordstream 关注」RC 加性项（默认 0）
    const lnCsAdd = v11Cfg(`lnCsAdd${kTotal}`, v11Cfg("lnCsAdd", LN_CS_ADD_BY_KEY[kTotal] ?? 0.0));
    if (lnCsAdd !== 0.0 && lnCs) {
        nForLn = nForLn + lnCsAdd * lnCs.csDensity;
    }
    // 锚层交叉项（实验，默认 0）：密度高但保持弱（sHold 低）→ 下调锚；松手结构重（sRelease 高）→ 上调锚。
    const LN_ANCHOR_MX = (kTotal === 6) ? 0.40 : ((kTotal === 4) ? 0.65 : 0.0);
    const lnAnchorMx = v11Cfg(`lnAnchorMx${kTotal}`, v11Cfg("lnAnchorMx", LN_ANCHOR_MX));
    const lnAnchorRel = v11Cfg(`lnAnchorRel${kTotal}`, v11Cfg("lnAnchorRel", 0.0));
    if (lnAnchorMx !== 0.0 || lnAnchorRel !== 0.0) {
        const sHoldV = (lnMult.parts && lnMult.parts.sHold) || 0.0;
        const sReleaseV = (lnMult.parts && lnMult.parts.sRelease) || 0.0;
        if (lnAnchorMx !== 0.0) {
            nForLn *= Math.max(0.4, 1.0 - lnAnchorMx * (1.0 - Math.min(1.0, sHoldV / 0.5)));
        }
        if (lnAnchorRel !== 0.0) {
            nForLn *= (1.0 + lnAnchorRel * sReleaseV);
        }
    }
    // LN 也与 RC 用同一段位对齐折线（6K/7K 无独立 LN 段位参考，两通道数值尺度接近），
    // 使 LN 胶囊的数字同样等于段位序号；4K 保持原 ln 尺度。
    const lnCShift = v11Cfg("lnCShift", 0.0);
    // 线性 + 倍率联合：u = ln(1 + LN_B·(k·n + b))，b 为逐键常数项（默认 0）。
    const lnAddB = v11Cfg("lnAddB", LN_ADD_B[kTotal] || 0.0);
    // 释放精度通道：与密度无关的独立难度，直接叠加在指数域（u 里），随密度自然淡出。
    const lnShieldHeadW = v11Cfg("lnShieldHead", LN_SHIELD_HEAD_W[kTotal] || 0.0);
    const lnReleaseW = v11Cfg("lnRelease", LN_RELEASE_W[kTotal] || 0.0);
    const sShieldHeadValue = (lnMult.parts && lnMult.parts.sShieldHead) || 0.0;
    const sReleaseValue = (lnMult.parts && lnMult.parts.sRelease) || 0.0;
    const lnShortLin = v11Cfg(`lnShortLin${kTotal}`, v11Cfg("lnShortLin", 0.0));
    const lnShortLinTerm = lnShortLin * (lnShortShareChart - 0.5);
    const lnAddExtra = LN_B * (lnAddB + lnShieldHeadW * sShieldHeadValue + lnReleaseW * sReleaseValue + lnShortLinTerm);
    const lnRawValue = lnFromRawMapped(k, nForLn, lnAddExtra) + lnCShift;
    const lnShift = fdDanTier(kTotal)?.lnShift || 0.0;
    const lnValue = Math.max(0.0, fixedDensityLnAlign(lnRawValue, kTotal) + lnShift);
    if (typeof globalThis !== "undefined" && globalThis.__RC_DEBUG_MODE) {
        globalThis.__RC_DEBUG_LN = {
            lnU: (LN_B * k * nForLn > 0.0) ? Math.log1p(LN_B * k * nForLn) : 0.0,
            lnUEff: (LN_B * k * nForLn + lnAddExtra > 0.0) ? Math.log1p(LN_B * k * nForLn + lnAddExtra) : 0.0,
            lnK: k, nForLn, nLocal, lnLocW, lnOccupancy, nRcPart, kTotal, lnRawValue, lnValue,
            lnAbsorb: lnMult.absorb, lnValid: lnMult.valid,
            lnParts: lnMult.parts || null,
        };
    }

    // LN 逐松手事件的赋值：γ 加权结构量 × F(ρ)（与 k 的组件一致），再映射到
    // ln_value 尺度：value = A·ln(1 + B·raw·n)，与图上整数 ln 标注线互为反函数。
    const lnSeqTimes = [];
    const lnSeqValue = [];
    const lnSeqWeights = [];
    if (lnMult.events.length > 0) {
        const factor = lnMult.factor;
        const absorb = lnMult.absorb;
        const valid = lnMult.valid;
        // 逐事件未缩放赋值 u_i 与整谱 k 的比值：把 u_i 缩放到「其聚合分位 = k」的尺度，
        // 使图上逐事件 value 的分位与 ln_value 对齐（否则逐事件求和普遍高于聚合值）。
        const uArr = new Float64Array(lnMult.events.length);
        for (let i = 0; i < lnMult.events.length; i += 1) {
            const ev = lnMult.events[i];
            uArr[i] = factor * (
                LN_MULT_REL * ev.rel * absorb * valid * rateNorm
                + LN_MULT_JACK * ev.jack * absorb * valid * rateNorm
                + LN_MULT_ADJ * ev.adj * absorb * valid * rateNorm
                + LN_MULT_CROSS * ev.cross * valid
                + LN_MULT_HOLD * ev.hold * valid
            );
        }
        const uAgg = lnAggregate(uArr);
        const scale = (k > 0.0 && uAgg > 0.0) ? k / uAgg : 1.0;
        for (let i = 0; i < lnMult.events.length; i += 1) {
            const raw = uArr[i] * scale;
            const val = lnFromRawMapped(raw, nForLn, lnAddExtra);
            lnSeqTimes.push(lnMult.events[i].t * invRate);
            lnSeqValue.push(val);
            lnSeqWeights.push(Number.isFinite(lnMult.events[i].w) && lnMult.events[i].w > 0.0 ? lnMult.events[i].w : 1.0);
        }
    }
    const lnTicks = [];
    const lnAnchor = { value: lnValue, raw: Number.NaN, label: fixedDensityLnLabel(lnValue, kTotal) };
    let lnDelta = 0.0;
    if (lnSeqValue.length > 0) {
        const lnSeqValueSorted = Float64Array.from(lnSeqValue).sort();
        const lnP75 = quantileLinear(lnSeqValueSorted, CHART_ANCHOR_Q_LN);
        const lnMean = weightedMean(lnSeqValue, lnSeqWeights);
        const lnAnchorBase = CHART_ANCHOR_MODE === "mean" && Number.isFinite(lnMean) ? lnMean : lnP75;
        lnDelta = lnRawValue - lnAnchorBase;
        if (Number.isFinite(lnDelta)) {
            for (let i = 0; i < lnSeqValue.length; i += 1) {
                lnSeqValue[i] += lnDelta;
            }
        }
        if (nForLn > 0.0 && Number.isFinite(lnAnchorBase)) {
            lnAnchor.raw = lnRawFromValue(lnAnchorBase, nForLn, lnAddExtra);
        }
        if (nForLn > 0.0) {
            if (rcDanTier) {
                // 6K/7K：刻度取整数段位，换算回 LN 映射尺度再求 raw。
                const tickMax = rcDanTier.names.length - 1;
                const tickShift = rcDanTier.lnShift || 0;
                for (let k = tickShift; k <= tickMax; k += 1) {
                    const raw = lnRawFromValue(fixedDensityLnUnalign(k - tickShift, kTotal) - lnDelta, nForLn, lnAddExtra);
                    if (Number.isFinite(raw) && raw > 0.0) {
                        lnTicks.push({ value: k, raw });
                    }
                }
            } else {
                for (let v = 0; v <= 40; v += 1) {
                    const raw = lnRawFromValue(v - lnDelta, nForLn, lnAddExtra);
                    if (Number.isFinite(raw) && raw > 0.0) {
                        lnTicks.push({ value: v, raw });
                    }
                }
            }
        }
    }

    // LN 曲线的平滑：窗口内只统计 LN 松手事件（按长度权重 w 加权），
    // 每 100ms 采样、400ms 窗口；窗口内没有 LN 事件时该点直接为 0。
    const lnEventCount = lnSeqTimes.length;
    const lnToRaw = nForLn > 0.0
        ? (v) => lnRawFromValue(v - lnDelta, nForLn, lnAddExtra)
        : null;
    const lnYMaxHint = sequenceYMaxHint(lnSeqValue, lnToRaw);
    const gridT0 = rcSeqTimes.length > 0 ? rcSeqTimes[0] : (lnSeqTimes.length > 0 ? lnSeqTimes[0] : 0.0);
    const gridT1 = rcSeqTimes.length > 0
        ? rcSeqTimes[rcSeqTimes.length - 1]
        : (lnSeqTimes.length > 0 ? lnSeqTimes[lnSeqTimes.length - 1] : 0.0);
    const lnSmoothed = (lnEventCount > 0 && lnToRaw)
        ? smoothAssignmentSequence(lnSeqTimes, lnSeqValue, lnSeqWeights, gridT0, gridT1, lnToRaw,
            CHART_LN_SMOOTH_WINDOW_MS, CHART_LN_SMOOTH_STEP_MS)
        : { times: [], value: [], raw: [] };

    const __debug = (options && options.debug)
        ? {
            keyCount: kTotal,
            hands,
            rcMapProfile: kTotal === 6 ? "6k" : (kTotal === 7 ? "7k" : "4k"),
            totalCount,
            n,
            chord,
            noteCount,
            lnN,
            gPred,
            k,
            nForLn,
            lnRaw1p: Math.log1p(LN_B * k * nForLn),
            rcValue,
            lnValue,
        }
        : null;
    return {
        ...(__debug ? { __debug } : {}),
        rcValue,
        lnValue,
        rcSequence: { times: rcSmoothed.times, raw: rcSmoothed.raw, value: rcSmoothed.value, yMax: rcYMaxHint, smoothed: true },
        lnSequence: { times: lnSmoothed.times, raw: lnSmoothed.raw, value: lnSmoothed.value, eventCount: lnEventCount, yMax: lnYMaxHint, smoothed: true },
        rcTicks,
        lnTicks,
        rcAnchor,
        lnAnchor,
        overlaps,
    };
}

export default calculateFixedDensity;

// ---- 数字上方的文字说明：段位名 + 分段名 ----

// 希腊字母表小写：11→α、12→β … 34→ω
const FD_GREEK_LOWER = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ",
    "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
];

// 分段名：按小数部分以 .1/.3/.5/.7/.9 分段
// [.1,.3)→"+"、[.3,.5)→"++"、[.5,.7)→"--"、[.7,.9)→"-"、其余（[.9,1)∪[0,.1)）→""
function fixedDensitySegment(value) {
    const frac = value - Math.floor(value);
    if (frac < 0.1) {
        return "";
    }
    if (frac < 0.3) {
        return "+";
    }
    if (frac < 0.5) {
        return "++";
    }
    if (frac < 0.7) {
        return "--";
    }
    if (frac < 0.9) {
        return "-";
    }
    return "";
}

// 6K/7K 段位表：数字胶囊上方的注释显示真实段位名（4K 仍用 rf/希腊字母），
// 且数字本身就是段位序号（12 → Mystery / Azimuth），结构与 4K 一致：
//   段位 = Math.round(值)，段内后缀复用 4K 的 fixedDensitySegment 五段规则。
// alignX = benchmark-6krc / benchmark-7krc 各档参考谱 rc 中位数的逐档中位
// （第 i 档中位数 → 整数 i），alignY = 段位序号；fixedDensityDanAlign /
// fixedDensityDanUnalign 是这条分段折线的互逆映射（端部按端点斜率外推），
// 把映射层输出换算到「段位 = 整数」的尺度，刻度再换算回映射层尺度取 raw。
export const FD_DAN_TIERS = {
    6: {
        names: ["Start", "reg1", "reg2", "reg3", "reg4", "reg5", "reg6", "reg7", "reg8", "reg9",
            "Terra", "Celestial", "Mystery", "Nihility", "Finish", ">Finish++"],
        alignX: [0.12987, 1.08652, 1.89934, 3.12732, 3.89575, 4.94815, 6.22813, 6.88041, 8.02569, 8.92407, 9.95625],
        alignY: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        // LN 段位对齐：线性，锚点 = 零 LN 谱的映射下限（0.47022 → 0）与 14 档
        // LN 参考谱中位（15.298/16.381 → 14）。6K/7K 的 lnValue 与段位近似线性，
        // 逐档中位数在低档不单调（每档仅 4 首），故不用逐档折线。
        lnAlignA: 2.844069,
        lnAlignB: 1.145096,
        lnShift: 1.0,
    },
    7: {
        names: ["reg0", "reg1", "reg2", "reg3", "reg4", "reg5", "reg6", "reg7", "reg8", "reg9", "reg10",
            "Gamma", "Azimuth", "Zenith", "Stellium", ">Stellium++"],
        alignX: [0.10859, 0.86939, 2.33368, 3.35119, 3.80531, 5.27076, 6.55909, 6.90430, 8.19752, 8.46649, 9.00737, 10.85177, 11.66500, 13.25279, 14.41561],
        alignY: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        lnAlignA: 1.791843,
        lnAlignB: 1.034750,
        lnShift: 1.0,
    },
};

function fdDanTier(keyCount) {
    return FD_DAN_TIERS[Number(keyCount)] || null;
}

// 分段线性：映射层尺度 → 段位尺度（第 i 档中位数恰好等于整数 i）。
export function fixedDensityDanAlign(value, keyCount) {
    const tier = fdDanTier(keyCount);
    if (!tier || !Number.isFinite(value)) {
        return value;
    }
    const xs = tier.alignX;
    const ys = tier.alignY;
    const n = Math.min(xs.length, ys.length);
    if (n < 2) {
        return value;
    }
    if (value <= xs[0]) {
        return ys[0] + (value - xs[0]) * ((ys[1] - ys[0]) / (xs[1] - xs[0]));
    }
    if (value >= xs[n - 1]) {
        return ys[n - 1] + (value - xs[n - 1]) * ((ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]));
    }
    let i = 0;
    while (i < n - 2 && value >= xs[i + 1]) {
        i += 1;
    }
    const t = (value - xs[i]) / (xs[i + 1] - xs[i]);
    return ys[i] + t * (ys[i + 1] - ys[i]);
}

// 逆映射：段位尺度 → 映射层尺度（图表的整数段位刻度换算回 raw 时用）。
export function fixedDensityDanUnalign(value, keyCount) {
    const tier = fdDanTier(keyCount);
    if (!tier || !Number.isFinite(value)) {
        return value;
    }
    const xs = tier.alignX;
    const ys = tier.alignY;
    const n = Math.min(xs.length, ys.length);
    if (n < 2) {
        return value;
    }
    if (value <= ys[0]) {
        return xs[0] + (value - ys[0]) * ((xs[1] - xs[0]) / (ys[1] - ys[0]));
    }
    if (value >= ys[n - 1]) {
        return xs[n - 1] + (value - ys[n - 1]) * ((xs[n - 1] - xs[n - 2]) / (ys[n - 1] - ys[n - 2]));
    }
    let i = 0;
    while (i < n - 2 && value >= ys[i + 1]) {
        i += 1;
    }
    const t = (value - ys[i]) / (ys[i + 1] - ys[i]);
    return xs[i] + t * (xs[i + 1] - xs[i]);
}

// LN 的段位对齐：线性（ln ≈ a + b·段位），与 RC 的逐档折线各自独立标定。
// 映射层对「几乎无 LN」的谱会给 0~0.47 的下限值（与谱面 n 有关），对齐后可能略为负；
// LN 不存在「低于 Start」的真实含义，故下限截到 0（显示 0.00 / Start / reg0）。
export function fixedDensityLnAlign(value, keyCount) {
    const tier = fdDanTier(keyCount);
    if (!tier || !Number.isFinite(value) || !(tier.lnAlignB > 0)) {
        return value;
    }
    // 不在这里截 0：低于 Start 的谱需要保留负值，最终显示值由调用处 clamp 到 0。
    return (value - tier.lnAlignA) / tier.lnAlignB;
}

export function fixedDensityLnUnalign(value, keyCount) {
    const tier = fdDanTier(keyCount);
    if (!tier || !Number.isFinite(value) || !(tier.lnAlignB > 0)) {
        return value;
    }
    return tier.lnAlignA + tier.lnAlignB * value;
}

// 段内后缀：与 4K 的 fixedDensitySegment 同义（按与最近整数的偏移给 +/++/--/-）。
function fixedDensityDanSuffix(offset) {
    const a = Math.abs(offset);
    if (!Number.isFinite(a) || a < 0.1) {
        return "";
    }
    if (a < 0.3) {
        return offset > 0 ? "+" : "-";
    }
    return offset > 0 ? "++" : "--";
}

// 段位序号：四舍五入到最近整数档；负值归首档，≥末档（">..." 档）直接返回末档。
function fixedDensityDanIndex(value, tier) {
    const rounded = Math.round(value);
    if (!Number.isFinite(rounded)) {
        return -1;
    }
    const last = tier.names.length - 1;
    if (rounded >= last) {
        return last;
    }
    return rounded < 0 ? 0 : rounded;
}

// 图表刻度只看段位名（4K 刻度的 rfN 同样不带后缀）；胶囊注释才带段内后缀。
export function fixedDensityDanTickLabel(value, keyCount) {
    const tier = fdDanTier(keyCount);
    if (!tier || !Number.isFinite(value)) {
        return "";
    }
    const i = fixedDensityDanIndex(value, tier);
    return i < 0 ? "" : tier.names[i];
}

function fixedDensityDanLabel(value, tier) {
    const last = tier.names.length - 1;
    const i = fixedDensityDanIndex(value, tier);
    if (i < 0) {
        return "";
    }
    if (i >= last) {
        return tier.names[last];
    }
    // 低于首档（负值）用 value 自身作偏移，其余用与最近整数的偏移。
    const offset = value < 0 ? value : value - Math.round(value);
    return tier.names[i] + fixedDensityDanSuffix(offset);
}

// rc 段位名：≤10 → "rf<k>"；≥11 → 希腊小写（11→α、14→δ …）；≥34.5 → 直接 ">ω++"。
export function fixedDensityRcLabel(value, keyCount) {
    if (!Number.isFinite(value)) {
        return "";
    }
    const danTier = FD_DAN_TIERS[Number(keyCount)];
    if (danTier) {
        return fixedDensityDanLabel(value, danTier);
    }
    const rounded = Math.round(value);
    if (value >= 34.5 || rounded - 11 >= FD_GREEK_LOWER.length) {
        return ">ω++";
    }
    const tier = rounded <= 10 ? `rf${rounded}` : FD_GREEK_LOWER[rounded - 11];
    return tier + fixedDensitySegment(value);
}

// ln 段位名："ln<k>"。
export function fixedDensityLnLabel(value, keyCount) {
    if (!Number.isFinite(value)) {
        return "";
    }
    const danTier = FD_DAN_TIERS[Number(keyCount)];
    if (danTier) {
        return fixedDensityDanLabel(value - (danTier.lnShift || 0), danTier);
    }
    return `ln${Math.round(value)}${fixedDensitySegment(value)}`;
}
