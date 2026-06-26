#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace on_top_human_filter {

struct HumanBBox {
    // Equivalent to Python item being an iterable/dict box: [x1, y1, x2, y2]
    // For dict inputs, set class_id from "class_id" or "cls". Default Python class_id is 1.
    std::vector<double> box;
    int class_id = 1;
};

struct TruckMask {
    // Equivalent to Python truck mask item, or dict {"mask": mask, "conf": conf}
    cv::Mat mask;
    double conf = 1.0;
};

struct BoxXYXY {
    int x1 = 0;
    int y1 = 0;
    int x2 = -1;
    int y2 = -1;

    bool valid() const {
        return x2 >= x1 && y2 >= y1;
    }
};

struct ModeConfig {
    double MIN_ASSOC_OVERLAP_PERCENT = 0.0;
    double FULL_OVERLAP_PERCENT_STRICT = 0.0;
    double LOWER_OVERLAP_PERCENT_RELAXED = 0.0;
    double BOTTOM_SUPPORT_PERCENT_MIN = 0.0;
    double BOTTOM_DOMINANCE_RATIO = 0.0;
    double SIDE_OVERLAP_PERCENT_MAX_STRICT = 0.0;
    std::string LOW_OVERLAP_REJECT_MODE;
    bool ERODE_TRUCK_MASK = false;
    int ERODE_KERNEL_SIZE = 3;
    int ERODE_ITERATIONS = 1;
};

struct NormalizedTruck {
    int truck_index = -1;
    cv::Mat mask;
    double conf = 1.0;
    int class_id = 0;
    std::string class_name = "truck";
};

struct RegionOverlapResult {
    double percent = 0.0;
    int pixels = 0;
    int area = 0;
};

struct EvalInfo {
    std::string reject_mode = "ok";
    double assoc_overlap_percent = 0.0;
    double full_overlap_percent = 0.0;
    double bottom_contact_percent = 0.0;
    double lower_overlap_percent = 0.0;
    double side_overlap_percent = 0.0;
    int assoc_overlap_pixels = 0;
    int full_overlap_pixels = 0;
    int bottom_contact_pixels = 0;
    int lower_overlap_pixels = 0;
    int side_overlap_pixels = 0;
    int assoc_area = 0;
    int full_area = 0;
    int bottom_area = 0;
    int lower_area = 0;
    int side_area = 0;
    std::optional<BoxXYXY> bottom_band_box;
    std::optional<BoxXYXY> lower_body_box;
    std::optional<BoxXYXY> left_side_box;
    std::optional<BoxXYXY> right_side_box;
};

struct CandidateTruck {
    int truck_idx = -1;
    cv::Mat truck_mask;
    double truck_conf = 1.0;
    double assoc_overlap_percent = 0.0;
    int assoc_overlap_pixels = 0;
    int assoc_area = 0;
};

struct EvaluatedTruck {
    int truck_idx = -1;
    cv::Mat truck_mask;
    bool ok = false;
    EvalInfo info;
    double score = 0.0;
};

// Shared constants, copied from the Python implementation.
static constexpr int BOTTOM_STRIP_HEIGHT = 10;
static constexpr int BOTTOM_STRIP_OFFSET_UP = 10;
static constexpr double LOWER_BODY_FRAC = 0.50;
static constexpr double SIDE_STRIP_WIDTH_FRAC = 0.18;
static constexpr double CENTER_WIDTH_FRAC = 0.40;
static constexpr bool USE_CENTER_BOTTOM_BAND = true;

static constexpr int HOOK_CLASS_ID = 2;
static constexpr double HOOK_TOP_CROP_FRAC = 0.32;
static constexpr double HOOK_MIN_KEEP_FRAC = 0.50;

static const std::unordered_map<std::string, ModeConfig>& modeConfigs() {
    static const std::unordered_map<std::string, ModeConfig> configs = {
        {
            "relaxed",
            ModeConfig{
                2.0,    // MIN_ASSOC_OVERLAP_PERCENT
                6.0,    // FULL_OVERLAP_PERCENT_STRICT
                3.0,    // LOWER_OVERLAP_PERCENT_RELAXED
                2.5,    // BOTTOM_SUPPORT_PERCENT_MIN
                1.20,   // BOTTOM_DOMINANCE_RATIO
                45.0,   // SIDE_OVERLAP_PERCENT_MAX_STRICT
                "low_rel_overlap",
                false,  // ERODE_TRUCK_MASK
                3,      // ERODE_KERNEL_SIZE
                1       // ERODE_ITERATIONS
            }
        },
        {
            "restricted",
            ModeConfig{
                4.0,    // MIN_ASSOC_OVERLAP_PERCENT
                24.0,   // FULL_OVERLAP_PERCENT_STRICT
                18.0,   // LOWER_OVERLAP_PERCENT_RELAXED
                18.0,   // BOTTOM_SUPPORT_PERCENT_MIN
                5.0,    // BOTTOM_DOMINANCE_RATIO
                6.0,    // SIDE_OVERLAP_PERCENT_MAX_STRICT
                "low_percent_overlap",
                true,   // ERODE_TRUCK_MASK
                5,      // ERODE_KERNEL_SIZE
                1       // ERODE_ITERATIONS
            }
        }
    };
    return configs;
}

static int pythonRoundToInt(double value) {
    if (!std::isfinite(value)) {
        return 0;
    }

    // Python round() uses ties-to-even. std::round() does not.
    const double lower = std::floor(value);
    const double upper = std::ceil(value);
    const double dl = std::abs(value - lower);
    const double du = std::abs(upper - value);

    double result = lower;
    if (dl < du) {
        result = lower;
    } else if (du < dl) {
        result = upper;
    } else {
        const long long li = static_cast<long long>(lower);
        result = (std::llabs(li) % 2 == 0) ? lower : upper;
    }

    if (result > static_cast<double>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    if (result < static_cast<double>(std::numeric_limits<int>::min())) {
        return std::numeric_limits<int>::min();
    }
    return static_cast<int>(result);
}

static int clipInt(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

static BoxXYXY invalidBox() {
    return BoxXYXY{0, 0, -1, -1};
}

static BoxXYXY clipBoxXYXY(const std::vector<double>& box, int hh, int ww) {
    if (hh <= 0 || ww <= 0 || box.size() < 4) {
        return invalidBox();
    }

    try {
        if (!std::isfinite(box[0]) || !std::isfinite(box[1]) ||
            !std::isfinite(box[2]) || !std::isfinite(box[3])) {
            return invalidBox();
        }

        const int x1 = clipInt(pythonRoundToInt(static_cast<double>(box[0])), 0, ww - 1);
        const int y1 = clipInt(pythonRoundToInt(static_cast<double>(box[1])), 0, hh - 1);
        const int x2 = clipInt(pythonRoundToInt(static_cast<double>(box[2])), 0, ww - 1);
        const int y2 = clipInt(pythonRoundToInt(static_cast<double>(box[3])), 0, hh - 1);
        return BoxXYXY{x1, y1, x2, y2};
    } catch (...) {
        return invalidBox();
    }
}

static BoxXYXY clipBoxXYXY(const BoxXYXY& box, int hh, int ww) {
    if (hh <= 0 || ww <= 0) {
        return invalidBox();
    }

    try {
        const int x1 = clipInt(box.x1, 0, ww - 1);
        const int y1 = clipInt(box.y1, 0, hh - 1);
        const int x2 = clipInt(box.x2, 0, ww - 1);
        const int y2 = clipInt(box.y2, 0, hh - 1);
        return BoxXYXY{x1, y1, x2, y2};
    } catch (...) {
        return invalidBox();
    }
}

static int boxAreaXYXY(const BoxXYXY& box, int hh, int ww) {
    const BoxXYXY clipped = clipBoxXYXY(box, hh, ww);
    if (!clipped.valid()) {
        return 0;
    }
    return (clipped.x2 - clipped.x1 + 1) * (clipped.y2 - clipped.y1 + 1);
}

static cv::Mat erodeMask(const cv::Mat& mask01, int kernel_size, int iterations) {
    try {
        if (mask01.empty()) {
            return cv::Mat();
        }

        if (kernel_size <= 1 || iterations <= 0) {
            return mask01.clone();
        }

        const cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8U);
        cv::Mat eroded;
        cv::erode(mask01, eroded, kernel, cv::Point(-1, -1), iterations);

        // If erosion wipes out the mask completely, keep the original.
        if (eroded.empty() || cv::countNonZero(eroded) == 0) {
            return mask01.clone();
        }

        return eroded;
    } catch (...) {
        return mask01.empty() ? cv::Mat() : mask01.clone();
    }
}

static BoxXYXY makeLogicBox(const BoxXYXY& raw_box, int class_id, int hh, int ww) {
    const BoxXYXY clipped = clipBoxXYXY(raw_box, hh, ww);

    if (!clipped.valid()) {
        return invalidBox();
    }

    if (class_id != HOOK_CLASS_ID) {
        return clipped;
    }

    const int box_h = std::max(1, clipped.y2 - clipped.y1 + 1);
    const int crop_top = pythonRoundToInt(static_cast<double>(box_h) * HOOK_TOP_CROP_FRAC);
    const int keep_min = pythonRoundToInt(static_cast<double>(box_h) * HOOK_MIN_KEEP_FRAC);

    int ny1 = std::min(clipped.y2, clipped.y1 + crop_top);
    const int kept_h = clipped.y2 - ny1 + 1;

    if (kept_h < keep_min) {
        ny1 = std::max(clipped.y1, clipped.y2 - keep_min + 1);
    }

    return BoxXYXY{clipped.x1, ny1, clipped.x2, clipped.y2};
}

static int intersectionPixelsBoxMask(const BoxXYXY& box, const cv::Mat& truck_mask) {
    if (truck_mask.empty()) {
        return 0;
    }

    const BoxXYXY clipped = clipBoxXYXY(box, truck_mask.rows, truck_mask.cols);
    if (!clipped.valid()) {
        return 0;
    }

    const int width = clipped.x2 - clipped.x1 + 1;
    const int height = clipped.y2 - clipped.y1 + 1;
    if (width <= 0 || height <= 0) {
        return 0;
    }

    const cv::Rect roi_rect(clipped.x1, clipped.y1, width, height);
    const cv::Mat roi = truck_mask(roi_rect);
    if (roi.empty()) {
        return 0;
    }

    return cv::countNonZero(roi);
}

static std::optional<BoxXYXY> getCenterWidthBox(const BoxXYXY& box, double frac = CENTER_WIDTH_FRAC) {
    if (!box.valid()) {
        return std::nullopt;
    }

    const int bw = std::max(1, box.x2 - box.x1 + 1);
    const int keep_w = std::max(1, pythonRoundToInt(static_cast<double>(bw) * frac));
    const double cx = 0.5 * static_cast<double>(box.x1 + box.x2);

    int nx1 = pythonRoundToInt(cx - static_cast<double>(keep_w) / 2.0);
    int nx2 = nx1 + keep_w - 1;

    nx1 = std::max(box.x1, nx1);
    nx2 = std::min(box.x2, nx2);

    if (nx2 < nx1) {
        nx1 = box.x1;
        nx2 = box.x2;
    }

    return BoxXYXY{nx1, box.y1, nx2, box.y2};
}

static std::optional<BoxXYXY> getBottomBandBox(
    const BoxXYXY& box,
    int hh,
    int ww,
    int band_height = BOTTOM_STRIP_HEIGHT,
    int offset_up = BOTTOM_STRIP_OFFSET_UP,
    bool center_only = USE_CENTER_BOTTOM_BAND
) {
    const BoxXYXY clipped = clipBoxXYXY(box, hh, ww);

    if (!clipped.valid()) {
        return std::nullopt;
    }

    const int band_y2 = std::max(clipped.y1, clipped.y2 - offset_up);
    const int band_y1 = std::max(clipped.y1, band_y2 - band_height + 1);

    if (band_y2 < band_y1) {
        return std::nullopt;
    }

    BoxXYXY band_box{clipped.x1, band_y1, clipped.x2, band_y2};

    if (center_only) {
        return getCenterWidthBox(band_box, CENTER_WIDTH_FRAC);
    }

    return band_box;
}

static std::optional<BoxXYXY> getLowerBodyBox(
    const BoxXYXY& box,
    int hh,
    int ww,
    double lower_frac = LOWER_BODY_FRAC
) {
    const BoxXYXY clipped = clipBoxXYXY(box, hh, ww);

    if (!clipped.valid()) {
        return std::nullopt;
    }

    const int bh = std::max(1, clipped.y2 - clipped.y1 + 1);
    const int keep_h = std::max(1, pythonRoundToInt(static_cast<double>(bh) * lower_frac));
    const int ny1 = std::max(clipped.y1, clipped.y2 - keep_h + 1);

    return BoxXYXY{clipped.x1, ny1, clipped.x2, clipped.y2};
}

static std::optional<BoxXYXY> getLeftSideBox(
    const BoxXYXY& box,
    int hh,
    int ww,
    double side_frac = SIDE_STRIP_WIDTH_FRAC
) {
    const BoxXYXY clipped = clipBoxXYXY(box, hh, ww);
    if (!clipped.valid()) {
        return std::nullopt;
    }

    const int bw = std::max(1, clipped.x2 - clipped.x1 + 1);
    const int sw = std::max(1, pythonRoundToInt(static_cast<double>(bw) * side_frac));
    return BoxXYXY{clipped.x1, clipped.y1, std::min(clipped.x2, clipped.x1 + sw - 1), clipped.y2};
}

static std::optional<BoxXYXY> getRightSideBox(
    const BoxXYXY& box,
    int hh,
    int ww,
    double side_frac = SIDE_STRIP_WIDTH_FRAC
) {
    const BoxXYXY clipped = clipBoxXYXY(box, hh, ww);
    if (!clipped.valid()) {
        return std::nullopt;
    }

    const int bw = std::max(1, clipped.x2 - clipped.x1 + 1);
    const int sw = std::max(1, pythonRoundToInt(static_cast<double>(bw) * side_frac));
    return BoxXYXY{std::max(clipped.x1, clipped.x2 - sw + 1), clipped.y1, clipped.x2, clipped.y2};
}

static int computeRegionOverlapPixels(const std::optional<BoxXYXY>& box, const cv::Mat& truck_mask) {
    if (!box.has_value() || truck_mask.empty()) {
        return 0;
    }
    return intersectionPixelsBoxMask(box.value(), truck_mask);
}

static RegionOverlapResult computeRegionOverlapPercent(const std::optional<BoxXYXY>& box, const cv::Mat& truck_mask) {
    if (!box.has_value() || truck_mask.empty()) {
        return RegionOverlapResult{0.0, 0, 0};
    }

    const int area = boxAreaXYXY(box.value(), truck_mask.rows, truck_mask.cols);
    if (area <= 0) {
        return RegionOverlapResult{0.0, 0, 0};
    }

    const int overlap_pixels = computeRegionOverlapPixels(box, truck_mask);
    const double overlap_percent = 100.0 * static_cast<double>(overlap_pixels) / static_cast<double>(area);
    return RegionOverlapResult{overlap_percent, overlap_pixels, area};
}

static RegionOverlapResult computeRegionOverlapPercent(const BoxXYXY& box, const cv::Mat& truck_mask) {
    return computeRegionOverlapPercent(std::optional<BoxXYXY>(box), truck_mask);
}

static std::pair<bool, EvalInfo> evaluateHumanTruck(
    const BoxXYXY& logic_box,
    const cv::Mat& truck_mask,
    const ModeConfig& mode_cfg
) {
    const double full_overlap_percent_strict = mode_cfg.FULL_OVERLAP_PERCENT_STRICT;
    const double lower_overlap_percent_relaxed = mode_cfg.LOWER_OVERLAP_PERCENT_RELAXED;
    const double bottom_support_percent_min = mode_cfg.BOTTOM_SUPPORT_PERCENT_MIN;
    const double bottom_dominance_ratio = mode_cfg.BOTTOM_DOMINANCE_RATIO;
    const double side_overlap_percent_max_strict = mode_cfg.SIDE_OVERLAP_PERCENT_MAX_STRICT;
    const std::string& low_overlap_reject_mode = mode_cfg.LOW_OVERLAP_REJECT_MODE;

    const RegionOverlapResult full = computeRegionOverlapPercent(logic_box, truck_mask);

    const std::optional<BoxXYXY> bottom_band_box = getBottomBandBox(logic_box, truck_mask.rows, truck_mask.cols);
    const std::optional<BoxXYXY> lower_body_box = getLowerBodyBox(logic_box, truck_mask.rows, truck_mask.cols);
    const std::optional<BoxXYXY> left_side_box = getLeftSideBox(logic_box, truck_mask.rows, truck_mask.cols);
    const std::optional<BoxXYXY> right_side_box = getRightSideBox(logic_box, truck_mask.rows, truck_mask.cols);

    const RegionOverlapResult bottom = computeRegionOverlapPercent(bottom_band_box, truck_mask);
    const RegionOverlapResult lower = computeRegionOverlapPercent(lower_body_box, truck_mask);
    const RegionOverlapResult left = computeRegionOverlapPercent(left_side_box, truck_mask);
    const RegionOverlapResult right = computeRegionOverlapPercent(right_side_box, truck_mask);

    double side_overlap_percent = 0.0;
    int side_overlap_pixels = 0;
    int side_area = 0;

    if (left.percent >= right.percent) {
        side_overlap_percent = left.percent;
        side_overlap_pixels = left.pixels;
        side_area = left.area;
    } else {
        side_overlap_percent = right.percent;
        side_overlap_pixels = right.pixels;
        side_area = right.area;
    }

    EvalInfo info;
    info.reject_mode = "ok";
    info.assoc_overlap_percent = 0.0;
    info.full_overlap_percent = static_cast<double>(full.percent);
    info.bottom_contact_percent = static_cast<double>(bottom.percent);
    info.lower_overlap_percent = static_cast<double>(lower.percent);
    info.side_overlap_percent = static_cast<double>(side_overlap_percent);
    info.full_overlap_pixels = static_cast<int>(full.pixels);
    info.bottom_contact_pixels = static_cast<int>(bottom.pixels);
    info.lower_overlap_pixels = static_cast<int>(lower.pixels);
    info.side_overlap_pixels = static_cast<int>(side_overlap_pixels);
    info.full_area = static_cast<int>(full.area);
    info.bottom_area = static_cast<int>(bottom.area);
    info.lower_area = static_cast<int>(lower.area);
    info.side_area = static_cast<int>(side_area);
    info.bottom_band_box = bottom_band_box;
    info.lower_body_box = lower_body_box;
    info.left_side_box = left_side_box;
    info.right_side_box = right_side_box;

    if (bottom.percent < bottom_support_percent_min) {
        info.reject_mode = "bottom_no_support";
        return {false, info};
    }

    if (full.percent >= full_overlap_percent_strict) {
        return {true, info};
    }

    const bool bottom_dominates_side = (
        bottom.percent >= (side_overlap_percent * bottom_dominance_ratio)
    );

    const bool relaxed_bottom_ok = (
        bottom.percent >= bottom_support_percent_min &&
        lower.percent >= lower_overlap_percent_relaxed &&
        bottom_dominates_side
    );

    if (relaxed_bottom_ok) {
        info.reject_mode = "ok_relaxed_bottom_support";
        return {true, info};
    }

    if (side_overlap_percent > side_overlap_percent_max_strict && !bottom_dominates_side) {
        info.reject_mode = "side_contact_dominant";
    } else {
        info.reject_mode = low_overlap_reject_mode;
    }

    return {false, info};
}

static std::vector<CandidateTruck> getCandidateTrucks(
    const BoxXYXY& logic_box,
    const std::vector<NormalizedTruck>& trucks,
    const ModeConfig& mode_cfg
) {
    const double min_assoc_overlap_percent = mode_cfg.MIN_ASSOC_OVERLAP_PERCENT;
    std::vector<CandidateTruck> candidates;

    for (const NormalizedTruck& t : trucks) {
        const RegionOverlapResult assoc = computeRegionOverlapPercent(logic_box, t.mask);
        if (assoc.percent >= min_assoc_overlap_percent) {
            CandidateTruck cand;
            cand.truck_idx = t.truck_index;
            cand.truck_mask = t.mask;
            cand.truck_conf = t.conf;
            cand.assoc_overlap_percent = static_cast<double>(assoc.percent);
            cand.assoc_overlap_pixels = static_cast<int>(assoc.pixels);
            cand.assoc_area = static_cast<int>(assoc.area);
            candidates.push_back(cand);
        }
    }

    return candidates;
}

static std::optional<EvaluatedTruck> selectBestTruckResult(
    const BoxXYXY& logic_box,
    const std::vector<CandidateTruck>& candidates,
    const ModeConfig& mode_cfg
) {
    if (candidates.empty()) {
        return std::nullopt;
    }

    std::vector<EvaluatedTruck> evaluated;
    evaluated.reserve(candidates.size());

    for (const CandidateTruck& cand : candidates) {
        const auto eval = evaluateHumanTruck(logic_box, cand.truck_mask, mode_cfg);
        const bool ok = eval.first;
        EvalInfo info = eval.second;

        info.assoc_overlap_percent = static_cast<double>(cand.assoc_overlap_percent);
        info.assoc_overlap_pixels = static_cast<int>(cand.assoc_overlap_pixels);
        info.assoc_area = static_cast<int>(cand.assoc_area);

        const double score = (
            100000.0 * static_cast<int>(ok) +
            1000.0 * static_cast<double>(info.bottom_contact_percent) +
            100.0 * static_cast<double>(info.lower_overlap_percent) +
            10.0 * static_cast<double>(info.full_overlap_percent) +
            1.0 * static_cast<double>(info.assoc_overlap_percent) -
            50.0 * static_cast<double>(info.side_overlap_percent)
        );

        EvaluatedTruck e;
        e.truck_idx = cand.truck_idx;
        e.truck_mask = cand.truck_mask;
        e.ok = static_cast<bool>(ok);
        e.info = std::move(info);
        e.score = static_cast<double>(score);
        evaluated.push_back(std::move(e));
    }

    std::vector<const EvaluatedTruck*> pool;
    for (const EvaluatedTruck& e : evaluated) {
        if (e.ok) {
            pool.push_back(&e);
        }
    }
    if (pool.empty()) {
        for (const EvaluatedTruck& e : evaluated) {
            pool.push_back(&e);
        }
    }

    auto key = [](const EvaluatedTruck& e) {
        return std::make_tuple(
            e.score,
            e.info.bottom_contact_percent,
            e.info.lower_overlap_percent,
            e.info.full_overlap_percent,
            e.info.assoc_overlap_percent,
            -e.info.side_overlap_percent
        );
    };

    const EvaluatedTruck* best = pool.front();
    for (const EvaluatedTruck* e : pool) {
        if (key(*best) < key(*e)) {
            best = e;
        }
    }

    return *best;
}

static cv::Mat normalizeMaskTo01(const cv::Mat& mask_arr) {
    if (mask_arr.empty() || mask_arr.dims != 2 || mask_arr.channels() != 1) {
        return cv::Mat();
    }

    cv::Mat compared;
    cv::compare(mask_arr, cv::Scalar(0), compared, cv::CMP_GT);  // 0 or 255

    cv::Mat mask01;
    compared.convertTo(mask01, CV_8U, 1.0 / 255.0);              // 0 or 1, matching Python astype(uint8)
    return mask01;
}

std::vector<HumanBBox> filterOnTopHumans(
    const cv::Mat& frame,
    const std::vector<TruckMask>& truck_masks,
    const std::vector<HumanBBox>& human_bboxes,
    const std::string& mode = "relaxed"
) {
    const auto& configs = modeConfigs();
    const auto it = configs.find(mode);
    if (it == configs.end()) {
        std::vector<std::string> keys;
        keys.reserve(configs.size());
        for (const auto& kv : configs) {
            keys.push_back(kv.first);
        }
        throw std::invalid_argument("Unsupported mode: " + mode + ". Use relaxed or restricted.");
    }

    if (frame.empty()) {
        return {};
    }

    if (frame.dims < 2) {
        return {};
    }

    const int h = frame.rows;
    const int w = frame.cols;
    if (h <= 0 || w <= 0) {
        return {};
    }

    if (truck_masks.empty() || human_bboxes.empty()) {
        return {};
    }

    const ModeConfig& mode_cfg = it->second;

    // ---------------------------------------------------------
    // Normalize truck masks
    // ---------------------------------------------------------
    std::vector<NormalizedTruck> trucks;

    for (size_t truck_idx = 0; truck_idx < truck_masks.size(); ++truck_idx) {
        try {
            const TruckMask& t = truck_masks[truck_idx];

            if (t.mask.empty()) {
                continue;
            }

            cv::Mat mask_arr = t.mask;
            if (mask_arr.empty()) {
                continue;
            }

            // Allow HxW only, matching Python mask_arr.ndim == 2.
            if (mask_arr.dims != 2 || mask_arr.channels() != 1) {
                continue;
            }

            if (mask_arr.rows != h || mask_arr.cols != w) {
                continue;
            }

            cv::Mat mask01 = normalizeMaskTo01(mask_arr);
            if (mask01.empty() || cv::countNonZero(mask01) == 0) {
                continue;
            }

            if (mode_cfg.ERODE_TRUCK_MASK) {
                mask01 = erodeMask(mask01, mode_cfg.ERODE_KERNEL_SIZE, mode_cfg.ERODE_ITERATIONS);
                if (mask01.empty() || cv::countNonZero(mask01) == 0) {
                    continue;
                }
            }

            NormalizedTruck nt;
            nt.truck_index = static_cast<int>(truck_idx);
            nt.mask = mask01;
            nt.conf = t.conf;
            nt.class_id = 0;
            nt.class_name = "truck";
            trucks.push_back(std::move(nt));
        } catch (...) {
            // Skip bad truck entry instead of crashing the whole batch.
            continue;
        }
    }

    if (trucks.empty()) {
        return {};
    }

    // ---------------------------------------------------------
    // Process humans
    // ---------------------------------------------------------
    std::vector<HumanBBox> filtered;

    for (size_t human_idx = 0; human_idx < human_bboxes.size(); ++human_idx) {
        (void)human_idx;
        try {
            const HumanBBox& item = human_bboxes[human_idx];

            const BoxXYXY raw_box = clipBoxXYXY(item.box, h, w);
            if (!raw_box.valid()) {
                continue;
            }

            const int raw_area = (raw_box.x2 - raw_box.x1 + 1) * (raw_box.y2 - raw_box.y1 + 1);
            if (raw_area <= 0) {
                continue;
            }

            const BoxXYXY logic_box = makeLogicBox(raw_box, item.class_id, h, w);
            if (!logic_box.valid()) {
                continue;
            }

            const int logic_area = (logic_box.x2 - logic_box.x1 + 1) * (logic_box.y2 - logic_box.y1 + 1);
            if (logic_area <= 0) {
                continue;
            }

            const std::vector<CandidateTruck> candidates = getCandidateTrucks(logic_box, trucks, mode_cfg);
            if (candidates.empty()) {
                continue;
            }

            const std::optional<EvaluatedTruck> selected = selectBestTruckResult(logic_box, candidates, mode_cfg);
            if (!selected.has_value()) {
                continue;
            }

            if (selected->ok) {
                filtered.push_back(item);
            }
        } catch (...) {
            // Skip bad human entry instead of torpedoing the whole function.
            continue;
        }
    }

    return filtered;
}

}  // namespace on_top_human_filter

/*
Example usage:

#include "filter_on_top_human.cpp"

int main() {
    cv::Mat frame = cv::imread("frame.jpg");
    cv::Mat truck_mask = cv::imread("truck_mask.png", cv::IMREAD_GRAYSCALE);

    std::vector<on_top_human_filter::TruckMask> truck_masks = {
        {truck_mask, 1.0}
    };

    std::vector<on_top_human_filter::HumanBBox> humans = {
        {{100, 50, 180, 250}, 1},
        {{300, 80, 390, 260}, 2}
    };

    auto filtered = on_top_human_filter::filterOnTopHumans(frame, truck_masks, humans, "relaxed");
    return 0;
}

Compile example:
g++ -std=c++17 filter_on_top_human.cpp -c $(pkg-config --cflags --libs opencv4)
*/
