import cv2
import numpy as np


def filter_on_top_humans(frame, truck_masks, human_bboxes, mode="relaxed"):
    """
    Filters input human bboxes and returns only the humans classified as 'on top'.

    Robustness goals:
    - Safely handles None / empty inputs
    - Safely skips malformed boxes or masks
    - Avoids crashes on empty ROIs, wrong shapes, bad types
    - Returns [] instead of failing for normal "nothing detected" cases

    Parameters
    ----------
    frame : np.ndarray
        Input image frame, shape (H, W, 3) or (H, W).
    truck_masks : list
        Each item may be:
          - np.ndarray mask of shape (H, W), or
          - dict with key "mask" -> np.ndarray, optional "conf"
    human_bboxes : list
        Each item may be:
          - dict with one of: "raw_box", "xyxy", "bbox"
            and optionally "class_id" or "cls"
          - or iterable [x1, y1, x2, y2]
    mode : str
        "relaxed" or "restricted"

    Returns
    -------
    list
        Filtered list of the original human_bboxes items that are on top.
    """

    MODE_CONFIGS = {
        "relaxed": {
            "MIN_ASSOC_OVERLAP_PERCENT": 2.0,
            "FULL_OVERLAP_PERCENT_STRICT": 6.0,
            "LOWER_OVERLAP_PERCENT_RELAXED": 3.0,
            "BOTTOM_SUPPORT_PERCENT_MIN": 2.5,
            "BOTTOM_DOMINANCE_RATIO": 1.20,
            "SIDE_OVERLAP_PERCENT_MAX_STRICT": 45.0,
            "LOW_OVERLAP_REJECT_MODE": "low_rel_overlap",
            "ERODE_TRUCK_MASK": False,
            "ERODE_KERNEL_SIZE": 3,
            "ERODE_ITERATIONS": 1,
        },
        "restricted": {
            "MIN_ASSOC_OVERLAP_PERCENT": 4.0,
            "FULL_OVERLAP_PERCENT_STRICT": 24.0,
            "LOWER_OVERLAP_PERCENT_RELAXED": 18.0,
            "BOTTOM_SUPPORT_PERCENT_MIN": 18.0,
            "BOTTOM_DOMINANCE_RATIO": 5.0,
            "SIDE_OVERLAP_PERCENT_MAX_STRICT": 6.0,
            "LOW_OVERLAP_REJECT_MODE": "low_percent_overlap",
            "ERODE_TRUCK_MASK": True,
            "ERODE_KERNEL_SIZE": 5,
            "ERODE_ITERATIONS": 1,
        }
    }

    # Shared constants
    BOTTOM_STRIP_HEIGHT = 10
    BOTTOM_STRIP_OFFSET_UP = 10
    LOWER_BODY_FRAC = 0.50
    SIDE_STRIP_WIDTH_FRAC = 0.18
    CENTER_WIDTH_FRAC = 0.40
    USE_CENTER_BOTTOM_BAND = True

    HOOK_CLASS_ID = 2
    HOOK_TOP_CROP_FRAC = 0.32
    HOOK_MIN_KEEP_FRAC = 0.50

    # ---------------------------------------------------------
    # Early guards for common empty / invalid situations
    # ---------------------------------------------------------
    if mode not in MODE_CONFIGS:
        raise ValueError(f"Unsupported mode: {mode}. Use one of {list(MODE_CONFIGS.keys())}")

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    if frame.ndim < 2:
        return []

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return []

    if truck_masks is None:
        truck_masks = []
    if human_bboxes is None:
        human_bboxes = []

    if not isinstance(truck_masks, (list, tuple)):
        truck_masks = [truck_masks]
    if not isinstance(human_bboxes, (list, tuple)):
        human_bboxes = [human_bboxes]

    if len(truck_masks) == 0 or len(human_bboxes) == 0:
        return []

    mode_cfg = MODE_CONFIGS[mode]

    # ---------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------
    def clip_box_xyxy(box, shape_hw):
        hh, ww = shape_hw
        if box is None:
            return 0, 0, -1, -1

        try:
            if len(box) < 4:
                return 0, 0, -1, -1
            x1, y1, x2, y2 = box[:4]
            x1 = int(np.clip(round(float(x1)), 0, ww - 1))
            y1 = int(np.clip(round(float(y1)), 0, hh - 1))
            x2 = int(np.clip(round(float(x2)), 0, ww - 1))
            y2 = int(np.clip(round(float(y2)), 0, hh - 1))
            return x1, y1, x2, y2
        except Exception:
            return 0, 0, -1, -1

    def box_area_xyxy(box, shape_hw):
        x1, y1, x2, y2 = clip_box_xyxy(box, shape_hw)
        if x2 < x1 or y2 < y1:
            return 0
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    def erode_mask(mask01, kernel_size, iterations):
        try:
            if mask01 is None or mask01.size == 0:
                return None

            if kernel_size <= 1 or iterations <= 0:
                return mask01.copy()

            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            eroded = cv2.erode(mask01.astype(np.uint8), kernel, iterations=iterations)

            # If erosion wipes out the mask completely, keep the original
            if eroded is None or int(eroded.sum()) == 0:
                return mask01.copy()

            return eroded
        except Exception:
            # In robustness mode, fail softly
            return mask01.copy() if mask01 is not None else None

    def make_logic_box(raw_box, class_id, shape_hw):
        x1, y1, x2, y2 = clip_box_xyxy(raw_box, shape_hw)

        if x2 < x1 or y2 < y1:
            return np.array([0, 0, -1, -1], dtype=np.int32)

        if class_id != HOOK_CLASS_ID:
            return np.array([x1, y1, x2, y2], dtype=np.int32)

        hh = max(1, y2 - y1 + 1)
        crop_top = int(round(hh * HOOK_TOP_CROP_FRAC))
        keep_min = int(round(hh * HOOK_MIN_KEEP_FRAC))

        ny1 = min(y2, y1 + crop_top)
        kept_h = y2 - ny1 + 1

        if kept_h < keep_min:
            ny1 = max(y1, y2 - keep_min + 1)

        return np.array([x1, ny1, x2, y2], dtype=np.int32)

    def intersection_pixels_box_mask(box, truck_mask):
        if truck_mask is None or truck_mask.size == 0:
            return 0

        x1, y1, x2, y2 = clip_box_xyxy(box, truck_mask.shape[:2])
        if x2 < x1 or y2 < y1:
            return 0

        roi = truck_mask[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0:
            return 0

        return int(np.count_nonzero(roi))

    def get_center_width_box(box, frac=CENTER_WIDTH_FRAC):
        x1, y1, x2, y2 = box
        if x2 < x1 or y2 < y1:
            return None

        bw = max(1, x2 - x1 + 1)
        keep_w = max(1, int(round(bw * frac)))
        cx = 0.5 * (x1 + x2)

        nx1 = int(round(cx - keep_w / 2.0))
        nx2 = nx1 + keep_w - 1

        nx1 = max(x1, nx1)
        nx2 = min(x2, nx2)

        if nx2 < nx1:
            nx1, nx2 = x1, x2

        return (nx1, y1, nx2, y2)

    def get_bottom_band_box(box, shape_hw,
                            band_height=BOTTOM_STRIP_HEIGHT,
                            offset_up=BOTTOM_STRIP_OFFSET_UP,
                            center_only=USE_CENTER_BOTTOM_BAND):
        x1, y1, x2, y2 = clip_box_xyxy(box, shape_hw)

        if x2 < x1 or y2 < y1:
            return None

        band_y2 = max(y1, y2 - offset_up)
        band_y1 = max(y1, band_y2 - band_height + 1)

        if band_y2 < band_y1:
            return None

        band_box = (x1, band_y1, x2, band_y2)

        if center_only:
            band_box = get_center_width_box(band_box, CENTER_WIDTH_FRAC)

        return band_box

    def get_lower_body_box(box, shape_hw, lower_frac=LOWER_BODY_FRAC):
        x1, y1, x2, y2 = clip_box_xyxy(box, shape_hw)

        if x2 < x1 or y2 < y1:
            return None

        bh = max(1, y2 - y1 + 1)
        keep_h = max(1, int(round(bh * lower_frac)))
        ny1 = max(y1, y2 - keep_h + 1)

        return (x1, ny1, x2, y2)

    def get_left_side_box(box, shape_hw, side_frac=SIDE_STRIP_WIDTH_FRAC):
        x1, y1, x2, y2 = clip_box_xyxy(box, shape_hw)
        if x2 < x1 or y2 < y1:
            return None

        bw = max(1, x2 - x1 + 1)
        sw = max(1, int(round(bw * side_frac)))
        return (x1, y1, min(x2, x1 + sw - 1), y2)

    def get_right_side_box(box, shape_hw, side_frac=SIDE_STRIP_WIDTH_FRAC):
        x1, y1, x2, y2 = clip_box_xyxy(box, shape_hw)
        if x2 < x1 or y2 < y1:
            return None

        bw = max(1, x2 - x1 + 1)
        sw = max(1, int(round(bw * side_frac)))
        return (max(x1, x2 - sw + 1), y1, x2, y2)

    def compute_region_overlap_pixels(box, truck_mask):
        if box is None or truck_mask is None:
            return 0
        return intersection_pixels_box_mask(box, truck_mask)

    def compute_region_overlap_percent(box, truck_mask):
        if box is None or truck_mask is None:
            return 0.0, 0, 0

        area = box_area_xyxy(box, truck_mask.shape[:2])
        if area <= 0:
            return 0.0, 0, 0

        overlap_pixels = compute_region_overlap_pixels(box, truck_mask)
        overlap_percent = 100.0 * float(overlap_pixels) / float(area)
        return overlap_percent, overlap_pixels, area

    def evaluate_human_truck(logic_box, truck_mask, mode_cfg):
        full_overlap_percent_strict = mode_cfg["FULL_OVERLAP_PERCENT_STRICT"]
        lower_overlap_percent_relaxed = mode_cfg["LOWER_OVERLAP_PERCENT_RELAXED"]
        bottom_support_percent_min = mode_cfg["BOTTOM_SUPPORT_PERCENT_MIN"]
        bottom_dominance_ratio = mode_cfg["BOTTOM_DOMINANCE_RATIO"]
        side_overlap_percent_max_strict = mode_cfg["SIDE_OVERLAP_PERCENT_MAX_STRICT"]
        low_overlap_reject_mode = mode_cfg["LOW_OVERLAP_REJECT_MODE"]

        full_overlap_percent, full_overlap_pixels, full_area = compute_region_overlap_percent(logic_box, truck_mask)

        bottom_band_box = get_bottom_band_box(logic_box, truck_mask.shape[:2])
        lower_body_box = get_lower_body_box(logic_box, truck_mask.shape[:2])
        left_side_box = get_left_side_box(logic_box, truck_mask.shape[:2])
        right_side_box = get_right_side_box(logic_box, truck_mask.shape[:2])

        bottom_contact_percent, bottom_contact_pixels, bottom_area = compute_region_overlap_percent(bottom_band_box, truck_mask)
        lower_overlap_percent, lower_overlap_pixels, lower_area = compute_region_overlap_percent(lower_body_box, truck_mask)
        left_side_overlap_percent, left_side_overlap_pixels, left_side_area = compute_region_overlap_percent(left_side_box, truck_mask)
        right_side_overlap_percent, right_side_overlap_pixels, right_side_area = compute_region_overlap_percent(right_side_box, truck_mask)

        if left_side_overlap_percent >= right_side_overlap_percent:
            side_overlap_percent = left_side_overlap_percent
            side_overlap_pixels = left_side_overlap_pixels
            side_area = left_side_area
        else:
            side_overlap_percent = right_side_overlap_percent
            side_overlap_pixels = right_side_overlap_pixels
            side_area = right_side_area

        info = {
            "reject_mode": "ok",
            "assoc_overlap_percent": 0.0,
            "full_overlap_percent": float(full_overlap_percent),
            "bottom_contact_percent": float(bottom_contact_percent),
            "lower_overlap_percent": float(lower_overlap_percent),
            "side_overlap_percent": float(side_overlap_percent),
            "full_overlap_pixels": int(full_overlap_pixels),
            "bottom_contact_pixels": int(bottom_contact_pixels),
            "lower_overlap_pixels": int(lower_overlap_pixels),
            "side_overlap_pixels": int(side_overlap_pixels),
            "full_area": int(full_area),
            "bottom_area": int(bottom_area),
            "lower_area": int(lower_area),
            "side_area": int(side_area),
            "bottom_band_box": bottom_band_box,
            "lower_body_box": lower_body_box,
            "left_side_box": left_side_box,
            "right_side_box": right_side_box,
        }

        if bottom_contact_percent < bottom_support_percent_min:
            info["reject_mode"] = "bottom_no_support"
            return False, info

        if full_overlap_percent >= full_overlap_percent_strict:
            return True, info

        bottom_dominates_side = (
            bottom_contact_percent >= (side_overlap_percent * bottom_dominance_ratio)
        )

        relaxed_bottom_ok = (
            bottom_contact_percent >= bottom_support_percent_min
            and lower_overlap_percent >= lower_overlap_percent_relaxed
            and bottom_dominates_side
        )

        if relaxed_bottom_ok:
            info["reject_mode"] = "ok_relaxed_bottom_support"
            return True, info

        if side_overlap_percent > side_overlap_percent_max_strict and not bottom_dominates_side:
            info["reject_mode"] = "side_contact_dominant"
        else:
            info["reject_mode"] = low_overlap_reject_mode

        return False, info

    def get_candidate_trucks(logic_box, trucks, mode_cfg):
        min_assoc_overlap_percent = mode_cfg["MIN_ASSOC_OVERLAP_PERCENT"]
        candidates = []

        for t in trucks:
            truck_idx = t["truck_index"]
            truck_mask = t["mask"]

            assoc_overlap_percent, assoc_overlap_pixels, assoc_area = compute_region_overlap_percent(logic_box, truck_mask)
            if assoc_overlap_percent >= min_assoc_overlap_percent:
                candidates.append({
                    "truck_idx": truck_idx,
                    "truck_mask": truck_mask,
                    "truck_conf": t["conf"],
                    "assoc_overlap_percent": float(assoc_overlap_percent),
                    "assoc_overlap_pixels": int(assoc_overlap_pixels),
                    "assoc_area": int(assoc_area),
                })

        return candidates

    def select_best_truck_result(logic_box, candidates, mode_cfg):
        if not candidates:
            return None

        evaluated = []
        for cand in candidates:
            truck_idx = cand["truck_idx"]
            truck_mask = cand["truck_mask"]
            assoc_overlap_percent = cand["assoc_overlap_percent"]
            assoc_overlap_pixels = cand["assoc_overlap_pixels"]
            assoc_area = cand["assoc_area"]

            ok, info = evaluate_human_truck(logic_box, truck_mask, mode_cfg)
            info["assoc_overlap_percent"] = float(assoc_overlap_percent)
            info["assoc_overlap_pixels"] = int(assoc_overlap_pixels)
            info["assoc_area"] = int(assoc_area)

            score = (
                100000 * int(ok)
                + 1000 * float(info.get("bottom_contact_percent", 0.0))
                + 100 * float(info.get("lower_overlap_percent", 0.0))
                + 10 * float(info.get("full_overlap_percent", 0.0))
                + 1 * float(info.get("assoc_overlap_percent", 0.0))
                - 50 * float(info.get("side_overlap_percent", 0.0))
            )

            evaluated.append({
                "truck_idx": truck_idx,
                "truck_mask": truck_mask,
                "ok": bool(ok),
                "info": info,
                "score": float(score),
            })

        positives = [e for e in evaluated if e["ok"]]
        pool = positives if positives else evaluated

        return max(
            pool,
            key=lambda x: (
                x["score"],
                x["info"].get("bottom_contact_percent", 0.0),
                x["info"].get("lower_overlap_percent", 0.0),
                x["info"].get("full_overlap_percent", 0.0),
                x["info"].get("assoc_overlap_percent", 0.0),
                -x["info"].get("side_overlap_percent", 0.0),
            )
        )

    # ---------------------------------------------------------
    # Normalize truck masks
    # ---------------------------------------------------------
    trucks = []
    for truck_idx, t in enumerate(truck_masks):
        try:
            if isinstance(t, dict):
                if "mask" not in t:
                    continue
                mask = t["mask"]
                conf = float(t.get("conf", 1.0))
            else:
                mask = t
                conf = 1.0

            if mask is None:
                continue

            mask_arr = np.asarray(mask)
            if mask_arr.size == 0:
                continue

            # Allow HxW only
            if mask_arr.ndim != 2:
                continue

            if mask_arr.shape != (h, w):
                continue

            mask01 = (mask_arr > 0).astype(np.uint8)
            if int(mask01.sum()) == 0:
                continue

            if mode_cfg.get("ERODE_TRUCK_MASK", False):
                mask01 = erode_mask(
                    mask01,
                    kernel_size=mode_cfg.get("ERODE_KERNEL_SIZE", 3),
                    iterations=mode_cfg.get("ERODE_ITERATIONS", 1)
                )
                if mask01 is None or int(mask01.sum()) == 0:
                    continue

            trucks.append({
                "truck_index": truck_idx,
                "mask": mask01,
                "conf": conf,
                "class_id": 0,
                "class_name": "truck",
            })

        except Exception:
            # Skip bad truck entry instead of crashing the whole batch
            continue

    if len(trucks) == 0:
        return []

    # ---------------------------------------------------------
    # Process humans
    # ---------------------------------------------------------
    filtered = []

    for human_idx, item in enumerate(human_bboxes):
        try:
            if item is None:
                continue

            if isinstance(item, dict):
                if "raw_box" in item:
                    raw_box = item["raw_box"]
                elif "xyxy" in item:
                    raw_box = item["xyxy"]
                elif "bbox" in item:
                    raw_box = item["bbox"]
                else:
                    continue

                class_id = int(item.get("class_id", item.get("cls", 1)))
            else:
                raw_box = item
                class_id = 1

            raw_box = np.array(clip_box_xyxy(raw_box, (h, w)), dtype=np.int32)
            x1, y1, x2, y2 = raw_box
            if x2 < x1 or y2 < y1:
                continue

            raw_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            if raw_area <= 0:
                continue

            logic_box = make_logic_box(raw_box, class_id, (h, w))
            lx1, ly1, lx2, ly2 = logic_box
            if lx2 < lx1 or ly2 < ly1:
                continue

            logic_area = (lx2 - lx1 + 1) * (ly2 - ly1 + 1)
            if logic_area <= 0:
                continue

            candidates = get_candidate_trucks(logic_box, trucks, mode_cfg)
            if not candidates:
                continue

            selected = select_best_truck_result(logic_box, candidates, mode_cfg)
            if selected is None:
                continue

            if selected["ok"]:
                filtered.append(item)

        except Exception:
            # Skip bad human entry instead of torpedoing the whole function
            continue

    return filtered