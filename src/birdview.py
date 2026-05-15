import cv2
import numpy as np


def preprocess_mask(mask):
    """2値化とモルフォロジー処理で道路マスクを整形する。"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return select_tallest_component(binary)


def select_tallest_component(mask):
    """縦方向に最も長い連結成分を道路候補として選ぶ。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    best_label = None
    best_score = -1

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        if area < 200:
            continue

        score = height + 0.001 * area

        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return mask

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def fit_line_ransac_y_to_x(points):
    """点群に対して x = a*y + b を RANSAC で近似する。"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return None

    ys = points[:, 1]
    xs = points[:, 0]

    best_a = None
    best_b = None
    best_inliers = 0

    rng = np.random.default_rng(0)
    iterations = 200
    threshold = 8.0

    for _ in range(iterations):
        idx = rng.choice(len(points), size=2, replace=False)
        y1, y2 = ys[idx[0]], ys[idx[1]]
        x1, x2 = xs[idx[0]], xs[idx[1]]

        if abs(y2 - y1) < 1e-6:
            continue

        a = (x2 - x1) / (y2 - y1)
        b = x1 - a * y1

        pred_x = a * ys + b
        errors = np.abs(xs - pred_x)
        inliers = int(np.sum(errors < threshold))

        if inliers > best_inliers:
            best_inliers = inliers
            best_a = a
            best_b = b

    if best_a is None:
        return None

    pred_x = best_a * ys + best_b
    errors = np.abs(xs - pred_x)
    inlier_mask = errors < threshold

    if np.sum(inlier_mask) >= 2:
        a, b = np.polyfit(ys[inlier_mask], xs[inlier_mask], 1)
        return float(a), float(b)

    return best_a, best_b


def fit_poly_robust_y_to_x(points, degree=2, max_iter=6):
    """
    点群に対して x = f(y) をロバストに多項式近似する。
    近似は y 正規化後に行い、外れ値に対して反復重み付けで安定化する。
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < max(3, degree + 1):
        return None

    ys = points[:, 1]
    xs = points[:, 0]
    y_mean = float(np.mean(ys))
    y_scale = float(np.std(ys))
    if y_scale < 1e-6:
        return None

    yn = (ys - y_mean) / y_scale
    fit_degree = min(degree, len(points) - 1)
    coeff = np.polyfit(yn, xs, fit_degree)

    for _ in range(max_iter):
        pred = np.polyval(coeff, yn)
        residual = xs - pred
        mad = float(np.median(np.abs(residual - np.median(residual))))
        if mad < 1e-6:
            break
        sigma = 1.4826 * mad + 1e-6
        u = residual / (2.5 * sigma)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u**2) ** 2, 0.05)
        coeff = np.polyfit(yn, xs, fit_degree, w=weights)

    return {
        "coeff": coeff,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def eval_poly_model_x(model, y):
    yn = (float(y) - model["y_mean"]) / model["y_scale"]
    return float(np.polyval(model["coeff"], yn))


def eval_poly_model_dx_dy(model, y):
    yn = (float(y) - model["y_mean"]) / model["y_scale"]
    dcoeff = np.polyder(model["coeff"])
    return float(np.polyval(dcoeff, yn) / model["y_scale"])


def local_median_width_at_y(width_table, y, window):
    y_int = int(round(float(y)))
    y0 = max(0, y_int - window)
    y1 = min(len(width_table), y_int + window + 1)
    if y0 >= y1:
        return np.nan

    values = width_table[y0:y1]
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


def skeletonize_mask(binary_mask):
    """
    2値マスクを細線化して1px幅のスケルトンを得る。
    OpenCV標準機能のみで動作するようにモルフォロジー反復で実装。
    """
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for _ in range(1024):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        residue = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, residue)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return skel


def nearest_foreground_x_on_row(mask, y, x_hint):
    """
    指定行の前景画素のうち、x_hint に最も近い x を返す。
    """
    y_int = int(round(float(y)))
    if y_int < 0 or y_int >= mask.shape[0]:
        return None

    xs = np.where(mask[y_int] > 0)[0]
    if len(xs) == 0:
        return None

    x_hint_int = int(round(float(x_hint)))
    idx = int(np.argmin(np.abs(xs - x_hint_int)))
    return int(xs[idx])


def trace_last_inside(mask, start_x, start_y, dir_x, dir_y, max_steps):
    """
    開始点から方向ベクトルへ進み、前景が切れる直前の点を返す。
    """
    h, w = mask.shape[:2]
    prev = None
    step_size = 1.0

    for step in range(1, max_steps + 1):
        x = int(round(start_x + dir_x * step * step_size))
        y = int(round(start_y + dir_y * step * step_size))
        if x < 0 or x >= w or y < 0 or y >= h:
            return prev
        if mask[y, x] == 0:
            return prev
        prev = (float(x), float(y))

    return prev


def estimate_quad_skeleton_normal(mask):
    """
    追加手法:
    1) 道路マスクをスケルトン化して中心線を抽出
    2) 中心線法線方向へ左右境界を探索して幅を推定
    3) 左右境界回帰 + 幾何制約最適化で4点を決定
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    binary = select_tallest_component(binary)
    h, w = binary.shape[:2]

    skel = skeletonize_mask(binary)
    skel = select_tallest_component(skel)

    center_points = []
    for y in range(h):
        xs = np.where(skel[y] > 0)[0]
        if len(xs) == 0:
            continue
        center_points.append([float(np.median(xs)), float(y)])

    if len(center_points) < 25:
        return None

    center_points = np.asarray(center_points, dtype=np.float32)
    center_model = fit_poly_robust_y_to_x(center_points, degree=2)
    if center_model is None:
        return None

    y_values = center_points[:, 1]
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    span = y_max - y_min
    if span < 30:
        return None

    sample_count = min(90, max(30, int(span / 2.5)))
    sample_ys = np.linspace(y_min + 0.08 * span, y_min + 0.96 * span, num=sample_count)
    max_steps = int(max(h, w) * 0.75)
    min_width = max(20.0, 0.03 * w)

    left_points = []
    right_points = []
    widths = []
    width_table = np.full(h, np.nan, dtype=np.float32)

    for y in sample_ys:
        cy = int(round(float(y)))
        if cy < 0 or cy >= h:
            continue

        cx_est = eval_poly_model_x(center_model, y)
        cx = nearest_foreground_x_on_row(binary, cy, cx_est)
        if cx is None:
            continue

        slope = eval_poly_model_dx_dy(center_model, y)
        nx, ny = 1.0, -float(slope)
        norm = float(np.hypot(nx, ny))
        if norm < 1e-6:
            continue
        nx /= norm
        ny /= norm

        p_pos = trace_last_inside(binary, float(cx), float(cy), nx, ny, max_steps)
        p_neg = trace_last_inside(binary, float(cx), float(cy), -nx, -ny, max_steps)
        if p_pos is None or p_neg is None:
            continue

        if p_pos[0] <= p_neg[0]:
            left_pt, right_pt = p_pos, p_neg
        else:
            left_pt, right_pt = p_neg, p_pos

        width_x = right_pt[0] - left_pt[0]
        if width_x < min_width:
            continue

        left_points.append(left_pt)
        right_points.append(right_pt)
        widths.append(width_x)
        width_table[cy] = float(width_x)

    if len(left_points) < 20:
        return None

    left_points = np.asarray(left_points, dtype=np.float32)
    right_points = np.asarray(right_points, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)

    median_width = float(np.median(widths))
    valid = (widths > median_width * 0.45) & (widths < median_width * 1.9)
    left_points = left_points[valid]
    right_points = right_points[valid]
    widths = widths[valid]

    if len(left_points) < 16:
        return None

    left_model = fit_poly_robust_y_to_x(left_points, degree=2)
    right_model = fit_poly_robust_y_to_x(right_points, degree=2)
    if left_model is None or right_model is None:
        return None

    y_used = np.concatenate([left_points[:, 1], right_points[:, 1]])
    y_low = float(np.percentile(y_used, 10))
    y_high = float(np.percentile(y_used, 95))
    span_used = y_high - y_low
    if span_used < 25:
        return None

    top_candidates = np.linspace(y_low + 0.03 * span_used, y_low + 0.28 * span_used, num=8)
    bottom_candidates = np.linspace(y_low + 0.70 * span_used, y_low + 0.98 * span_used, num=10)
    window = max(5, int(h * 0.03))

    best_loss = None
    best_quad = None

    for y_top in top_candidates:
        for y_bottom in bottom_candidates:
            if y_bottom - y_top < span_used * 0.28:
                continue

            xL_top = eval_poly_model_x(left_model, y_top)
            xR_top = eval_poly_model_x(right_model, y_top)
            xL_bottom = eval_poly_model_x(left_model, y_bottom)
            xR_bottom = eval_poly_model_x(right_model, y_bottom)

            width_top = xR_top - xL_top
            width_bottom = xR_bottom - xL_bottom
            if width_top < min_width or width_bottom < min_width:
                continue
            if width_bottom < width_top * 0.92:
                continue

            obs_top = local_median_width_at_y(width_table, y_top, window)
            obs_bottom = local_median_width_at_y(width_table, y_bottom, window)
            if not np.isfinite(obs_top):
                obs_top = width_top
            if not np.isfinite(obs_bottom):
                obs_bottom = width_bottom

            width_fit = (
                abs(width_top - obs_top) / max(obs_top, 1.0)
                + abs(width_bottom - obs_bottom) / max(obs_bottom, 1.0)
            )

            center_top = eval_poly_model_x(center_model, y_top)
            center_bottom = eval_poly_model_x(center_model, y_bottom)
            symmetry_penalty = (
                abs((xL_top + xR_top) * 0.5 - center_top) / max(width_top, 1.0)
                + abs((xL_bottom + xR_bottom) * 0.5 - center_bottom) / max(width_bottom, 1.0)
            )

            slope_diff = (
                abs(eval_poly_model_dx_dy(left_model, y_top) - eval_poly_model_dx_dy(right_model, y_top))
                + abs(eval_poly_model_dx_dy(left_model, y_bottom) - eval_poly_model_dx_dy(right_model, y_bottom))
            )

            quad = np.float32([
                [xL_top, y_top],
                [xR_top, y_top],
                [xR_bottom, y_bottom],
                [xL_bottom, y_bottom],
            ])

            area = abs(cv2.contourArea(quad.astype(np.float32)))
            area_score = area / max(float(h * w), 1.0)
            loss = 1.2 * width_fit + 1.0 * symmetry_penalty + 0.7 * slope_diff - 0.5 * area_score

            if best_loss is None or loss < best_loss:
                best_loss = float(loss)
                best_quad = quad

    if best_quad is None:
        return None

    best_quad[:, 0] = np.clip(best_quad[:, 0], 0, w - 1)
    best_quad[:, 1] = np.clip(best_quad[:, 1], 0, h - 1)

    if best_quad[0, 0] >= best_quad[1, 0] or best_quad[3, 0] >= best_quad[2, 0]:
        return None
    if cv2.contourArea(best_quad.astype(np.float32)) < max(300.0, 0.003 * h * w):
        return None

    return best_quad


def estimate_quad_boundary_poly_opt(mask):
    """
    追加手法:
    1) 左右境界点を抽出してロバストな多項式で回帰
    2) 4点候補を走査し、幾何制約（幅整合・左右境界の向き整合）で最適化
    """
    h, w = mask.shape[:2]
    width_table = np.full(h, np.nan, dtype=np.float32)

    left_points = []
    right_points = []
    widths = []

    min_row_pixels = max(10, int(w * 0.02))
    min_width = max(20, int(w * 0.03))

    for y in range(h):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) < min_row_pixels:
            continue

        left_x = int(xs.min())
        right_x = int(xs.max())
        width = right_x - left_x
        if width < min_width:
            continue

        left_points.append([left_x, y])
        right_points.append([right_x, y])
        widths.append(width)
        width_table[y] = float(width)

    if len(left_points) < 30:
        return None

    left_points = np.asarray(left_points, dtype=np.float32)
    right_points = np.asarray(right_points, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)

    # 明らかに異常な道路幅を除去して境界回帰を安定化する。
    q1, q3 = np.percentile(widths, [15, 85])
    valid = (widths >= q1 * 0.6) & (widths <= q3 * 1.6)
    left_points = left_points[valid]
    right_points = right_points[valid]
    if len(left_points) < 20:
        return None

    left_model = fit_poly_robust_y_to_x(left_points, degree=2)
    right_model = fit_poly_robust_y_to_x(right_points, degree=2)
    if left_model is None or right_model is None:
        return None

    y_values = left_points[:, 1]
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    span = y_max - y_min
    if span < 25:
        return None

    top_candidates = np.linspace(y_min + 0.08 * span, y_min + 0.35 * span, num=10)
    bottom_candidates = np.linspace(y_min + 0.65 * span, y_min + 0.95 * span, num=12)
    window = max(5, int(h * 0.03))

    best_loss = None
    best_quad = None

    for y_top in top_candidates:
        for y_bottom in bottom_candidates:
            if y_bottom - y_top < span * 0.25:
                continue

            xL_top = eval_poly_model_x(left_model, y_top)
            xR_top = eval_poly_model_x(right_model, y_top)
            xL_bottom = eval_poly_model_x(left_model, y_bottom)
            xR_bottom = eval_poly_model_x(right_model, y_bottom)

            width_top = xR_top - xL_top
            width_bottom = xR_bottom - xL_bottom
            if width_top < min_width or width_bottom < min_width:
                continue
            if width_bottom <= width_top * 1.02:
                continue

            obs_top = local_median_width_at_y(width_table, y_top, window)
            obs_bottom = local_median_width_at_y(width_table, y_bottom, window)
            if not np.isfinite(obs_top):
                obs_top = width_top
            if not np.isfinite(obs_bottom):
                obs_bottom = width_bottom

            width_fit = (
                abs(width_top - obs_top) / max(obs_top, 1.0)
                + abs(width_bottom - obs_bottom) / max(obs_bottom, 1.0)
            )

            # 左右境界の接線傾き差を抑え、幾何的に無理のない台形を優先する。
            slope_diff = (
                abs(eval_poly_model_dx_dy(left_model, y_top) - eval_poly_model_dx_dy(right_model, y_top))
                + abs(eval_poly_model_dx_dy(left_model, y_bottom) - eval_poly_model_dx_dy(right_model, y_bottom))
            )
            ratio = width_bottom / max(width_top, 1e-6)
            ratio_penalty = 0.0 if 1.05 <= ratio <= 8.0 else abs(np.log(max(ratio, 1e-6) / 2.5))

            quad = np.float32([
                [xL_top, y_top],
                [xR_top, y_top],
                [xR_bottom, y_bottom],
                [xL_bottom, y_bottom],
            ])

            area = abs(cv2.contourArea(quad.astype(np.float32)))
            area_score = area / max(float(h * w), 1.0)
            loss = 1.2 * width_fit + 0.9 * slope_diff + 0.7 * ratio_penalty - 0.6 * area_score

            if best_loss is None or loss < best_loss:
                best_loss = float(loss)
                best_quad = quad

    if best_quad is None:
        return None

    best_quad[:, 0] = np.clip(best_quad[:, 0], 0, w - 1)
    best_quad[:, 1] = np.clip(best_quad[:, 1], 0, h - 1)

    if best_quad[0, 0] >= best_quad[1, 0] or best_quad[3, 0] >= best_quad[2, 0]:
        return None
    if cv2.contourArea(best_quad.astype(np.float32)) < max(300.0, 0.003 * h * w):
        return None

    return best_quad


def estimate_quad_width_filter_ransac(mask):
    """
    提案手法:
    1) 各走査線の道路幅を計測して外れ値を除外
    2) 左右境界を RANSAC で直線近似
    3) 上下端から4点を自動推定
    """
    h, w = mask.shape[:2]

    left_points = []
    right_points = []
    widths = []

    for y in range(h):
        xs = np.where(mask[y] > 0)[0]

        if len(xs) < 10:
            continue

        left_x = int(xs.min())
        right_x = int(xs.max())
        width = right_x - left_x

        if width < max(20, int(w * 0.03)):
            continue

        left_points.append([left_x, y])
        right_points.append([right_x, y])
        widths.append(width)

    if len(left_points) < 20:
        return None

    left_points = np.array(left_points, dtype=np.float32)
    right_points = np.array(right_points, dtype=np.float32)
    widths = np.array(widths, dtype=np.float32)

    median_width = np.median(widths)
    valid = (widths > median_width * 0.45) & (widths < median_width * 1.8)

    left_points = left_points[valid]
    right_points = right_points[valid]

    if len(left_points) < 20:
        return None

    left_model = fit_line_ransac_y_to_x(left_points)
    right_model = fit_line_ransac_y_to_x(right_points)

    if left_model is None or right_model is None:
        return None

    aL, bL = left_model
    aR, bR = right_model

    y_min = int(max(left_points[:, 1].min(), right_points[:, 1].min()))
    y_max = int(min(left_points[:, 1].max(), right_points[:, 1].max()))

    if y_max <= y_min + 10:
        return None

    y_top = y_min + int((y_max - y_min) * 0.05)
    y_bottom = y_min + int((y_max - y_min) * 0.95)

    xL_top = aL * y_top + bL
    xR_top = aR * y_top + bR
    xR_bottom = aR * y_bottom + bR
    xL_bottom = aL * y_bottom + bL

    src_pts = np.float32([
        [xL_top, y_top],
        [xR_top, y_top],
        [xR_bottom, y_bottom],
        [xL_bottom, y_bottom],
    ])

    src_pts[:, 0] = np.clip(src_pts[:, 0], 0, w - 1)
    src_pts[:, 1] = np.clip(src_pts[:, 1], 0, h - 1)

    if src_pts[0, 0] >= src_pts[1, 0] or src_pts[3, 0] >= src_pts[2, 0]:
        return None

    return src_pts


def estimate_quad_by_contour(mask):
    """比較手法: 最大輪郭から四角形近似を行う。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 200:
        return None

    peri = cv2.arcLength(contour, True)

    for eps_ratio in [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
        approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
        if len(approx) == 4:
            return order_points(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    return order_points(box)


def order_points(pts):
    """点列を [左上, 右上, 右下, 左下] に並べ替える。"""
    pts = np.asarray(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.float32([tl, tr, br, bl])


def make_birdview(image, src_pts, out_w=512, out_h=768):
    """4点対応から射影変換を行い俯瞰画像を生成する。"""
    dst_pts = np.float32([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ])

    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def draw_quad(image, src_pts):
    """推定4点と四角形を重畳描画する。"""
    vis = image.copy()
    pts = src_pts.astype(np.int32)

    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)

    labels = ["LT", "RT", "RB", "LB"]
    for (x, y), label in zip(pts, labels):
        cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), -1)
        cv2.putText(
            vis, label, (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )

    return vis
