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
