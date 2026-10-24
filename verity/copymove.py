"""Clue 4: Copy-move detection.

A common trick is to copy one part of a photo and paste it elsewhere in
the SAME photo (hide an object, duplicate a crowd). We find small
distinctive points (keypoints), and look for points that match each
other but sit far apart. Many such matching pairs = possible copy-paste.
"""
import numpy as np
import cv2


def analyze_copymove(path, min_dist=30):
    img = cv2.imread(path)
    if img is None:
        return {"name": "Copy-Move", "available": False, "score": 0.0,
                "flags": ["Could not read image."]}

    # work on a reasonable size for speed
    scale = 1024 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use contrastThreshold=0.03 for robust keypoint extraction on subtle natural textures
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03)
    kps, desc = sift.detectAndCompute(gray, None)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if desc is None or len(kps) < 10:
        return {"name": "Copy-Move", "available": True, "score": 0.0,
                "flags": ["Too few details to check for copy-paste."], "heatmap": vis}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc, desc, k=3)

    pairs = []
    seen = set()
    for m in knn:
        # m[0] is the point itself; m[1] is its best OTHER match
        if len(m) < 3:
            continue
        best, second = m[1], m[2]
        # Relaxed Lowe ratio (0.72) allows detecting rotated, scaled, or compressed copies
        # while naturally rejecting repeating patterns (grids/tiles have equal nearest-neighbor distances)
        if best.distance < 0.72 * second.distance:
            q_idx, t_idx = best.queryIdx, best.trainIdx
            key = (min(q_idx, t_idx), max(q_idx, t_idx))
            if key in seen:
                continue
            seen.add(key)
            p1 = np.array(kps[q_idx].pt)
            p2 = np.array(kps[t_idx].pt)
            if np.linalg.norm(p1 - p2) > min_dist:
                pairs.append((p1, p2))

    good = []
    # 1. Translation shift clustering (for exact translation pastes)
    if len(pairs) >= 4:
        shifts = np.array([p2 - p1 for p1, p2 in pairs])
        for i, s in enumerate(shifts):
            sim = np.sum(np.linalg.norm(shifts - s, axis=1) < 20)
            if sim >= 4:
                good.append(pairs[i])

    # 2. RANSAC affine estimation (for rotated, scaled, or sheared pastes)
    if len(good) < 5 and len(pairs) >= 4:
        pts1 = np.array([p1 for p1, p2 in pairs], dtype=np.float32)
        pts2 = np.array([p2 for p1, p2 in pairs], dtype=np.float32)
        M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=6.0)
        if inliers is not None and inliers.sum() >= 4:
            ransac_pairs = [pairs[i] for i, inl in enumerate(inliers.ravel()) if inl]
            if len(ransac_pairs) > len(good):
                good = ransac_pairs

    for p1, p2 in good:
        a, b = tuple(int(x) for x in p1), tuple(int(x) for x in p2)
        cv2.line(vis, a, b, (255, 60, 60), 1)
        cv2.circle(vis, a, 3, (255, 220, 0), -1)
        cv2.circle(vis, b, 3, (255, 220, 0), -1)

    n = len(good)
    score = 0.0 if n < 4 else min(1.0, 0.35 + n / 20)
    flags = [f"{n} matching point pairs suggest a region may be copied and pasted "
             "within the image." if score > 0 else
             "No duplicated regions detected."]
    if score > 0:
        flags.append("Repeating patterns (tiles, windows, fabric) can also cause matches.")
    return {"name": "Copy-Move", "score": float(score), "flags": flags,
            "heatmap": vis, "available": True}
