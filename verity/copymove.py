"""Clue 4: Copy-move detection.

A common trick is to copy one part of a photo and paste it elsewhere in
the SAME photo (hide an object, duplicate a crowd). We find small
distinctive points (keypoints), and look for points that match each
other but sit far apart. Many such matching pairs = possible copy-paste.
"""
import numpy as np
import cv2


def analyze_copymove(path, min_dist=40):
    img = cv2.imread(path)
    if img is None:
        return {"name": "Copy-Move", "available": False, "score": 0.0,
                "flags": ["Could not read image."]}

    # work on a reasonable size for speed
    scale = 1024 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=4000)
    kps, desc = sift.detectAndCompute(gray, None)
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if desc is None or len(kps) < 10:
        return {"name": "Copy-Move", "available": True, "score": 0.0,
                "flags": ["Too few details to check for copy-paste."], "heatmap": vis}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc, desc, k=3)

    pairs = []
    for m in knn:
        # m[0] is the point itself; m[1] is its best OTHER match
        if len(m) < 3:
            continue
        best, second = m[1], m[2]
        if best.distance < 0.5 * second.distance:          # clear, unique match
            p1 = np.array(kps[best.queryIdx].pt)
            p2 = np.array(kps[best.trainIdx].pt)
            if np.linalg.norm(p1 - p2) > min_dist:
                pairs.append((p1, p2))

    # keep pairs whose shift agrees with others (real copy-move = same shift)
    good = []
    if len(pairs) >= 4:
        shifts = np.array([p2 - p1 for p1, p2 in pairs])
        for i, s in enumerate(shifts):
            similar = np.sum(np.linalg.norm(shifts - s, axis=1) < 15)
            similar += np.sum(np.linalg.norm(shifts + s, axis=1) < 15)
            if similar >= 4:
                good.append(pairs[i])

    for p1, p2 in good:
        a, b = tuple(int(x) for x in p1), tuple(int(x) for x in p2)
        cv2.line(vis, a, b, (255, 60, 60), 1)
        cv2.circle(vis, a, 3, (255, 220, 0), -1)
        cv2.circle(vis, b, 3, (255, 220, 0), -1)

    n = len(good) // 2   # each pair is found from both sides
    score = 0.0 if n < 8 else min(1.0, 0.3 + n / 40)   # fewer than 8 = normal coincidence
    flags = [f"{n} matching point pairs suggest a region may be copied and pasted "
             "within the image." if score > 0 else
             "No duplicated regions detected."]
    if score > 0:
        flags.append("Repeating patterns (tiles, windows, fabric) can also cause matches.")
    return {"name": "Copy-Move", "score": float(score), "flags": flags,
            "heatmap": vis, "available": True}
