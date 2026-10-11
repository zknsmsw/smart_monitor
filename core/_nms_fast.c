/*
 * _nms_fast.c — C implementations of RetinaFace box_decode and NMS.
 *
 * Built as a shared library and called from Python via ctypes.
 * Replaces the pure-Python _box_decode() / _nms() methods in face_detector.py.
 *
 * Build (on RK3566 device):
 *   gcc -O3 -march=armv8.2-a -shared -fPIC -o _nms_fast.so _nms_fast.c -lm
 *
 * Cross-compile (on x86 PC):
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -shared -fPIC \
 *       -o _nms_fast.so _nms_fast.c -lm
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ===================================================================
 * box_decode
 *
 * Convert RetinaFace location offsets + prior boxes into (x1,y1,x2,y2)
 * bounding boxes.
 *
 *   loc:    flat float array, shape [N, 4], model output
 *   priors: flat float array, shape [N, 4], precomputed anchors
 *   boxes:  flat float array, shape [N, 4], output — caller allocates
 *   N:      number of anchors
 *
 * Nothing is allocated or freed here — the caller owns all memory.
 * =================================================================== */
void box_decode(const float *loc, const float *priors, float *boxes, int N)
{
    const float var_center = 0.1f;
    const float var_size   = 0.2f;

    for (int i = 0; i < N; i++) {
        float px = priors[i*4 + 0];
        float py = priors[i*4 + 1];
        float pw = priors[i*4 + 2];
        float ph = priors[i*4 + 3];

        /* decoded centre */
        float cx = px + loc[i*4 + 0] * var_center * pw;
        float cy = py + loc[i*4 + 1] * var_center * ph;

        /* decoded width / height */
        float w = pw * expf(loc[i*4 + 2] * var_size);
        float h = ph * expf(loc[i*4 + 3] * var_size);

        /* corner format */
        boxes[i*4 + 0] = cx - w * 0.5f;
        boxes[i*4 + 1] = cy - h * 0.5f;
        boxes[i*4 + 2] = cx + w * 0.5f;
        boxes[i*4 + 3] = cy + h * 0.5f;
    }
}

/* ===================================================================
 * nms
 *
 * Standard greedy Non-Maximum Suppression.
 *
 *   dets:     flat float array, shape [N, 5]  — [x1,y1,x2,y2,score]
 *   N:        number of boxes
 *   thresh:   IoU threshold (boxes with overlap > thresh are suppressed)
 *   keep_out: int array of length N, caller-allocated — receives indices
 *   returns:  number of kept indices written to keep_out
 * =================================================================== */

typedef struct {
    int   idx;
    float score;
} IdxScore;

static int _cmp_score_desc(const void *a, const void *b)
{
    float sa = ((const IdxScore *)a)->score;
    float sb = ((const IdxScore *)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

int nms(const float *dets, int N, float thresh, int *keep_out)
{
    if (N == 0) return 0;

    /* ── sort indices by score descending ── */
    IdxScore *order = (IdxScore *)malloc((size_t)N * sizeof(IdxScore));
    if (!order) return 0;

    for (int i = 0; i < N; i++) {
        order[i].idx   = i;
        order[i].score = dets[i * 5 + 4];
    }
    qsort(order, (size_t)N, sizeof(IdxScore), _cmp_score_desc);

    /* ── precompute areas ── */
    float *areas = (float *)malloc((size_t)N * sizeof(float));
    if (!areas) { free(order); return 0; }

    for (int i = 0; i < N; i++) {
        float w = dets[i * 5 + 2] - dets[i * 5 + 0] + 1.0f;
        float h = dets[i * 5 + 3] - dets[i * 5 + 1] + 1.0f;
        areas[i] = (w > 0.0f && h > 0.0f) ? w * h : 0.0f;
    }

    /* ── greedy NMS ── */
    int  *suppressed = (int *)calloc((size_t)N, sizeof(int));
    if (!suppressed) { free(areas); free(order); return 0; }

    int keep_count = 0;

    for (int i = 0; i < N; i++) {
        int idx_i = order[i].idx;
        if (suppressed[idx_i]) continue;

        keep_out[keep_count++] = idx_i;

        float x1_i = dets[idx_i * 5 + 0];
        float y1_i = dets[idx_i * 5 + 1];
        float x2_i = dets[idx_i * 5 + 2];
        float y2_i = dets[idx_i * 5 + 3];

        for (int j = i + 1; j < N; j++) {
            int idx_j = order[j].idx;
            if (suppressed[idx_j]) continue;

            float xx1 = (x1_i > dets[idx_j * 5 + 0]) ? x1_i : dets[idx_j * 5 + 0];
            float yy1 = (y1_i > dets[idx_j * 5 + 1]) ? y1_i : dets[idx_j * 5 + 1];
            float xx2 = (x2_i < dets[idx_j * 5 + 2]) ? x2_i : dets[idx_j * 5 + 2];
            float yy2 = (y2_i < dets[idx_j * 5 + 3]) ? y2_i : dets[idx_j * 5 + 3];

            float ow = xx2 - xx1 + 1.0f;
            float oh = yy2 - yy1 + 1.0f;
            if (ow <= 0.0f || oh <= 0.0f) continue;

            float inter   = ow * oh;
            float union_a = areas[idx_i] + areas[idx_j] - inter;
            float overlap = inter / union_a;

            if (overlap > thresh) {
                suppressed[idx_j] = 1;
            }
        }
    }

    free(suppressed);
    free(areas);
    free(order);
    return keep_count;
}
