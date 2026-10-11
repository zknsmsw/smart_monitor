/*
 * _decode_landm.c — C implementation of RetinaFace landmark decode.
 *
 * Converts the five facial-landmark offset predictions into pixel coordinates,
 * given the precomputed prior (anchor) boxes.
 *
 * Build (on RK3566 device):
 *   gcc -O3 -march=armv8.2-a -shared -fPIC -o _decode_landm.so _decode_landm.c -lm
 *
 * Cross-compile (on x86 PC):
 *   aarch64-linux-gnu-gcc -O3 -march=armv8.2-a -shared -fPIC \
 *       -o _decode_landm.so _decode_landm.c -lm
 */
#include <stdlib.h>

/*
 * decode_5_landmarks
 *
 *   pre:    flat float array, shape [N, 10]  — model landmark-offset output
 *           layout: [x0,y0, x1,y1, x2,y2, x3,y3, x4,y4]
 *   priors: flat float array, shape [N, 4]   — precomputed anchor boxes
 *           layout: [cx, cy, w, h]
 *   out:    flat float array, shape [N, 10]  — caller-allocated, receives
 *           decoded landmark coordinates in same [x0,y0, … x4,y4] layout
 *   N:      number of anchors
 *
 * Formula (for each of the 5 landmarks k ∈ {0…4}):
 *   landmark[k].x = prior.cx + offset.x * 0.1 * prior.w
 *   landmark[k].y = prior.cy + offset.y * 0.1 * prior.h
 *
 * Same variance constant (0.1) as _box_decode — defined in the original
 * RetinaFace paper and shared across all coordinate regression heads.
 */
void decode_5_landmarks(const float *pre, const float *priors,
                        float *out, int N)
{
    const float var_landmark = 0.1f;

    for (int i = 0; i < N; i++) {
        float px = priors[i * 4 + 0];   /* anchor centre x */
        float py = priors[i * 4 + 1];   /* anchor centre y */
        float pw = priors[i * 4 + 2];   /* anchor width     */
        float ph = priors[i * 4 + 3];   /* anchor height    */

        for (int k = 0; k < 5; k++) {
            float dx = pre[i * 10 + k * 2 + 0];
            float dy = pre[i * 10 + k * 2 + 1];

            out[i * 10 + k * 2 + 0] = px + dx * var_landmark * pw;
            out[i * 10 + k * 2 + 1] = py + dy * var_landmark * ph;
        }
    }
}
