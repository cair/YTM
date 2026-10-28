#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "cuda.h"
#endif

__device__ inline void add_clause_feats(float* __restrict__ out, int py, int px, float wm, const FBOUND_T* __restrict__ cfb,
                                        const NFEAT_T* __restrict__ cfids, int n_feats,
                                        const uint64_t* __restrict__ pos_bits, const uint64_t* __restrict__ neg_bits,
                                        const FBOUND_T* __restrict__ therm_bits) {
    if (ALL_BINARY_FEATS) {
        for (int k = 0; k < N_BIN_FEAT_CHUNKS; k++) {
            for (uint64_t w = pos_bits[k]; w; w &= w - 1)
                atomicAdd(&out[feature_offset(k * BIN_FEAT_CHUNK_SZ + __ffsll(w) - 1, py, px)], wm);
            for (uint64_t w = neg_bits[k]; w; w &= w - 1)
                atomicAdd(&out[feature_offset(k * BIN_FEAT_CHUNK_SZ + __ffsll(w) - 1, py, px)], -wm);
        }
        return;
    }
    for (int i = 0; i < n_feats; ++i) {
        int k = (int)cfids[i];
        float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
        atomicAdd(&out[feature_offset(k, py, px)], cp * wm);
    }
}

extern "C" __global__ void wic(int class_id, int polarity, const float* __restrict__ clause_weights,
                               const FBOUND_T* __restrict__ clause_feat_bounds,
                               const NFEAT_T* __restrict__ clause_feat_ids, const NFEAT_T* __restrict__ clause_n_feats,
                               const uint64_t* __restrict__ clause_pos_bits, const uint64_t* __restrict__ clause_neg_bits,
                               const PBOUND_T* __restrict__ clause_position_bounds,
                               const int8_t* __restrict__ has_contra, const float* __restrict__ patch_weights_norm,
                               const FBOUND_T* __restrict__ therm_bits, float pw_th, float* __restrict__ output) {
    GRID_STRIDE_LOOP(idx, (ull)TOTAL_CLAUSES * N_PATCHES) {
        ull clause_id = idx / N_PATCHES;
        int p = idx % N_PATCHES;

        if (!COALESCED && (ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;

        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

        float pwv = 1.0f;
        if (N_PATCHES > 1) {
            const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause_id, 0)];
            if (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3])
                continue;

            pwv = patch_weights_norm[patch_weights_offset(clause_id, p)];
            if (pwv <= pw_th)
                continue;
        }

        float wm = fabsf(w) * pwv;
        const FBOUND_T* cfb = ALL_BINARY_FEATS ? NULL : &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = ALL_BINARY_FEATS ? NULL : &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = ALL_BINARY_FEATS ? 0 : (int)clause_n_feats[clause_id];
        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;

        add_clause_feats(output, py, px, wm, cfb, cfids, n_feats, pos_bits, neg_bits, therm_bits);
    }
}

extern "C" __global__ void wac(const int* __restrict__ target_classes, int polarity, int N,
                               const float* __restrict__ clause_weights,
                               const FBOUND_T* __restrict__ clause_feat_bounds,
                               const NFEAT_T* __restrict__ clause_feat_ids, const NFEAT_T* __restrict__ clause_n_feats,
                               const uint64_t* __restrict__ clause_pos_bits, const uint64_t* __restrict__ clause_neg_bits,
                               const int8_t* __restrict__ patch_output, const int8_t* __restrict__ has_contra,
                               const FBOUND_T* __restrict__ therm_bits, float* __restrict__ output) {
    GRID_STRIDE_LOOP(idx, (ull)N * TOTAL_CLAUSES * N_PATCHES) {
        ull e = idx / ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_patch = idx % ((ull)TOTAL_CLAUSES * N_PATCHES);
        ull clause_id = clause_patch / (ull)N_PATCHES;
        int p = clause_patch % N_PATCHES;

        int class_id = target_classes[e];

        if (!COALESCED && (ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;

        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        ull act_idx = patch_output_offset(e, clause_id, p);
        if (patch_output[act_idx] <= 0)
            continue;

        float wm = fabsf(w);
        const FBOUND_T* cfb = ALL_BINARY_FEATS ? NULL : &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = ALL_BINARY_FEATS ? NULL : &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = ALL_BINARY_FEATS ? 0 : (int)clause_n_feats[clause_id];
        float* out_e = &output[sample_offset(e)];

        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;

        int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

        add_clause_feats(out_e, py, px, wm, cfb, cfids, n_feats, pos_bits, neg_bits, therm_bits);
    }
}
