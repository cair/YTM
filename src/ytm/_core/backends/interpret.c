#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#endif

#include <math.h>

static inline void add_clause_feats(float* restrict out, int py, int px, float wm, const FBOUND_T* restrict cfb,
                                    const NFEAT_T* restrict cfids, int n_feats, const uint64_t* restrict pos_bits,
                                    const uint64_t* restrict neg_bits, const FBOUND_T* restrict therm_bits) {
    if (ALL_BINARY_FEATS) {
        for (int k = 0; k < N_BIN_FEAT_CHUNKS; k++) {
            for (uint64_t w = pos_bits[k]; w; w &= w - 1)
                out[feature_offset(k * BIN_FEAT_CHUNK_SZ + __builtin_ctzll(w), py, px)] += wm;
            for (uint64_t w = neg_bits[k]; w; w &= w - 1)
                out[feature_offset(k * BIN_FEAT_CHUNK_SZ + __builtin_ctzll(w), py, px)] -= wm;
        }
        return;
    }
    for (int i = 0; i < n_feats; i++) {
        int k = (int)cfids[i];
        float cp = (float)((int)cfb[i * 2] + (int)cfb[i * 2 + 1] - (int)therm_bits[k]);
        out[feature_offset(k, py, px)] += cp * wm;
    }
}

void wic(int class_id, int polarity, const float* restrict clause_weights, const FBOUND_T* restrict clause_feat_bounds,
         const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
         const uint64_t* restrict clause_pos_bits, const uint64_t* restrict clause_neg_bits,
         const PBOUND_T* restrict clause_position_bounds, const int8_t* restrict has_contra, const float* restrict patch_weights_norm,
         const FBOUND_T* restrict therm_bits, float pw_th, float* restrict output) {
#pragma omp parallel for schedule(static) reduction(+ : output[ : HEIGHT * WIDTH * DEPTH]) num_threads(ytm_n_threads)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
        if (!COALESCED && (ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        const FBOUND_T* cfb = ALL_BINARY_FEATS ? NULL : &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = ALL_BINARY_FEATS ? NULL : &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = ALL_BINARY_FEATS ? 0 : (int)clause_n_feats[clause_id];
        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;

        if (N_PATCHES == 1) {
            float wm = fabsf(w);
            add_clause_feats(output, 0, 0, wm, cfb, cfids, n_feats, pos_bits, neg_bits, therm_bits);
            continue;
        }

        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause_id, 0)];
        for (int py = pos[0]; py <= pos[1]; py++) {
            for (int px = pos[2]; px <= pos[3]; px++) {
                int p = py * N_PATCHES_X + px;
                float pwv = patch_weights_norm[patch_weights_offset(clause_id, p)];
                if (pwv <= pw_th)
                    continue;

                float wm = fabsf(w) * pwv;

                add_clause_feats(output, py, px, wm, cfb, cfids, n_feats, pos_bits, neg_bits, therm_bits);
            }
        }
    }
}

void wac_sample(int class_id, int polarity, const int8_t* restrict patch_output, const int e, const float* restrict clause_weights,
                const FBOUND_T* restrict clause_feat_bounds, const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
                const uint64_t* restrict clause_pos_bits, const uint64_t* restrict clause_neg_bits, const int8_t* restrict has_contra, const FBOUND_T* restrict therm_bits,
                float* restrict output) {
    const int8_t* patch_e = &patch_output[patch_output_offset(e, 0, 0)];
    float* out_e = &output[sample_offset(e)];

#pragma omp parallel for schedule(static) reduction(+ : out_e[ : HEIGHT * WIDTH * DEPTH]) num_threads(ytm_n_threads)
    for (ull clause_id = 0; clause_id < (ull)TOTAL_CLAUSES; clause_id++) {
        if (!COALESCED && (ull)class_id != clause_id / (ull)CLAUSES_PER_CLASS)
            continue;
        if (has_contra[clause_id])
            continue;

        float w = clause_weights[weight_offset(class_id, clause_id)];
        if ((polarity > 0 && w <= 0.0f) || (polarity < 0 && w >= 0.0f))
            continue;

        const int8_t* clause_act = &patch_e[clause_id * (ull)N_PATCHES];
        const FBOUND_T* cfb = ALL_BINARY_FEATS ? NULL : &clause_feat_bounds[feat_bounds_offset(clause_id, 0, 0)];
        const NFEAT_T* cfids = ALL_BINARY_FEATS ? NULL : &clause_feat_ids[feat_ids_offset(clause_id, 0)];
        const int n_feats = ALL_BINARY_FEATS ? 0 : (int)clause_n_feats[clause_id];
        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause_id * N_BIN_FEAT_CHUNKS] : NULL;
        float wm = fabsf(w);

        for (int p = 0; p < N_PATCHES; p++) {
            if (clause_act[p] <= 0)
                continue;

            int py = p / N_PATCHES_X, px = p % N_PATCHES_X;

            add_clause_feats(out_e, py, px, wm, cfb, cfids, n_feats, pos_bits, neg_bits, therm_bits);
        }
    }
}
