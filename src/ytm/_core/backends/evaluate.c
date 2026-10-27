#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "rng.h"
#endif

// Output of a clause that needs no matching: 0 with a contradiction, 1 when empty, -1 when it must be matched.
static inline int eval_trivial(ull clause, const int8_t* restrict has_contra, const NLITS_T* restrict clause_len) {
    if (has_contra[clause])
        return 0;
    if (clause_len[clause] == 0)
        return 1;
    return -1;
}

static inline bool match_bits(const uint64_t* restrict x, const uint64_t* restrict pos, const uint64_t* restrict neg,
                              const uint64_t* restrict drop) {
    int k = 0;
    for (; k + 4 <= N_FEAT_CHUNKS; k += 4) {
        uint64_t miss = 0;
        for (int u = 0; u < 4; u++)
            miss |= ((pos[k + u] & ~x[k + u]) | (neg[k + u] & x[k + u])) & (drop ? ~drop[k + u] : ~0ULL);
        if (miss)
            return false;
    }
    for (; k < N_FEAT_CHUNKS; k++)
        if (((pos[k] & ~x[k]) | (neg[k] & x[k])) & (drop ? ~drop[k] : ~0ULL))
            return false;
    return true;
}

static inline bool match_clause_patch(const FBOUND_T* restrict Xe, const uint64_t* restrict xe_bits, int py, int px,
                                      const FBOUND_T* restrict feat_bounds, const NFEAT_T* restrict feat_ids, int n_feats,
                                      const uint64_t* restrict pos, const uint64_t* restrict neg,
                                      const uint64_t* restrict drop_bits, const uint8_t* restrict lit_drop_mask,
                                      const NLITS_T* restrict literal_offsets) {
    if (ALL_BINARY_FEATS)
        return match_bits(&xe_bits[(ull)(py * N_PATCHES_X + px) * N_FEAT_CHUNKS], pos, neg, drop_bits);
    return match_patch(Xe, py, px, feat_ids, feat_bounds, n_feats, lit_drop_mask, literal_offsets);
}

static inline bool match_window(const FBOUND_T* restrict Xe, const uint64_t* restrict xe_bits, const PBOUND_T* restrict pos,
                                const FBOUND_T* restrict feat_bounds, const NFEAT_T* restrict feat_ids, int n_feats,
                                const uint64_t* restrict pos_bits, const uint64_t* restrict neg_bits) {
    for (int py = pos[0]; py <= pos[1]; py++)
        for (int px = pos[2]; px <= pos[3]; px++)
            if (match_clause_patch(Xe, xe_bits, py, px, feat_bounds, feat_ids, n_feats, pos_bits, neg_bits, NULL, NULL, NULL))
                return true;
    return false;
}

/*
 * Output of every clause on every patch, for N samples.
 *
 * Arguments:
 *   X                       features of the batch
 *   x_bits                  (N, N_PATCHES, N_FEAT_CHUNKS) bit packed X, all binary features only
 *   N                       number of samples in the batch
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   clause_pos_bits         per clause included feature bits, all binary features only
 *   clause_neg_bits         per clause included negated feature bits, all binary features only
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   patch_output            output, (N, TOTAL_CLAUSES, N_PATCHES) clause outputs per patch
 */
void eval_patchwise(const FBOUND_T* restrict X, const uint64_t* restrict x_bits, const int N,
                    const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds,
                    const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
                    const uint64_t* restrict clause_pos_bits, const uint64_t* restrict clause_neg_bits,
                    const int8_t* restrict has_contra, const NLITS_T* restrict clause_len, int8_t* restrict patch_output) {
#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (ull idx = 0; idx < (ull)N * TOTAL_CLAUSES; idx++) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;
        int8_t* output = &patch_output[patch_output_offset(e, clause, 0)];

        // Trivial checks
        int trivial = eval_trivial(clause, has_contra, clause_len);
        if (trivial >= 0) {
            for (int patch = 0; patch < N_PATCHES; patch++)
                output[patch] = (int8_t)trivial;
            continue;
        }

        const FBOUND_T* Xe = &X[sample_offset(e)];
        const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
        const FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* bounded_fids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        int n_bounded_fids = (int)clause_n_feats[clause];
        const uint64_t* xe_bits = ALL_BINARY_FEATS ? &x_bits[e * N_PATCHES * N_FEAT_CHUNKS] : NULL;
        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause * N_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause * N_FEAT_CHUNKS] : NULL;

        for (int patch = 0; patch < N_PATCHES; patch++) {
            int py = patch / N_PATCHES_X;
            int px = patch % N_PATCHES_X;

            // If position bounds contradict, then it cant match any position.
            if (N_PATCHES > 1 && (py < pos[0] || py > pos[1] || px < pos[2] || px > pos[3])) {
                output[patch] = 0;
                continue;
            }

            // match a patch
            output[patch] =
                match_clause_patch(Xe, xe_bits, py, px, cfb, bounded_fids, n_bounded_fids, pos_bits, neg_bits, NULL, NULL, NULL);
        }
    }
}

/*
 * Clause outputs of N samples at inference.
 *
 * Arguments:
 *   X                       features of the batch
 *   x_bits                  (N, N_PATCHES, N_FEAT_CHUNKS) bit packed X, all binary features only
 *   N                       number of samples in the batch
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   clause_pos_bits         per clause included feature bits, all binary features only
 *   clause_neg_bits         per clause included negated feature bits, all binary features only
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   clause_outputs          output, (N, TOTAL_CLAUSES) clause outputs
 */
void eval_infer(const FBOUND_T* restrict X, const uint64_t* restrict x_bits, const int N,
                const PBOUND_T* restrict clause_position_bounds, const FBOUND_T* restrict clause_feat_bounds,
                const NFEAT_T* restrict clause_feat_ids, const NFEAT_T* restrict clause_n_feats,
                const uint64_t* restrict clause_pos_bits, const uint64_t* restrict clause_neg_bits,
                const int8_t* restrict has_contra, const NLITS_T* restrict clause_len, int8_t* restrict clause_outputs) {
#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (ull idx = 0; idx < (ull)N * TOTAL_CLAUSES; idx++) {
        ull e = idx / (ull)TOTAL_CLAUSES;
        ull clause = idx % (ull)TOTAL_CLAUSES;

        // Trivial checks
        int out = eval_trivial(clause, has_contra, clause_len);
        if (out < 0) {
            const FBOUND_T* Xe = &X[sample_offset(e)];
            const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
            const NFEAT_T* feat_ids = &clause_feat_ids[feat_ids_offset(clause, 0)];
            const int n_feats = (int)clause_n_feats[clause];
            const uint64_t* xe_bits = ALL_BINARY_FEATS ? &x_bits[e * N_PATCHES * N_FEAT_CHUNKS] : NULL;
            const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause * N_FEAT_CHUNKS] : NULL;
            const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause * N_FEAT_CHUNKS] : NULL;

            if (N_PATCHES > 1) {
                // Conv
                const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
                out = match_window(Xe, xe_bits, pos, feat_bounds, feat_ids, n_feats, pos_bits, neg_bits);
            } else {
                // Non conv
                out = match_clause_patch(Xe, xe_bits, 0, 0, feat_bounds, feat_ids, n_feats, pos_bits, neg_bits, NULL, NULL, NULL);
            }
        }

        clause_outputs[idx] = (int8_t)out;
    }
}

static inline void bit_pack_sample(const FBOUND_T* restrict Xe, uint64_t* restrict xe_bits) {
    for (int py = 0; py < N_PATCHES_Y; py++) {
        for (int px = 0; px < N_PATCHES_X; px++) {
            uint64_t* bits = &xe_bits[(ull)(py * N_PATCHES_X + px) * N_FEAT_CHUNKS];

            for (int k = 0; k < N_FEAT_CHUNKS; k++) {
                const int base = k * FEAT_CHUNK_SIZE;
                const int n = N_RAW_PATCH_FEATS - base < FEAT_CHUNK_SIZE ? N_RAW_PATCH_FEATS - base : FEAT_CHUNK_SIZE;
                uint64_t word = 0;
                for (int b = 0; b < n; b++)
                    word |= (uint64_t)(get_feature_value(Xe, py, px, base + b) != 0) << b;
                bits[k] = word;
            }
        }
    }
}

void bit_pack_X(const FBOUND_T* restrict X, const int N, uint64_t* restrict x_bits) {
#pragma omp parallel for schedule(static) if (N > 1) num_threads(ytm_n_threads)
    for (int e = 0; e < N; e++)
        bit_pack_sample(&X[sample_offset(e)], &x_bits[(ull)e * N_PATCHES * N_FEAT_CHUNKS]);
}

/*
 * Clause outputs of the current sample during fit. Use reservior sampling to select one matching patch.
 *
 * Arguments:
 *   seed                    RNG key of the current sample
 *   X                       features of the batch
 *   e                       index of the current sample
 *   clause_drop_mask        1 for clauses dropped this epoch
 *   class_active            1 for classes that get feedback on this sample
 *   lit_drop_mask           1 for literals that get no feedback
 *   feat_drop_bits          bit set for skipped features, all binary features only
 *   literal_offsets         first literal of each patch feature
 *   x_bits                  output, (N_PATCHES, N_FEAT_CHUNKS) bit packed current sample, all binary features only
 *   clause_position_bounds  per clause [y_lo, y_hi, x_lo, x_hi] of allowed patches
 *   clause_feat_bounds      per clause [lo, hi] of each bounded feature
 *   clause_feat_ids         per clause ids of its bounded features
 *   clause_n_feats          number of bounded features per clause
 *   clause_pos_bits         per clause included feature bits, all binary features only
 *   clause_neg_bits         per clause included negated feature bits, all binary features only
 *   has_contra              1 for clauses with a contradiction
 *   clause_len              number of included literals per clause
 *   clause_output           output, clause outputs
 *   selected_patch_ids      output, picked patch per firing clause (patches only)
 *   patch_weights           output, per clause and patch pick counts when TRACK_PATCH_WEIGHTS
 */
void eval_fit(const ull* restrict rng_keys, const FBOUND_T* restrict X, const int e, const int8_t* restrict clause_drop_mask,
              const uint8_t* restrict class_active, const uint8_t* restrict lit_drop_mask,
              const uint64_t* restrict feat_drop_bits, const NLITS_T* restrict literal_offsets,
              uint64_t* restrict x_bits, const PBOUND_T* restrict clause_position_bounds,
              const FBOUND_T* restrict clause_feat_bounds, const NFEAT_T* restrict clause_feat_ids,
              const NFEAT_T* restrict clause_n_feats, const uint64_t* restrict clause_pos_bits,
              const uint64_t* restrict clause_neg_bits, const int8_t* restrict has_contra,
              const NLITS_T* restrict clause_len, int8_t* restrict clause_output,
              NPATCHES_T* restrict selected_patch_ids, int* restrict patch_weights) {
    const ull seed = rng_keys[e];
    const FBOUND_T* Xe = &X[sample_offset(e)];
    if (ALL_BINARY_FEATS)
        bit_pack_sample(Xe, x_bits);

#pragma omp parallel for schedule(static) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        // trivial checks
        bool skip = clause_drop_mask[clause] == 1 || (!COALESCED && !class_active[clause / CLAUSES_PER_CLASS]);
        int out = skip ? 0 : eval_trivial(clause, has_contra, clause_len);
        int selected_id = -1;

        const FBOUND_T* feat_bounds = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
        const NFEAT_T* feat_ids = &clause_feat_ids[feat_ids_offset(clause, 0)];
        const int n_feats = (int)clause_n_feats[clause];
        const uint64_t* pos_bits = ALL_BINARY_FEATS ? &clause_pos_bits[clause * N_FEAT_CHUNKS] : NULL;
        const uint64_t* neg_bits = ALL_BINARY_FEATS ? &clause_neg_bits[clause * N_FEAT_CHUNKS] : NULL;

        if (N_PATCHES > 1) {
            ull rng_k = rng_hash(seed, clause, 0xDEADBEEFULL);
            uint rng_counter = 0;

            if (out == 1) {
                // Empty clause: any patch.
                selected_id = (int)rand_int(rng_k, &rng_counter, N_PATCHES);
            } else if (out < 0) {
                const PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
                int count = 0;
                for (int py = pos[0]; py <= pos[1]; py++) {
                    for (int px = pos[2]; px <= pos[3]; px++) {
                        if (!match_clause_patch(Xe, x_bits, py, px, feat_bounds, feat_ids, n_feats, pos_bits, neg_bits,
                                                feat_drop_bits, lit_drop_mask, literal_offsets))
                            continue;
                        count++;
                        if (rand_int(rng_k, &rng_counter, (uint)count) == 0)
                            selected_id = py * N_PATCHES_X + px;
                    }
                }
                out = (selected_id >= 0);
            }
        } else {
            if (out < 0)
                out = match_clause_patch(Xe, x_bits, 0, 0, feat_bounds, feat_ids, n_feats, pos_bits, neg_bits, feat_drop_bits,
                                         lit_drop_mask, literal_offsets);
        }

        clause_output[clause] = (int8_t)out;
        if (N_PATCHES > 1 && out == 1) {
            selected_patch_ids[clause] = (NPATCHES_T)selected_id;
            if (TRACK_PATCH_WEIGHTS)
                patch_weights[patch_weights_offset(clause, selected_id)]++;
        }
    }
}

static inline float vote_dot(const int8_t* restrict out, const float* restrict w, int n) {
    float acc[16] = {0.0f};
    int j = 0;
    for (; j + 16 <= n; j += 16)
        for (int k = 0; k < 16; k++)
            acc[k] += (float)out[j + k] * w[j + k];
    float s = 0.0f;
    for (; j < n; j++)
        s += (float)out[j] * w[j];
    for (int k = 0; k < 16; k++)
        s += acc[k];
    return s;
}

/*
 * Class sums of N samples.
 *
 * Arguments:
 *   clause_outputs  (N, TOTAL_CLAUSES) clause outputs
 *   clause_weights  clause weights per class
 *   N               number of samples
 *   class_sums      output, (N, CLASSES)
 */
void vote_sum(const int8_t* restrict clause_outputs, const float* restrict clause_weights, const int N,
              float* restrict class_sums) {
#pragma omp parallel for schedule(static) if (N > 1) num_threads(ytm_n_threads)
    for (int e = 0; e < N; e++) {
        const int8_t* out_e = &clause_outputs[clause_output_offset(e, 0)];
        float* sums_e = &class_sums[(ull)e * CLASSES];
        for (int class_id = 0; class_id < CLASSES; class_id++) {
            ull first = COALESCED ? 0 : (ull)class_id * CLAUSES_PER_CLASS;
            sums_e[class_id] =
                vote_dot(&out_e[first], &clause_weights[weight_offset(class_id, first)], CLAUSES_PER_CLASS);
        }
    }
}
