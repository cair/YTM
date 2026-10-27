#ifdef IS_NEOVIM_CLANGD_ENV
#include "common.h"
#include "pack_clauses.h"
#endif

static inline PositionResult scan_position_literals(const TA_STATE_T* restrict ta_state, int full) {
    int pos0 = 0, pos1 = N_PATCHES_Y - 1;
    int pos2 = 0, pos3 = N_PATCHES_X - 1;
    uint includes = 0;
    bool valid = true;

    for (int lit = 0; lit < N_POSITION_FEATS_Y; ++lit) {
        update_pos_bounds(ta_state, lit, 0, &pos0, &pos1, &includes);
        if (pos0 > pos1) {
            valid = false;
            if (!full)
                return (PositionResult){pos0, pos1, pos2, pos3, includes, false};
        }
    }

    for (int lit = 0; lit < N_POSITION_FEATS_X; ++lit) {
        update_pos_bounds(ta_state, lit, N_POSITION_FEATS_Y, &pos2, &pos3, &includes);
        if (pos2 > pos3) {
            valid = false;
            if (!full)
                return (PositionResult){pos0, pos1, pos2, pos3, includes, false};
        }
    }

    return (PositionResult){pos0, pos1, pos2, pos3, includes, valid};
}

static inline FeatureResult scan_feature_literals(const TA_STATE_T* restrict ta_state, const FBOUND_T* restrict therm_bits,
                                                  const NLITS_T* restrict literal_offsets, NFEAT_T* restrict feat_ids,
                                                  FBOUND_T* restrict feat_bounds, int dont_skip) {
    uint n_includes = 0;
    int write_offset = 0;
    bool all_valid = true;

    for (int fid = 0; fid < N_RAW_PATCH_FEATS; ++fid) {
        FeatScan fs = calc_single_feat_bounds(ta_state, therm_bits, literal_offsets, fid);

        n_includes += fs.includes;
        if (fs.is_bounded) {
            feat_ids[write_offset] = (NFEAT_T)fid;
            feat_bounds[write_offset * 2 + 0] = fs.lb;
            feat_bounds[write_offset * 2 + 1] = fs.ub;
            write_offset++;
        }

        if (fs.lb > fs.ub) {
            all_valid = false;
            if (!dont_skip)
                break;
        }
    }

    return (FeatureResult){write_offset, n_includes, all_valid};
}

static inline FeatureResult scan_feature_bits(const TA_STATE_T* restrict ta_state, uint64_t* restrict pos_bits,
                                              uint64_t* restrict neg_bits, int dont_skip) {
    uint n_includes = 0;
    int n_bounded = 0;
    bool all_valid = true;

    for (int k = 0; k < N_FEAT_CHUNKS; k++) {
        const int base = k * FEAT_CHUNK_SIZE;
        const int n = N_RAW_PATCH_FEATS - base < FEAT_CHUNK_SIZE ? N_RAW_PATCH_FEATS - base : FEAT_CHUNK_SIZE;
        uint64_t pos = 0, neg = 0;

        for (int b = 0; b < n; b++) {
            pos |= (uint64_t)is_included(ta_state[N_POSITION_FEATS + base + b]) << b;
            if (NEGATED_LITERALS)
                neg |= (uint64_t)is_included(ta_state[N_POSITION_FEATS + N_LITERALS / 2 + base + b]) << b;
        }

        pos_bits[k] = pos;
        neg_bits[k] = neg;
        n_includes += (uint)(__builtin_popcountll(pos) + __builtin_popcountll(neg));
        n_bounded += __builtin_popcountll(pos | neg);

        if (pos & neg) {
            all_valid = false;
            if (!dont_skip)
                break;
        }
    }

    return (FeatureResult){n_bounded, n_includes, all_valid};
}

void pack_clauses(const TA_STATE_T* restrict global_ta_states, const FBOUND_T* restrict therm_bits,
                  const NLITS_T* restrict literal_offsets, PBOUND_T* restrict clause_position_bounds,
                  FBOUND_T* restrict clause_feat_bounds, NFEAT_T* restrict clause_feat_ids, NFEAT_T* restrict clause_n_feats,
                  int8_t* restrict has_contra, NLITS_T* restrict clause_len, uint64_t* restrict clause_pos_bits,
                  uint64_t* restrict clause_neg_bits, int8_t* restrict is_clause_synced, int dont_skip) {
#pragma omp parallel for schedule(dynamic) num_threads(ytm_n_threads)
    for (ull clause = 0; clause < (ull)TOTAL_CLAUSES; clause++) {
        if (is_clause_synced[clause])
            continue;

        const TA_STATE_T* ta_state = &global_ta_states[ta_offset(clause, 0)];

        PositionResult pr = {0, N_PATCHES_Y - 1, 0, N_PATCHES_X - 1, 0, true};
        if (POSITION_LITERALS)
            pr = scan_position_literals(ta_state, dont_skip);

        if (N_PATCHES > 1) {
            PBOUND_T* pos = &clause_position_bounds[pos_bounds_offset(clause, 0)];
            pos[0] = pr.pos0;
            pos[1] = pr.pos1;
            pos[2] = pr.pos2;
            pos[3] = pr.pos3;
        }

        if (!pr.valid && !dont_skip) {
            has_contra[clause] = 1;
            clause_len[clause] = 0;
            is_clause_synced[clause] = 1;
            continue;
        }

        FeatureResult fr;
        if (ALL_BINARY_FEATS) {
            fr = scan_feature_bits(ta_state, &clause_pos_bits[clause * N_FEAT_CHUNKS], &clause_neg_bits[clause * N_FEAT_CHUNKS],
                                   dont_skip);
        } else {
            NFEAT_T* cfids = &clause_feat_ids[feat_ids_offset(clause, 0)];
            FBOUND_T* cfb = &clause_feat_bounds[feat_bounds_offset(clause, 0, 0)];
            fr = scan_feature_literals(ta_state, therm_bits, literal_offsets, cfids, cfb, dont_skip);
            clause_n_feats[clause] = (NFEAT_T)fr.n_bounded_feats;
        }

        has_contra[clause] = !(pr.valid && fr.all_valid);
        clause_len[clause] = (NLITS_T)(pr.includes + fr.includes);
        is_clause_synced[clause] = 1;
    }
}
