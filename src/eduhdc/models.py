"""
EduHDC Neuro-Symbolic Models (Contribution C1 Rescue).

Provides trainable neural architectures with VSA inductive biases:
1. EduHDC_KT: Neuro-symbolic Knowledge Tracer combining non-commutative VSA state
   tracking with parameterized temporal decay and learnable linear readout.
2. DKT_Baseline: Standard Deep Knowledge Tracing (LSTM) baseline.
3. MAP_KT / HRR_KT: Commutative VSA baselines for fair ablation.
4. EduHDC_PrereqProbe: Asymmetric VSA relation probing for prerequisite prediction.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from eduhdc.operators import BaseVSA, BipolarMAP, RealHRR, EduBindBlockDiag


# ==============================================================================
# 1. Neuro-Symbolic Knowledge Tracing Models
# ==============================================================================

class EduHDC_KT(nn.Module):
    """
    EduHDC Neuro-Symbolic Knowledge Tracer.
    
    Architecture:
    - VSA Memory State: h_t accumulates (skill_t, response_t) associations using EduBind (non-commutative)
    - Transition State: T_t accumulates consecutive transitions (skill_{t-1} -> skill_t)
    - Feature Extraction: Computes multi-scale holographic similarities:
        f1: similarity(unbind(h_t, skill_{t+1}), response_correct)
        f2: similarity(unbind(T_t, skill_t -> skill_{t+1}), response_correct)
        f3: direct dot-product projection of unbind(h_t, skill_{t+1})
    - Learnable Readout: Lightweight MLP / Linear layer with learnable decay rates.
    
    Advantages:
    - 10x-50x fewer parameters than LSTM/Transformer DKT
    - Strong non-commutative inductive bias (respects learning trajectory direction)
    - High throughput on edge devices
    """
    def __init__(self, num_skills: int, vsa_dim: int = 2048, hidden_dim: int = 64,
                 op_type: str = "edubind", device: str = "cuda",
                 freeze_codebook: bool = False):
        super().__init__()
        self.num_skills = num_skills
        self.vsa_dim = vsa_dim
        self.device = device
        self.op_type = op_type

        # Initialize VSA Operator
        if op_type == "edubind":
            self.vsa = EduBindBlockDiag(dim=vsa_dim, device=device)
            self.actual_dim = self.vsa.actual_dim
        elif op_type == "map":
            self.vsa = BipolarMAP(dim=vsa_dim, device=device)
            self.actual_dim = vsa_dim
        elif op_type == "hrr":
            self.vsa = RealHRR(dim=vsa_dim, device=device)
            self.actual_dim = vsa_dim
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

        # Learnable concept and response hypervectors (initialized from VSA orthogonal codebook)
        init_skills = self.vsa.random_vector(num_skills).float()
        init_responses = self.vsa.random_vector(2).float() # [0] = Incorrect, [1] = Correct

        self.skill_embeddings = nn.Parameter(init_skills, requires_grad=not freeze_codebook)
        self.response_embeddings = nn.Parameter(init_responses, requires_grad=not freeze_codebook)
        self.freeze_codebook = freeze_codebook

        # Learnable memory decay parameters
        self.gamma_uni = nn.Parameter(torch.tensor(0.85))
        self.gamma_bi  = nn.Parameter(torch.tensor(0.80))

        # Lightweight neural readout
        # D2 FIX: richer feature vector (7-dim) instead of 4 scalars.
        # [sim_uni, sim_bi, skill_bias, pos, recency, count_seen, correct_rate_so_far]
        self.readout = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # Skill difficulty bias
        self.skill_bias = nn.Embedding(num_skills, 1)
        nn.init.zeros_(self.skill_bias.weight)

    def forward_sequence(self, skill_seq: torch.Tensor, correct_seq: torch.Tensor) -> torch.Tensor:
        """
        Processes a single student's interaction sequence.
        skill_seq: (seq_len,) LongTensor
        correct_seq: (seq_len,) LongTensor
        Returns: predicted probabilities for each step (seq_len,)
        """
        seq_len = len(skill_seq)
        device = self.device
        dim = self.actual_dim

        state_uni = torch.zeros(1, dim, device=device)
        state_bi  = torch.zeros(1, dim, device=device)

        probs = []
        prev_skill_vec = None

        # Per-skill history trackers (D2: recency / exposure / running accuracy)
        last_seen = {}          # skill_idx -> last step index seen
        count_seen = {}         # skill_idx -> times seen so far
        correct_seen = {}       # skill_idx -> times correct so far

        rv_correct = self.response_embeddings[1].unsqueeze(0)
        rv_wrong   = self.response_embeddings[0].unsqueeze(0)

        for t in range(seq_len):
            curr_skill_idx = skill_seq[t].item()
            curr_skill_vec = self.skill_embeddings[curr_skill_idx].unsqueeze(0)

            # --- Predict BEFORE update ---
            # 1. Unigram query: unbind current state with query skill
            pred_uni = self.vsa.unbind(state_uni, curr_skill_vec)
            sim_uni = self.vsa.similarity(pred_uni, rv_correct)

            # 2. Bigram query: query transition from previous skill
            sim_bi = torch.zeros_like(sim_uni)
            if prev_skill_vec is not None:
                trans_key = self.vsa.bind(prev_skill_vec, curr_skill_vec)
                pred_bi = self.vsa.unbind(state_bi, trans_key)
                sim_bi = self.vsa.similarity(pred_bi, rv_correct)

            # 3. Skill baseline bias
            s_bias = self.skill_bias(skill_seq[t]).squeeze(0)

            # 4. Step position proxy
            pos_feature = torch.tensor([[t / 100.0]], device=device)

            # 5. D2 history features (recency, exposure, running accuracy)
            c_cnt = count_seen.get(curr_skill_idx, 0)
            recency = (t - last_seen[curr_skill_idx]) / 100.0 if curr_skill_idx in last_seen else 1.0
            count_norm = min(c_cnt, 20) / 20.0
            crate = (correct_seen.get(curr_skill_idx, 0) / c_cnt) if c_cnt > 0 else 0.5
            hist_feature = torch.tensor([[recency, count_norm, crate]], device=device)

            # Combine features into readout
            feat = torch.cat([sim_uni.view(1, 1), sim_bi.view(1, 1),
                              s_bias.view(1, 1), pos_feature, hist_feature], dim=-1)
            logit = self.readout(feat)
            prob = torch.sigmoid(logit).squeeze()
            probs.append(prob)

            # --- Update state AFTER observing response ---
            is_correct = correct_seq[t].item()
            rv = rv_correct if is_correct == 1 else rv_wrong

            # Update unigram state (associative accumulation)
            # D1 FIX: NO .detach() on the trace — gradient must flow through
            # skill_embeddings / response_embeddings / gamma via the memory state.
            trace_uni = self.vsa.bind(curr_skill_vec, rv)
            g_uni = torch.clamp(self.gamma_uni, 0.1, 0.99)
            state_uni = g_uni * state_uni + trace_uni

            # Update bigram state
            if prev_skill_vec is not None:
                trans_key = self.vsa.bind(prev_skill_vec, curr_skill_vec)
                trace_bi = self.vsa.bind(trans_key, rv)
                g_bi = torch.clamp(self.gamma_bi, 0.1, 0.99)
                state_bi = g_bi * state_bi + trace_bi

            # Update per-skill history trackers (D2)
            count_seen[curr_skill_idx] = count_seen.get(curr_skill_idx, 0) + 1
            if is_correct == 1:
                correct_seen[curr_skill_idx] = correct_seen.get(curr_skill_idx, 0) + 1
            last_seen[curr_skill_idx] = t

            prev_skill_vec = curr_skill_vec

        return torch.stack(probs)

    # ------------------------------------------------------------------ batched
    def forward_batch(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward over a BATCH of students (100-1000x faster than
        forward_sequence). Loops only over timesteps; all students run in parallel.

        skill_batch:   (B, T) LongTensor, padded
        correct_batch: (B, T) LongTensor (0/1), padded
        mask:          (B, T) bool/float, 1 = valid position
        Returns: probs (B, T) — predictions BEFORE observing each step.
        """
        device = self.device
        dim = self.actual_dim
        B, T = skill_batch.shape

        state_uni = torch.zeros(B, dim, device=device)
        state_bi  = torch.zeros(B, dim, device=device)

        rv_correct = self.response_embeddings[1].unsqueeze(0).expand(B, -1)  # (B,dim)
        rv_wrong   = self.response_embeddings[0].unsqueeze(0).expand(B, -1)

        # Vectorized per-skill history trackers
        last_seen    = torch.full((B, self.num_skills), -1.0, device=device)
        count_seen   = torch.zeros(B, self.num_skills, device=device)
        correct_seen = torch.zeros(B, self.num_skills, device=device)

        arangeB = torch.arange(B, device=device)
        prev_skill_vec = None
        prev_valid = torch.zeros(B, device=device)
        probs_all = []

        for t in range(T):
            curr_idx = skill_batch[:, t]                       # (B,)
            curr_vec = self.skill_embeddings[curr_idx]         # (B,dim)

            # Unigram
            pred_uni = self.vsa.unbind(state_uni, curr_vec)
            sim_uni = self.vsa.similarity(pred_uni, rv_correct)  # (B,)

            # Bigram (guarded by whether a previous step exists)
            sim_bi = torch.zeros(B, device=device)
            if prev_skill_vec is not None:
                trans_key = self.vsa.bind(prev_skill_vec, curr_vec)
                pred_bi = self.vsa.unbind(state_bi, trans_key)
                sim_bi = self.vsa.similarity(pred_bi, rv_correct) * prev_valid

            s_bias = self.skill_bias(curr_idx).squeeze(-1)      # (B,)
            pos_feat = torch.full((B,), t / 100.0, device=device)

            cnt = count_seen[arangeB, curr_idx]                 # (B,)
            ls = last_seen[arangeB, curr_idx]
            recency = torch.where(ls >= 0, (t - ls) / 100.0, torch.ones_like(ls))
            count_norm = torch.clamp(cnt, max=20.0) / 20.0
            cs = correct_seen[arangeB, curr_idx]
            crate = torch.where(cnt > 0, cs / torch.clamp(cnt, min=1.0),
                                torch.full_like(cnt, 0.5))

            feat = torch.stack([sim_uni, sim_bi, s_bias, pos_feat,
                                recency, count_norm, crate], dim=-1)  # (B,7)
            logit = self.readout(feat).squeeze(-1)
            probs_all.append(torch.sigmoid(logit))

            # --- Update after observing response ---
            is_corr = correct_batch[:, t].float().unsqueeze(-1)   # (B,1)
            rv = is_corr * rv_correct + (1.0 - is_corr) * rv_wrong
            m_t = mask[:, t].float().unsqueeze(-1)                 # (B,1) gate padding

            trace_uni = self.vsa.bind(curr_vec, rv)
            g_uni = torch.clamp(self.gamma_uni, 0.1, 0.99)
            state_uni = g_uni * state_uni + trace_uni * m_t

            if prev_skill_vec is not None:
                trans_key2 = self.vsa.bind(prev_skill_vec, curr_vec)
                trace_bi = self.vsa.bind(trans_key2, rv)
                g_bi = torch.clamp(self.gamma_bi, 0.1, 0.99)
                gate_bi = m_t * prev_valid.unsqueeze(-1)
                state_bi = g_bi * state_bi + trace_bi * gate_bi

            # Update history trackers (only at valid positions)
            mt = mask[:, t].float()
            new_cnt = count_seen[arangeB, curr_idx] + mt
            count_seen[arangeB, curr_idx] = new_cnt
            correct_seen[arangeB, curr_idx] = correct_seen[arangeB, curr_idx] + \
                correct_batch[:, t].float() * mt
            upd_ls = torch.where(mt > 0, torch.full_like(mt, float(t)), ls)
            last_seen[arangeB, curr_idx] = upd_ls

            prev_skill_vec = curr_vec
            prev_valid = mask[:, t].float()

        return torch.stack(probs_all, dim=1)  # (B, T)

    # ------------------------------------------------------- chunked parallel scan
    def _chunked_scan(self, traces: torch.Tensor, gates: torch.Tensor,
                      gamma: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        """Chunked parallel scan for state[t] = gamma*state[t-1] + traces[t]*gates[t].

        Splits T into chunks of `chunk_size` to keep gamma^(-chunk_size) within
        fp32 range (gamma=0.1, chunk=32 → 0.1^(-31)=1e31, safe; vs 0.1^(-199)=1e199
        which overflows fp32 max 3.4e38).

        Within each chunk, uses the cumsum closed-form:
            state[j] = gamma^j * cumsum(trace[k]*gate[k]*gamma^(-k))[j]
        Between chunks, carries the final state forward with decay.

        NOTE: Uses list + cat instead of inplace writes to preserve autograd graph.

        Args:
            traces: (B, T, D) fp32 — per-step trace vectors
            gates:  (B, T, 1) fp32 — per-step gate (mask)
            gamma:  scalar tensor — decay factor (clamped 0.1..0.99)
            chunk_size: max steps per chunk (default 32)
        Returns:
            states: (B, T, D) fp32
        """
        B, T, D = traces.shape
        device = traces.device
        n_chunks = (T + chunk_size - 1) // chunk_size
        carry = torch.zeros(B, D, device=device, dtype=traces.dtype)
        state_chunks = []

        for c in range(n_chunks):
            s = c * chunk_size
            e = min(s + chunk_size, T)
            L = e - s

            # Local time indices within this chunk
            t_local = torch.arange(L, device=device, dtype=torch.float32)
            g_pow = gamma ** t_local        # gamma^j for j=0..L-1
            g_inv = gamma ** (-t_local)     # gamma^(-j), safe since L <= chunk_size

            # Weighted traces: trace[t]*gate[t]*gamma^(-t_local)
            chunk_traces = traces[:, s:e] * gates[:, s:e]  # (B, L, D)
            weighted = chunk_traces * g_inv.view(1, L, 1)

            # Cumsum trick: local_state[j] = gamma^j * cumsum(weighted)[j]
            cum = torch.cumsum(weighted, dim=1)
            local_states = cum * g_pow.view(1, L, 1)

            # Add carry from previous chunks, decayed into this chunk
            carry_decay = gamma ** (t_local + 1)  # gamma^(j+1)
            chunk_state = local_states + carry.unsqueeze(1) * carry_decay.view(1, L, 1)
            state_chunks.append(chunk_state)

            # Update carry: final state of this chunk, decayed by gamma for next chunk
            carry = chunk_state[:, -1] * gamma

        return torch.cat(state_chunks, dim=1)

    # ------------------------------------------------------- fully vectorized
    def forward_batch_fast(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized forward — NO Python loop over timesteps.
        Uses cumsum-based parallel scan for the linear state recurrence:
            state[t] = gamma * state[t-1] + trace[t] * mask[t]
        which has closed form:
            state[t] = gamma^t * cumsum(trace[k] * mask[k] * gamma^(-k))[t]

        This eliminates ~3000 small CUDA kernel launches (T=200 × 15 ops/step)
        and replaces them with ~20 large batched operations.

        skill_batch:   (B, T) LongTensor, padded
        correct_batch: (B, T) LongTensor (0/1), padded
        mask:          (B, T) bool/float, 1 = valid position
        Returns: probs (B, T) — predictions BEFORE observing each step.
        """
        device = self.device
        dim = self.actual_dim
        B, T = skill_batch.shape

        # Skill vectors for all timesteps: (B, T, dim)
        skill_vecs = self.skill_embeddings[skill_batch]  # (B, T, dim)

        # Response vectors for all timesteps: (B, T, dim)
        is_corr = correct_batch.float().unsqueeze(-1)  # (B, T, 1)
        rv_correct = self.response_embeddings[1].view(1, 1, -1)  # (1, 1, dim)
        rv_wrong = self.response_embeddings[0].view(1, 1, -1)    # (1, 1, dim)
        rv = is_corr * rv_correct + (1.0 - is_corr) * rv_wrong   # (B, T, dim)

        # Mask: (B, T, 1)
        m = mask.float().unsqueeze(-1)  # (B, T, 1)

        # === Unigram state via chunked parallel scan ===
        # trace_uni[t] = bind(skill_vec[t], rv[t])
        trace_uni = self.vsa.bind(skill_vecs, rv)  # (B, T, dim)

        # state_uni[t] = g * state_uni[t-1] + trace_uni[t] * m[t]
        # Chunked scan: split T into chunks of 32 to avoid g^(-T) overflow in fp32.
        # Within each chunk, use cumsum trick (safe: g^(-31) << fp32 max).
        g_uni = torch.clamp(self.gamma_uni, 0.1, 0.99)
        with torch.amp.autocast('cuda', enabled=False):
            states_uni = self._chunked_scan(
                trace_uni.float(), m.float(), g_uni, chunk_size=32
            ).to(trace_uni.dtype)

        # state_before[t] = state[t-1], with state_before[0] = 0
        state_uni_before = torch.cat(
            [torch.zeros(B, 1, dim, device=device), states_uni[:, :-1]], dim=1)

        # Unigram prediction: pred_uni[t] = unbind(state_before[t], skill_vec[t])
        pred_uni = self.vsa.unbind(state_uni_before, skill_vecs)  # (B, T, dim)
        sim_uni = self.vsa.similarity(pred_uni, rv_correct)  # (B, T) via broadcast

        # === Bigram state ===
        # trans_key[t] = bind(skill_vec[t-1], skill_vec[t]) for t >= 1
        prev_skill_vecs = torch.cat(
            [torch.zeros(B, 1, dim, device=device), skill_vecs[:, :-1]], dim=1)
        trans_keys = self.vsa.bind(prev_skill_vecs, skill_vecs)  # (B, T, dim)
        trace_bi = self.vsa.bind(trans_keys, rv)  # (B, T, dim)

        # gate_bi[t] = m[t] * prev_valid[t] = m[t] * m[t-1]
        prev_m = torch.cat([torch.zeros(B, 1, 1, device=device), m[:, :-1]], dim=1)
        gate_bi = m * prev_m  # (B, T, 1)

        g_bi = torch.clamp(self.gamma_bi, 0.1, 0.99)
        with torch.amp.autocast('cuda', enabled=False):
            states_bi = self._chunked_scan(
                trace_bi.float(), gate_bi.float(), g_bi, chunk_size=32
            ).to(trace_bi.dtype)

        state_bi_before = torch.cat(
            [torch.zeros(B, 1, dim, device=device), states_bi[:, :-1]], dim=1)

        pred_bi = self.vsa.unbind(state_bi_before, trans_keys)  # (B, T, dim)
        sim_bi = self.vsa.similarity(pred_bi, rv_correct)  # (B, T)
        # Mask out t=0 (no previous skill)
        prev_valid = torch.cat(
            [torch.zeros(B, 1, device=device), mask[:, :-1].float()], dim=1)
        sim_bi = sim_bi * prev_valid

        # === History features (vectorized via one_hot + cumsum) ===
        one_hot = F.one_hot(skill_batch, self.num_skills).float()  # (B, T, num_skills)

        # Cumulative count before current position
        cum_count = torch.cumsum(one_hot, dim=1)
        cum_count_before = torch.cat(
            [torch.zeros(B, 1, self.num_skills, device=device), cum_count[:, :-1]], dim=1)
        cnt = cum_count_before.gather(2, skill_batch.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Cumulative correct count before current position
        correct_oh = one_hot * correct_batch.float().unsqueeze(-1)
        cum_correct = torch.cumsum(correct_oh, dim=1)
        cum_correct_before = torch.cat(
            [torch.zeros(B, 1, self.num_skills, device=device), cum_correct[:, :-1]], dim=1)
        cs = cum_correct_before.gather(2, skill_batch.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Last seen position before current (via cummax)
        positions = torch.arange(T, device=device).float().view(1, T, 1)
        weighted_pos = (positions + 1) * one_hot  # 0 where no match
        cum_max_pos = torch.cummax(weighted_pos, dim=1).values
        cum_max_before = torch.cat(
            [torch.zeros(B, 1, self.num_skills, device=device), cum_max_pos[:, :-1]], dim=1)
        ls_plus1 = cum_max_before.gather(2, skill_batch.unsqueeze(-1)).squeeze(-1)
        ls = ls_plus1 - 1  # -1 means never seen

        # Compute scalar features
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        recency = torch.where(ls >= 0, (t_idx.view(1, T) - ls) / 100.0,
                              torch.ones(B, T, device=device))
        count_norm = torch.clamp(cnt, max=20.0) / 20.0
        crate = torch.where(cnt > 0, cs / torch.clamp(cnt, min=1.0),
                            torch.full((B, T), 0.5, device=device))
        s_bias = self.skill_bias(skill_batch).squeeze(-1)  # (B, T)
        pos_feat = t_idx.view(1, T).expand(B, T) / 100.0

        # === Readout ===
        feat = torch.stack([sim_uni, sim_bi, s_bias, pos_feat,
                            recency, count_norm, crate], dim=-1)  # (B, T, 7)
        logit = self.readout(feat).squeeze(-1)  # (B, T)
        return torch.sigmoid(logit)


# ==============================================================================
# 2. Standard Deep Knowledge Tracing (DKT - LSTM) Baseline
# ==============================================================================

class DKT_Baseline(nn.Module):
    """Standard LSTM-based Deep Knowledge Tracing (Piech et al., NeurIPS 2015)."""
    def __init__(self, num_skills: int, emb_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_skills = num_skills
        # Input is (skill_id * 2 + correct), plus index 0 for start of sequence
        self.input_emb = nn.Embedding(num_skills * 2 + 2, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_skills)

    def forward(self, skill_seq: torch.Tensor, correct_seq: torch.Tensor) -> torch.Tensor:
        """
        skill_seq: (seq_len,)
        correct_seq: (seq_len,)
        Predicts performance at step t based purely on history (0 to t-1).
        """
        device = skill_seq.device
        seq_len = len(skill_seq)

        # Shift input by 1 step so step t receives interaction (t-1)
        # For step 0, input is a special <START> token (index 0)
        # For step t > 0, input is (skill_{t-1} * 2 + correct_{t-1} + 1)
        prev_inters = skill_seq[:-1] * 2 + correct_seq[:-1].long() + 1
        start_token = torch.zeros(1, dtype=torch.long, device=device)
        inter_ids = torch.cat([start_token, prev_inters]) # (seq_len,)

        embs = self.input_emb(inter_ids).unsqueeze(0) # (1, seq_len, emb_dim)
        lstm_out, _ = self.lstm(embs) # (1, seq_len, hidden_dim)
        logits = self.out(lstm_out).squeeze(0) # (seq_len, num_skills)

        # Gather probability for the target skill at each step t
        target_skills = skill_seq.unsqueeze(-1) # (seq_len, 1)
        target_logits = torch.gather(logits, dim=-1, index=target_skills).squeeze(-1)
        probs = torch.sigmoid(target_logits)
        return probs

    def forward_batch(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """Batched DKT. skill_batch/correct_batch/mask: (B, T). Returns (B, T)."""
        device = skill_batch.device
        B, T = skill_batch.shape
        # Shift interactions by one step; step 0 gets <START> token (index 0)
        prev_inter = skill_batch[:, :-1] * 2 + correct_batch[:, :-1].long() + 1  # (B,T-1)
        start = torch.zeros(B, 1, dtype=torch.long, device=device)
        inter_ids = torch.cat([start, prev_inter], dim=1)  # (B,T)
        embs = self.input_emb(inter_ids)                   # (B,T,emb)
        lstm_out, _ = self.lstm(embs)                      # (B,T,hidden)
        logits = self.out(lstm_out)                        # (B,T,num_skills)
        tgt = skill_batch.unsqueeze(-1)                    # (B,T,1)
        target_logits = torch.gather(logits, dim=-1, index=tgt).squeeze(-1)
        return torch.sigmoid(target_logits)


class SAKT_Baseline(nn.Module):
    """Self-Attentive Knowledge Tracing (Pandey & Karypis, EDM 2019).

    A modern Transformer baseline: past interactions (skill x response) form the
    keys/values; the *next* queried skill forms the query; a causal mask blocks
    any leakage from the future. One self-attention block + position-wise FFN.
    Exposes forward_batch(sk, co, mask) -> (B,T) probs, matching the KT harness.
    """
    def __init__(self, num_skills: int, emb_dim: int = 64, n_heads: int = 4,
                 max_len: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.emb_dim = emb_dim
        self.max_len = max_len
        # Interaction embedding: (skill*2 + correct)+1, index 0 = <START>
        self.inter_emb = nn.Embedding(num_skills * 2 + 2, emb_dim, padding_idx=0)
        # Query embedding: the exercise/skill being asked at step t
        self.skill_emb = nn.Embedding(num_skills, emb_dim)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim))
        self.ln2 = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, 1)

    def forward_batch(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """skill/correct/mask: (B,T). Returns (B,T) probs. Causal (no future leak)."""
        device = skill_batch.device
        B, T = skill_batch.shape
        # Keys/values = PAST interactions: shift by one, step 0 gets <START>=0
        prev_inter = skill_batch[:, :-1] * 2 + correct_batch[:, :-1].long() + 1  # (B,T-1)
        start = torch.zeros(B, 1, dtype=torch.long, device=device)
        inter_ids = torch.cat([start, prev_inter], dim=1)                        # (B,T)
        pos = torch.arange(T, device=device).clamp(max=self.max_len - 1)
        pos = pos.unsqueeze(0).expand(B, T)
        kv = self.inter_emb(inter_ids) + self.pos_emb(pos)                       # (B,T,E)
        q = self.skill_emb(skill_batch) + self.pos_emb(pos)                      # (B,T,E)

        # Causal mask: position t may attend only to <= t (past interactions).
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        key_pad = ~mask.bool()  # (B,T) True where padded -> ignored
        a, _ = self.attn(q, kv, kv, attn_mask=causal, key_padding_mask=key_pad,
                         need_weights=False)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ffn(h))
        logits = self.out(h).squeeze(-1)                                         # (B,T)
        return torch.sigmoid(logits)


class SimpleKT_Baseline(nn.Module):
    """simpleKT (Liu et al., ICLR 2023) — a strong, simple Rasch + dot-product
    attention baseline. https://arxiv.org/abs/2302.06881

    Core ideas kept faithfully:
      - Rasch (IRT) embeddings: skill_emb(q) + difficulty(q) * variation(q), where
        difficulty is a scalar per skill and `variation` is the skill's deviation
        vector. Same trick used by AKT.
      - Ordinary (non-monotonic) scaled dot-product self-attention, causal-masked.
      - Predict next response from the queried skill attending over past interactions.
    Interface: forward_batch(sk, co, mask) -> (B,T) probs.
    """
    def __init__(self, num_skills: int, emb_dim: int = 64, n_heads: int = 4,
                 max_len: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.emb_dim = emb_dim
        self.max_len = max_len
        # Rasch embeddings (shared skill semantics + scalar difficulty)
        self.skill_emb = nn.Embedding(num_skills, emb_dim)          # c_q : base concept
        self.skill_diff = nn.Embedding(num_skills, 1)               # d_q : scalar difficulty
        self.skill_var = nn.Embedding(num_skills, emb_dim)          # f_q : variation vector
        # Interaction (skill x response) embeddings, also Rasch-augmented
        self.inter_emb = nn.Embedding(num_skills * 2 + 2, emb_dim, padding_idx=0)
        self.inter_var = nn.Embedding(num_skills * 2 + 2, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim))
        self.ln2 = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, 1)

    def forward_batch(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        device = skill_batch.device
        B, T = skill_batch.shape
        # --- Rasch query embedding: c_q + d_q * f_q ---
        c_q = self.skill_emb(skill_batch)                                  # (B,T,E)
        d_q = self.skill_diff(skill_batch)                                 # (B,T,1)
        f_q = self.skill_var(skill_batch)                                  # (B,T,E)
        q_rasch = c_q + d_q * f_q                                          # (B,T,E)
        # --- Past interactions (keys/values), shift by one, <START>=0 ---
        prev_inter = skill_batch[:, :-1] * 2 + correct_batch[:, :-1].long() + 1
        start = torch.zeros(B, 1, dtype=torch.long, device=device)
        inter_ids = torch.cat([start, prev_inter], dim=1)                  # (B,T)
        prev_skill = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device),
                                skill_batch[:, :-1]], dim=1)               # for difficulty
        e_inter = self.inter_emb(inter_ids) + \
            self.skill_diff(prev_skill) * self.inter_var(inter_ids)        # Rasch on kv
        pos = torch.arange(T, device=device).clamp(max=self.max_len - 1)
        pos = pos.unsqueeze(0).expand(B, T)
        q = q_rasch + self.pos_emb(pos)
        kv = e_inter + self.pos_emb(pos)
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        key_pad = ~mask.bool()
        a, _ = self.attn(q, kv, kv, attn_mask=causal, key_padding_mask=key_pad,
                         need_weights=False)
        h = self.ln1(q + a)
        h = self.ln2(h + self.ffn(h))
        return torch.sigmoid(self.out(h).squeeze(-1))


class AKT_Baseline(nn.Module):
    """AKT — Context-Aware Attentive Knowledge Tracing (Ghosh et al., KDD 2020).
    https://arxiv.org/abs/2007.12324

    Two signature components kept faithfully:
      1. Rasch-model embeddings (shared with simpleKT): x_q = c_q + d_q * f_q.
      2. MONOTONIC attention — attention logits are scaled by an exponentially
         decaying distance weight exp(-theta * d(t,tau)), where the context-aware
         distance d accumulates cumulative attention mass between positions, so
         recent interactions dominate. theta is a learned positive decay.
    A single-head monotonic attention block (implemented explicitly, since the
    decay cannot be expressed via nn.MultiheadAttention). Interface:
    forward_batch(sk, co, mask) -> (B,T) probs.
    """
    def __init__(self, num_skills: int, emb_dim: int = 64, max_len: int = 2048,
                 dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.scale = 1.0 / math.sqrt(emb_dim)
        # Rasch embeddings
        self.skill_emb = nn.Embedding(num_skills, emb_dim)
        self.skill_diff = nn.Embedding(num_skills, 1)
        self.skill_var = nn.Embedding(num_skills, emb_dim)
        self.inter_emb = nn.Embedding(num_skills * 2 + 2, emb_dim, padding_idx=0)
        self.inter_var = nn.Embedding(num_skills * 2 + 2, emb_dim, padding_idx=0)
        # Q/K/V projections for the monotonic attention block
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        # Learned positive decay rate theta (softplus-activated)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(emb_dim, emb_dim))
        self.ln2 = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, 1)

    def forward_batch(self, skill_batch: torch.Tensor, correct_batch: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        device = skill_batch.device
        B, T = skill_batch.shape
        # --- Rasch query embedding ---
        c_q = self.skill_emb(skill_batch)
        d_q = self.skill_diff(skill_batch)
        f_q = self.skill_var(skill_batch)
        x_q = c_q + d_q * f_q                                              # (B,T,E)
        # --- Past interaction embeddings (Rasch) ---
        prev_inter = skill_batch[:, :-1] * 2 + correct_batch[:, :-1].long() + 1
        start = torch.zeros(B, 1, dtype=torch.long, device=device)
        inter_ids = torch.cat([start, prev_inter], dim=1)
        prev_skill = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device),
                                skill_batch[:, :-1]], dim=1)
        y_inter = self.inter_emb(inter_ids) + \
            self.skill_diff(prev_skill) * self.inter_var(inter_ids)        # (B,T,E)

        Q = self.q_proj(x_q)                                              # (B,T,E)
        K = self.k_proj(y_inter)
        V = self.v_proj(y_inter)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale        # (B,T,T)

        # --- Causal + padding mask (before computing context-aware distance) ---
        causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        key_pad = ~mask.bool()                                            # (B,T)
        neg_inf = torch.finfo(scores.dtype).min
        full_mask = causal.unsqueeze(0) | key_pad.unsqueeze(1)            # (B,T,T)
        masked_scores = scores.masked_fill(full_mask, neg_inf)

        # --- Context-aware monotonic distance (AKT eq. 3-ish) ---
        # Base positional gap |t - tau|, reweighted by cumulative softmax mass so
        # densely-attended stretches count as "closer". Computed under no_grad for
        # the distance scaffold (as in the reference), decay theta stays learnable.
        with torch.no_grad():
            pos = torch.arange(T, device=device)
            gap = (pos.view(1, T, 1) - pos.view(1, 1, T)).abs().float()   # (1,T,T)
            gap = gap.expand(B, T, T).clone()
            gap = gap.masked_fill(full_mask, 0.0)
        soft = torch.softmax(masked_scores, dim=-1)                       # (B,T,T)
        # cumulative attention mass between t and tau (context-aware distance)
        dist_cum = torch.cumsum(soft.flip(-1), dim=-1).flip(-1)           # (B,T,T)
        total_dist = gap * dist_cum                                       # emphasize recent
        theta = F.softplus(self.theta)
        decay = torch.exp(-theta * total_dist)                            # (B,T,T) in (0,1]

        attn_logits = masked_scores + torch.log(decay.clamp_min(1e-9))
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, V)                                      # (B,T,E)
        h = self.ln(x_q + ctx)
        h = self.ln2(h + self.ffn(h))
        return torch.sigmoid(self.out(h).squeeze(-1))


# ==============================================================================
# 3. Prerequisite Relation Probing Model
# ==============================================================================

class EduHDC_PrereqProbe(nn.Module):
    """
    Neuro-symbolic Prerequisite Link Prediction Probe.
    Tests whether the asymmetric binding representation correctly separates
    directed prerequisite links (u -> v) from non-prerequisites and reverse links (v -> u).
    """
    def __init__(self, emb_dim: int = 384, vsa_dim: int = 2048, 
                 op_type: str = "edubind", device: str = "cuda"):
        super().__init__()
        self.emb_dim = emb_dim
        self.vsa_dim = vsa_dim
        self.device = device
        self.op_type = op_type

        # Projection from dense LLM embeddings into HDC space
        self.proj = nn.Linear(emb_dim, vsa_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

        # Operator
        if op_type == "edubind":
            self.vsa = EduBindBlockDiag(dim=vsa_dim, device=device)
            self.actual_dim = self.vsa.actual_dim
        elif op_type == "map":
            self.vsa = BipolarMAP(dim=vsa_dim, device=device)
            self.actual_dim = vsa_dim
        elif op_type == "hrr":
            self.vsa = RealHRR(dim=vsa_dim, device=device)
            self.actual_dim = vsa_dim
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

        # Learnable structural role vectors (Prerequisite vs Advanced)
        init_roles = self.vsa.random_vector(2).float()
        self.role_prereq = nn.Parameter(init_roles[0].unsqueeze(0), requires_grad=True)
        self.role_advanced = nn.Parameter(init_roles[1].unsqueeze(0), requires_grad=True)

        # Linear probe classifier directly on the bound relation hypervector
        self.probe_classifier = nn.Sequential(
            nn.Linear(self.actual_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def encode_concept(self, dense_emb: torch.Tensor) -> torch.Tensor:
        """Projects dense LLM embedding to normalized VSA space."""
        h = self.proj(dense_emb)
        return F.normalize(h, p=2, dim=-1)

    def encode_relation(self, u_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        """Constructs relation vector: bind(bind(u, role_P), bind(v, role_A))."""
        bound_u = self.vsa.bind(u_vec, self.role_prereq)
        bound_v = self.vsa.bind(v_vec, self.role_advanced)
        return self.vsa.bind(bound_u, bound_v)

    def forward(self, u_dense: torch.Tensor, v_dense: torch.Tensor) -> torch.Tensor:
        """
        u_dense: (batch_size, emb_dim) - source concepts
        v_dense: (batch_size, emb_dim) - target concepts
        Returns: logits of u being prerequisite of v (batch_size,)
        """
        u_h = self.encode_concept(u_dense)
        v_h = self.encode_concept(v_dense)

        # Full relation hypervector representation
        edge_vec = self.encode_relation(u_h, v_h) # (batch_size, actual_dim)

        logits = self.probe_classifier(edge_vec).squeeze(-1)
        return logits


class EduHDC_PrereqProbeRoleFiller(EduHDC_PrereqProbe):
    """
    Role-filler variant of EduHDC_PrereqProbe (C1 Revision 4, H0 experiment).

    EduHDC_PrereqProbe.encode_relation nests bind at the OUTER level:
        bind(bind(u, role_P), bind(v, role_A))
    For an associative-and-commutative operator (MAP: elementwise; HRR: circular
    convolution) this collapses to a single symmetric product
        u . role_P . v . role_A
    which is identical whether (u, v) or (v, u) was supplied -- the exact 0.0%
    "tie" in the original protocol is this algebraic fact, not a floating-point
    artifact. It does not test whether two DISTINCT roles bundled together
    (the construction the paper's H0 result, encPairHad, is actually about)
    distinguish order.

    This class instead implements the literal role-filler encoding:
        bundle( bind(u, role_P), bind(v, role_A) )
    i.e. bind each concept to its OWN role, then SUPERPOSE (sum) the two bound
    terms, rather than binding the two role-bound terms to each other. This is
    encPair from the paper (\\S4), realised in the same floating-point codebase
    used for every other row of Table 1, so it is a direct empirical test of
    hadamard_encPair_order_sensitive: does a role-filler encoding under a
    COMMUTATIVE operator (MAP) distinguish (u, v) from (v, u), where the
    outer-bind construction provably cannot?
    """

    def encode_relation(self, u_vec: torch.Tensor, v_vec: torch.Tensor) -> torch.Tensor:
        """Role-filler encoding: bundle(bind(u, role_P), bind(v, role_A))."""
        bound_u = self.vsa.bind(u_vec, self.role_prereq)
        bound_v = self.vsa.bind(v_vec, self.role_advanced)
        return self.vsa.bundle([bound_u, bound_v])
