# RLSD 思路(从 RLSD 论文搬)
# 算 evidence ratio = P_T(y_t) / P_S(y_t),作为 token-level credit assignment magnitude
# sign(A) 控制 direction,A 来自 GRPO group-relative advantage
def compute_evidence_ratio(p_T, p_S, eps=1e-8):
    return (p_T + eps) / (p_S + eps)
