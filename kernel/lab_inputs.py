import torch


def ref_torch(mixed_qkv, a, b, A_log, dt_bias, scale, state, indices, H, K):
    B = mixed_qkv.shape[0]
    HV, V = state.shape[1], state.shape[2]
    x = mixed_qkv.float()
    q = x[:, : H * K].view(B, H, K)
    k = x[:, H * K: 2 * H * K].view(B, H, K)
    v = x[:, 2 * H * K:].view(B, HV, V)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * scale
    rep = HV // H
    q = q.repeat_interleave(rep, 1)
    k = k.repeat_interleave(rep, 1)
    xg = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * torch.where(xg <= 20.0, torch.log1p(torch.exp(xg)), xg)
    beta = torch.sigmoid(b.float())
    h = state[indices.long()].float().clone()                 # [B, HV, V, K]
    h = h * torch.exp(g)[..., None, None]
    v_new = (v - torch.einsum("bhvk,bhk->bhv", h, k)) * beta[..., None]
    h = h + torch.einsum("bhv,bhk->bhvk", v_new, k)
    o = torch.einsum("bhvk,bhk->bhv", h, q)
    return o, h


def make_inputs(B, H, HV, K, V, state_dtype, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    def rn(*s):
        return torch.randn(*s, device=device, generator=gen)
    mixed_qkv = rn(B, 2 * H * K + HV * V).to(torch.bfloat16)
    a = rn(B, HV).to(torch.bfloat16)
    b = rn(B, HV).to(torch.bfloat16)
    A_log = (rn(HV) * 0.5 - 2.0).float()
    dt_bias = (rn(HV) * 0.5).float()
    state = (rn(B + 1, HV, V, K) * 0.1).to(state_dtype)       # 槽 0 留空
    indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    out = torch.empty(B, 1, HV, V, device=device, dtype=torch.bfloat16)
    return mixed_qkv, a, b, A_log, dt_bias, state, indices, out


def bytes_moved(B, H, HV, K, V, state_elem_bytes):
    state = B * HV * V * K * state_elem_bytes * 2             # 状态读一遍 + 写一遍
    qkv = B * (2 * H * K + HV * V) * 2
    ab = B * HV * 2 * 2
    o = B * HV * V * 2
    return state + qkv + ab + o


def make_input_sets(B, H, HV, K, V, sdt, min_total_bytes=160e6, max_sets=64):
    """多份互不相同的输入。轮流使用，保证连续调用之间不会命中 L2（模拟真实模型里几十层轮流跑）。"""
    per = bytes_moved(B, H, HV, K, V, sdt.itemsize)
    R = int(min(max_sets, max(8, -(-min_total_bytes // per))))
    return [make_inputs(B, H, HV, K, V, sdt, seed=i) for i in range(R)]


def bench_graph(call, sets, iters=20):
    """把 len(sets) 次调用录进一个 CUDA Graph，回放计时，返回每次调用的 us。
    这样没有 Python / launch 开销，和 vLLM decode 跑在 CUDA Graph 里的情况一致。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for x in sets:
                call(*x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for x in sets:
            call(*x)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        times.append(st.elapsed_time(en) * 1000 / len(sets))
    times.sort()
    return times[len(times) // 2]
