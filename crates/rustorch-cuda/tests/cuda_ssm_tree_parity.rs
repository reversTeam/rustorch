//! T246.7 TrackC.1 — Parity tests for the tree-aware Gated DeltaNet kernel.
//!
//! `delta_net_step_tree_bf16(wave_size=1, parents=[-1])` with the model
//! state pre-loaded into slot 0 MUST produce a result bit-identical to one
//! call of `delta_net_step_bf16` on that same state.
//!
//! Plus :
//!   - chain   : parents=[-1, 0, 1] applied as 3 successive depth waves
//!     produces the same final state as 3 successive scalar calls.
//!   - branch  : parents=[-1, 0, 0] produces two siblings forked from
//!     node 0's state ; their resulting states are :
//!       * different from each other (different (q,k,v) inputs)
//!       * each equal to the scalar update applied to the parent state
//!         using only that branch's inputs.
//!
//! Running on GB10 (DGX Spark) :
//!   PATH=/usr/local/cuda-13.0/bin:$PATH \
//!     cargo test --release --features cuda -p rustorch-cuda \
//!       --test cuda_ssm_tree_parity -- --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use rustorch_cuda::llm_kernels::LlmKernels;

const N_HEADS: usize = 4;
const HEAD_DIM: usize = 16;

fn bf16_vec(n: usize, seed: f32) -> Vec<half::bf16> {
    (0..n)
        .map(|i| half::bf16::from_f32(((i as f32 + 1.0) * seed).sin() * 0.3))
        .collect()
}

fn approx_eq_bf16(a: &[half::bf16], b: &[half::bf16], tol: f32) -> Option<(usize, f32, f32)> {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let xf = x.to_f32();
        let yf = y.to_f32();
        if (xf - yf).abs() > tol {
            return Some((i, xf, yf));
        }
    }
    None
}

#[test]
fn delta_net_step_tree_bf16_wave_size_1_matches_scalar() {
    let n_heads = N_HEADS;
    let head_dim = HEAD_DIM;
    let io_per_node = n_heads * head_dim;
    let state_per_node = n_heads * head_dim * head_dim;

    let q = bf16_vec(io_per_node, 0.013);
    let k = bf16_vec(io_per_node, 0.017);
    let v = bf16_vec(io_per_node, 0.019);
    let gate = bf16_vec(n_heads, 0.07);
    let beta = bf16_vec(n_heads, 0.11);
    let state_init = bf16_vec(state_per_node, 0.005);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // ── Reference : call scalar kernel once on state_init.
    let mut state_a = stream.memcpy_stod(&state_init).expect("state_a");
    let mut out_a = stream
        .alloc_zeros::<half::bf16>(io_per_node)
        .expect("out_a");
    let q_dev = stream.memcpy_stod(&q).expect("q");
    let k_dev = stream.memcpy_stod(&k).expect("k");
    let v_dev = stream.memcpy_stod(&v).expect("v");
    let g_dev = stream.memcpy_stod(&gate).expect("gate");
    let b_dev = stream.memcpy_stod(&beta).expect("beta");

    unsafe {
        let (qp, _g0) = q_dev.device_ptr(&stream);
        let (kp, _g1) = k_dev.device_ptr(&stream);
        let (vp, _g2) = v_dev.device_ptr(&stream);
        let (gp, _g3) = g_dev.device_ptr(&stream);
        let (bp, _g4) = b_dev.device_ptr(&stream);
        let (sp, _g5) = state_a.device_ptr_mut(&stream);
        let (op, _g6) = out_a.device_ptr_mut(&stream);
        kernels
            .delta_net_step_bf16(
                &stream,
                qp,
                kp,
                vp,
                gp,
                bp,
                sp,
                op,
                n_heads as i32,
                head_dim as i32,
            )
            .unwrap();
    }
    let state_a_host: Vec<half::bf16> = stream.memcpy_dtov(&state_a).expect("state_a dtoh");
    let out_a_host: Vec<half::bf16> = stream.memcpy_dtov(&out_a).expect("out_a dtoh");

    // ── Tree call : tree_size=1, wave_size=1, parents=[-1], wave=[0].
    //    Pre-load slot 0 with state_init (root convention).
    let tree_size = 1usize;
    let mut tree_states_b = stream.memcpy_stod(&state_init).expect("tree_states_b");
    let mut out_b = stream
        .alloc_zeros::<half::bf16>(io_per_node)
        .expect("out_b");
    let parents_dev = stream.memcpy_stod(&[-1i32]).expect("parents");
    let wave_dev = stream.memcpy_stod(&[0i32]).expect("wave");

    unsafe {
        let (qp, _g0) = q_dev.device_ptr(&stream);
        let (kp, _g1) = k_dev.device_ptr(&stream);
        let (vp, _g2) = v_dev.device_ptr(&stream);
        let (gp, _g3) = g_dev.device_ptr(&stream);
        let (bp, _g4) = b_dev.device_ptr(&stream);
        let (par_p, _g5) = parents_dev.device_ptr(&stream);
        let (wav_p, _g6) = wave_dev.device_ptr(&stream);
        let (sp, _g7) = tree_states_b.device_ptr_mut(&stream);
        let (op, _g8) = out_b.device_ptr_mut(&stream);
        kernels
            .delta_net_step_tree_bf16(
                &stream,
                qp,
                kp,
                vp,
                gp,
                bp,
                par_p,
                wav_p,
                sp,
                op,
                tree_size as i32,
                n_heads as i32,
                head_dim as i32,
            )
            .unwrap();
    }
    let state_b_host: Vec<half::bf16> = stream.memcpy_dtov(&tree_states_b).expect("state_b dtoh");
    let out_b_host: Vec<half::bf16> = stream.memcpy_dtov(&out_b).expect("out_b dtoh");

    // Bit-exact match expected.
    if let Some((i, a, b)) = approx_eq_bf16(&state_a_host, &state_b_host, 0.0) {
        panic!("state mismatch at {i}: scalar={a} tree={b}");
    }
    if let Some((i, a, b)) = approx_eq_bf16(&out_a_host, &out_b_host, 0.0) {
        panic!("out mismatch at {i}: scalar={a} tree={b}");
    }
}

#[test]
fn delta_net_step_tree_bf16_chain_3_matches_3_scalar() {
    // parents = [-1, 0, 1] — linear chain root → child1 → child2.
    // Each tree row gets its own (q, k, v, gate, beta) input. We launch
    // 3 depth waves : [0], [1], [2]. The resulting tree_states[2] MUST
    // equal the result of 3 successive scalar calls applied to the
    // initial state with the same per-step inputs.
    let n_heads = N_HEADS;
    let head_dim = HEAD_DIM;
    let io_per_node = n_heads * head_dim;
    let state_per_node = n_heads * head_dim * head_dim;
    let g_per_node = n_heads;
    let tree_size = 3usize;

    let mut q_all = Vec::with_capacity(tree_size * io_per_node);
    let mut k_all = Vec::with_capacity(tree_size * io_per_node);
    let mut v_all = Vec::with_capacity(tree_size * io_per_node);
    let mut g_all = Vec::with_capacity(tree_size * g_per_node);
    let mut b_all = Vec::with_capacity(tree_size * g_per_node);
    for tr in 0..tree_size {
        q_all.extend(bf16_vec(io_per_node, 0.013 + tr as f32 * 0.001));
        k_all.extend(bf16_vec(io_per_node, 0.017 + tr as f32 * 0.001));
        v_all.extend(bf16_vec(io_per_node, 0.019 + tr as f32 * 0.001));
        g_all.extend(bf16_vec(g_per_node, 0.07 + tr as f32 * 0.001));
        b_all.extend(bf16_vec(g_per_node, 0.11 + tr as f32 * 0.001));
    }
    let state_init = bf16_vec(state_per_node, 0.005);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // ── Reference : 3 scalar calls in sequence on a single rolling state.
    let mut state_ref = stream.memcpy_stod(&state_init).expect("state_ref");
    let mut out_ref = stream
        .alloc_zeros::<half::bf16>(io_per_node)
        .expect("out_ref");
    for tr in 0..tree_size {
        let q_step = stream
            .memcpy_stod(&q_all[tr * io_per_node..(tr + 1) * io_per_node])
            .expect("q_step");
        let k_step = stream
            .memcpy_stod(&k_all[tr * io_per_node..(tr + 1) * io_per_node])
            .expect("k_step");
        let v_step = stream
            .memcpy_stod(&v_all[tr * io_per_node..(tr + 1) * io_per_node])
            .expect("v_step");
        let g_step = stream
            .memcpy_stod(&g_all[tr * g_per_node..(tr + 1) * g_per_node])
            .expect("g_step");
        let b_step = stream
            .memcpy_stod(&b_all[tr * g_per_node..(tr + 1) * g_per_node])
            .expect("b_step");
        unsafe {
            let (qp, _g0) = q_step.device_ptr(&stream);
            let (kp, _g1) = k_step.device_ptr(&stream);
            let (vp, _g2) = v_step.device_ptr(&stream);
            let (gp, _g3) = g_step.device_ptr(&stream);
            let (bp, _g4) = b_step.device_ptr(&stream);
            let (sp, _g5) = state_ref.device_ptr_mut(&stream);
            let (op, _g6) = out_ref.device_ptr_mut(&stream);
            kernels
                .delta_net_step_bf16(
                    &stream,
                    qp,
                    kp,
                    vp,
                    gp,
                    bp,
                    sp,
                    op,
                    n_heads as i32,
                    head_dim as i32,
                )
                .unwrap();
        }
    }
    let state_ref_host: Vec<half::bf16> = stream.memcpy_dtov(&state_ref).expect("ref dtoh");

    // ── Tree call : pre-load state_init into slot 0 ; launch 3 waves.
    let mut buf = vec![half::bf16::ZERO; tree_size * state_per_node];
    buf[..state_per_node].copy_from_slice(&state_init);
    let mut tree_states = stream.memcpy_stod(&buf).expect("init tree_states");
    let mut tree_out = stream
        .alloc_zeros::<half::bf16>(tree_size * io_per_node)
        .expect("tree_out");

    let q_dev = stream.memcpy_stod(&q_all).expect("q_all");
    let k_dev = stream.memcpy_stod(&k_all).expect("k_all");
    let v_dev = stream.memcpy_stod(&v_all).expect("v_all");
    let g_dev = stream.memcpy_stod(&g_all).expect("g_all");
    let b_dev = stream.memcpy_stod(&b_all).expect("b_all");
    let parents_dev = stream.memcpy_stod(&[-1i32, 0, 1]).expect("parents");

    // 3 waves : [0], [1], [2].
    for tr in 0..tree_size {
        let wave_dev = stream.memcpy_stod(&[tr as i32]).expect("wave");
        unsafe {
            let (qp, _g0) = q_dev.device_ptr(&stream);
            let (kp, _g1) = k_dev.device_ptr(&stream);
            let (vp, _g2) = v_dev.device_ptr(&stream);
            let (gp, _g3) = g_dev.device_ptr(&stream);
            let (bp, _g4) = b_dev.device_ptr(&stream);
            let (par_p, _g5) = parents_dev.device_ptr(&stream);
            let (wav_p, _g6) = wave_dev.device_ptr(&stream);
            let (sp, _g7) = tree_states.device_ptr_mut(&stream);
            let (op, _g8) = tree_out.device_ptr_mut(&stream);
            kernels
                .delta_net_step_tree_bf16(
                    &stream,
                    qp,
                    kp,
                    vp,
                    gp,
                    bp,
                    par_p,
                    wav_p,
                    sp,
                    op,
                    1,
                    n_heads as i32,
                    head_dim as i32,
                )
                .unwrap();
        }
    }
    let tree_states_host: Vec<half::bf16> =
        stream.memcpy_dtov(&tree_states).expect("tree_states dtoh");

    // The deepest accepted slot (index 2) must equal the reference.
    let leaf_state = &tree_states_host[2 * state_per_node..3 * state_per_node];
    if let Some((i, a, b)) = approx_eq_bf16(&state_ref_host, leaf_state, 0.0) {
        panic!("chain leaf state mismatch at {i}: scalar={a} tree={b}");
    }
}

#[test]
fn delta_net_step_tree_bf16_branch_2_forks_from_parent() {
    // parents = [-1, 0, 0] — root has two children. Each child gets its
    // own (q, k, v, gate, beta) inputs. After 2 waves ([0] then [1, 2]) :
    //   - tree_states[1] must equal the scalar update applied to state_init
    //     using (q, k, v, gate, beta)[0] then (q, k, v, gate, beta)[1].
    //   - tree_states[2] must equal the scalar update applied to state_init
    //     using (q, k, v, gate, beta)[0] then (q, k, v, gate, beta)[2].
    //   - Therefore tree_states[1] != tree_states[2] as long as the inputs
    //     differ.
    let n_heads = N_HEADS;
    let head_dim = HEAD_DIM;
    let io_per_node = n_heads * head_dim;
    let state_per_node = n_heads * head_dim * head_dim;
    let g_per_node = n_heads;
    let tree_size = 3usize;

    let mut q_all = Vec::with_capacity(tree_size * io_per_node);
    let mut k_all = Vec::with_capacity(tree_size * io_per_node);
    let mut v_all = Vec::with_capacity(tree_size * io_per_node);
    let mut g_all = Vec::with_capacity(tree_size * g_per_node);
    let mut b_all = Vec::with_capacity(tree_size * g_per_node);
    for tr in 0..tree_size {
        q_all.extend(bf16_vec(io_per_node, 0.013 + tr as f32 * 0.005));
        k_all.extend(bf16_vec(io_per_node, 0.017 + tr as f32 * 0.005));
        v_all.extend(bf16_vec(io_per_node, 0.019 + tr as f32 * 0.005));
        g_all.extend(bf16_vec(g_per_node, 0.07 + tr as f32 * 0.005));
        b_all.extend(bf16_vec(g_per_node, 0.11 + tr as f32 * 0.005));
    }
    let state_init = bf16_vec(state_per_node, 0.005);

    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let kernels = LlmKernels::new(ctx);

    // ── Reference for child 1 : scalar(state_init, inputs[0]) → s_root ;
    //    then scalar(s_root, inputs[1]) → s_child1.
    let compute_chain = |steps: &[usize]| -> Vec<half::bf16> {
        let mut s = stream.memcpy_stod(&state_init).expect("s");
        let mut o = stream.alloc_zeros::<half::bf16>(io_per_node).expect("o");
        for &tr in steps {
            let q_step = stream
                .memcpy_stod(&q_all[tr * io_per_node..(tr + 1) * io_per_node])
                .unwrap();
            let k_step = stream
                .memcpy_stod(&k_all[tr * io_per_node..(tr + 1) * io_per_node])
                .unwrap();
            let v_step = stream
                .memcpy_stod(&v_all[tr * io_per_node..(tr + 1) * io_per_node])
                .unwrap();
            let g_step = stream
                .memcpy_stod(&g_all[tr * g_per_node..(tr + 1) * g_per_node])
                .unwrap();
            let b_step = stream
                .memcpy_stod(&b_all[tr * g_per_node..(tr + 1) * g_per_node])
                .unwrap();
            unsafe {
                let (qp, _g0) = q_step.device_ptr(&stream);
                let (kp, _g1) = k_step.device_ptr(&stream);
                let (vp, _g2) = v_step.device_ptr(&stream);
                let (gp, _g3) = g_step.device_ptr(&stream);
                let (bp, _g4) = b_step.device_ptr(&stream);
                let (sp, _g5) = s.device_ptr_mut(&stream);
                let (op, _g6) = o.device_ptr_mut(&stream);
                kernels
                    .delta_net_step_bf16(
                        &stream,
                        qp,
                        kp,
                        vp,
                        gp,
                        bp,
                        sp,
                        op,
                        n_heads as i32,
                        head_dim as i32,
                    )
                    .unwrap();
            }
        }
        stream.memcpy_dtov(&s).expect("s dtoh")
    };

    let ref_child1 = compute_chain(&[0, 1]);
    let ref_child2 = compute_chain(&[0, 2]);

    // ── Tree call : 2 waves [0] then [1, 2].
    let mut buf = vec![half::bf16::ZERO; tree_size * state_per_node];
    buf[..state_per_node].copy_from_slice(&state_init);
    let mut tree_states = stream.memcpy_stod(&buf).expect("init tree_states");
    let mut tree_out = stream
        .alloc_zeros::<half::bf16>(tree_size * io_per_node)
        .expect("tree_out");

    let q_dev = stream.memcpy_stod(&q_all).expect("q_all");
    let k_dev = stream.memcpy_stod(&k_all).expect("k_all");
    let v_dev = stream.memcpy_stod(&v_all).expect("v_all");
    let g_dev = stream.memcpy_stod(&g_all).expect("g_all");
    let b_dev = stream.memcpy_stod(&b_all).expect("b_all");
    let parents_dev = stream.memcpy_stod(&[-1i32, 0, 0]).expect("parents");

    for wave in &[vec![0i32], vec![1i32, 2i32]] {
        let wave_dev = stream.memcpy_stod(wave).expect("wave");
        unsafe {
            let (qp, _g0) = q_dev.device_ptr(&stream);
            let (kp, _g1) = k_dev.device_ptr(&stream);
            let (vp, _g2) = v_dev.device_ptr(&stream);
            let (gp, _g3) = g_dev.device_ptr(&stream);
            let (bp, _g4) = b_dev.device_ptr(&stream);
            let (par_p, _g5) = parents_dev.device_ptr(&stream);
            let (wav_p, _g6) = wave_dev.device_ptr(&stream);
            let (sp, _g7) = tree_states.device_ptr_mut(&stream);
            let (op, _g8) = tree_out.device_ptr_mut(&stream);
            kernels
                .delta_net_step_tree_bf16(
                    &stream,
                    qp,
                    kp,
                    vp,
                    gp,
                    bp,
                    par_p,
                    wav_p,
                    sp,
                    op,
                    wave.len() as i32,
                    n_heads as i32,
                    head_dim as i32,
                )
                .unwrap();
        }
    }
    let tree_states_host: Vec<half::bf16> =
        stream.memcpy_dtov(&tree_states).expect("tree_states dtoh");

    let child1 = &tree_states_host[state_per_node..2 * state_per_node];
    let child2 = &tree_states_host[2 * state_per_node..3 * state_per_node];

    if let Some((i, a, b)) = approx_eq_bf16(&ref_child1, child1, 0.0) {
        panic!("branch child1 mismatch at {i}: scalar={a} tree={b}");
    }
    if let Some((i, a, b)) = approx_eq_bf16(&ref_child2, child2, 0.0) {
        panic!("branch child2 mismatch at {i}: scalar={a} tree={b}");
    }

    // Sanity : siblings must differ (different inputs → different forks).
    let mut any_diff = false;
    for (a, b) in child1.iter().zip(child2.iter()) {
        if (a.to_f32() - b.to_f32()).abs() > 1e-4 {
            any_diff = true;
            break;
        }
    }
    assert!(
        any_diff,
        "branch siblings produced identical states — fork did not diverge"
    );
}
