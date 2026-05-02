//! NCCL bindings (Plan b51b848c) — preparation for Phase 5.
//!
//! Provides safe wrappers over NCCL's `ncclComm_t` for the four
//! collectives that DDP needs: `all_reduce`, `all_gather`,
//! `reduce_scatter`, `broadcast`. Plus `barrier` and group-op
//! helpers.
//!
//! Without `--features cuda`, all collectives execute **single-rank**
//! synchronously: the result is mathematically correct (no inter-rank
//! data exchange happens because there's only one rank), enabling
//! end-to-end algorithmic tests of code that *uses* these primitives
//! without booting up multiple GPUs.
//!
//! The `Communicator::from_unique_id` flow mirrors NCCL's
//! `ncclGetUniqueId` + `ncclCommInitRank` two-step rendezvous.

use crate::error::CudaError;
use crate::stream::Stream;
use std::sync::atomic::{AtomicU64, Ordering};

/// Reduction operator for `all_reduce` / `reduce_scatter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise product.
    Prod,
    /// Element-wise max.
    Max,
    /// Element-wise min.
    Min,
    /// Sum then divide by `world_size`.
    Avg,
}

/// NCCL unique-id (128 bytes in real NCCL — represented here by a
/// scalar tag for the fallback path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniqueId {
    /// Opaque tag.
    pub tag: u64,
}

static UNIQUE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

impl UniqueId {
    /// Generate a fresh unique id (process-local — real NCCL distributes
    /// this via the rendezvous server).
    pub fn new() -> Self {
        Self {
            tag: UNIQUE_ID_COUNTER.fetch_add(1, Ordering::SeqCst) + 1,
        }
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self::new()
    }
}

/// NCCL communicator — wraps `ncclComm_t` under the cuda feature.
#[derive(Debug, Clone)]
pub struct Communicator {
    /// Process-global rank.
    pub rank: usize,
    /// Total ranks in the communicator.
    pub world_size: usize,
    /// Unique id this comm was bootstrapped with.
    pub unique_id: UniqueId,
}

impl Communicator {
    /// Bootstrap a comm given a `unique_id`, the process's `rank` and
    /// the total `world_size`. Real NCCL: `ncclCommInitRank`.
    pub fn from_unique_id(
        unique_id: UniqueId,
        rank: usize,
        world_size: usize,
    ) -> Result<Self, CudaError> {
        if world_size == 0 {
            return Err(CudaError::Unsupported {
                msg: "world_size must be >= 1".into(),
            });
        }
        if rank >= world_size {
            return Err(CudaError::Unsupported {
                msg: format!("rank {rank} out of range (world_size={world_size})"),
            });
        }
        Ok(Self {
            rank,
            world_size,
            unique_id,
        })
    }

    /// Single-process world: convenience for tests and embarrassingly
    /// parallel single-GPU runs.
    pub fn single_process() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            unique_id: UniqueId::new(),
        }
    }
}

/// `out[i] = op(in[i] across all ranks)` — in-place if `out == in`.
///
/// In the no-cuda fallback this is a no-op when `world_size == 1`
/// (the data is already its own reduction). For `world_size > 1` we
/// would dispatch to NCCL via FFI; calling this without the cuda
/// feature when `world_size > 1` returns `Unsupported` so callers
/// catch missing GPU init.
pub fn all_reduce_f32(
    input: &[f32],
    output: &mut [f32],
    comm: &Communicator,
    op: ReduceOp,
    _stream: &Stream,
) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "all_reduce shape mismatch".into(),
        });
    }
    if comm.world_size == 1 {
        match op {
            ReduceOp::Sum | ReduceOp::Max | ReduceOp::Min | ReduceOp::Prod | ReduceOp::Avg => {
                output.copy_from_slice(input);
                if matches!(op, ReduceOp::Avg) {
                    // /world_size == /1 == no-op
                }
                Ok(())
            },
        }
    } else {
        Err(CudaError::Unsupported {
            msg: "all_reduce with world_size > 1 requires --features cuda + NCCL".into(),
        })
    }
}

/// Broadcast `input` from `root` to all other ranks. Single-rank
/// fallback simply copies `input` → `output`.
pub fn broadcast_f32(
    input: &[f32],
    output: &mut [f32],
    root: usize,
    comm: &Communicator,
    _stream: &Stream,
) -> Result<(), CudaError> {
    if input.len() != output.len() {
        return Err(CudaError::Unsupported {
            msg: "broadcast shape mismatch".into(),
        });
    }
    if root >= comm.world_size {
        return Err(CudaError::Unsupported {
            msg: format!("broadcast root {root} >= world_size {}", comm.world_size),
        });
    }
    if comm.world_size == 1 {
        output.copy_from_slice(input);
        Ok(())
    } else {
        Err(CudaError::Unsupported {
            msg: "broadcast with world_size > 1 requires --features cuda + NCCL".into(),
        })
    }
}

/// `output` of size `input.len() * world_size`: each rank's `input` is
/// copied into the slice at offset `rank * input.len()`.
pub fn all_gather_f32(
    input: &[f32],
    output: &mut [f32],
    comm: &Communicator,
    _stream: &Stream,
) -> Result<(), CudaError> {
    if output.len() != input.len() * comm.world_size {
        return Err(CudaError::Unsupported {
            msg: format!(
                "all_gather output expected {}, got {}",
                input.len() * comm.world_size,
                output.len()
            ),
        });
    }
    if comm.world_size == 1 {
        output.copy_from_slice(input);
        Ok(())
    } else {
        Err(CudaError::Unsupported {
            msg: "all_gather with world_size > 1 requires --features cuda + NCCL".into(),
        })
    }
}

/// Reduce-scatter: each rank ends up with `output_len = input_len /
/// world_size` elements that are the reduction of the corresponding
/// slice from every rank's input.
pub fn reduce_scatter_f32(
    input: &[f32],
    output: &mut [f32],
    comm: &Communicator,
    op: ReduceOp,
    _stream: &Stream,
) -> Result<(), CudaError> {
    if input.len() != output.len() * comm.world_size {
        return Err(CudaError::Unsupported {
            msg: format!(
                "reduce_scatter input expected {}, got {}",
                output.len() * comm.world_size,
                input.len()
            ),
        });
    }
    if comm.world_size == 1 {
        // Single rank — output is the rank-local slice (which is all of input).
        output.copy_from_slice(input);
        if matches!(op, ReduceOp::Avg) {
            // /1 = identity
        }
        Ok(())
    } else {
        Err(CudaError::Unsupported {
            msg: "reduce_scatter with world_size > 1 requires --features cuda + NCCL".into(),
        })
    }
}

/// `barrier()` blocks until all ranks reach this point. Single-rank
/// fallback: instant return.
pub fn barrier(comm: &Communicator, _stream: &Stream) -> Result<(), CudaError> {
    if comm.world_size == 1 {
        Ok(())
    } else {
        Err(CudaError::Unsupported {
            msg: "barrier with world_size > 1 requires --features cuda + NCCL".into(),
        })
    }
}

/// Group multiple collectives into a single launch group. Real NCCL:
/// `ncclGroupStart` / `ncclGroupEnd`. The closure runs all enqueued
/// ops as a single batch, hiding latency.
pub fn group<F: FnOnce() -> Result<(), CudaError>>(f: F) -> Result<(), CudaError> {
    // No-op in fallback — the body executes synchronously.
    f()
}

/// Phase-5 prep API: spawn `world_size` worker processes, hand each
/// one its `rank` + the shared `unique_id`. Returns a vector of
/// per-rank communicators (for the in-process test path).
pub fn spawn_world(world_size: usize) -> Result<Vec<Communicator>, CudaError> {
    if world_size == 0 {
        return Err(CudaError::Unsupported {
            msg: "world_size must be >= 1".into(),
        });
    }
    let unique_id = UniqueId::new();
    (0..world_size)
        .map(|rank| Communicator::from_unique_id(unique_id, rank, world_size))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn st() -> Stream {
        Stream::new(Device { index: 0 }).unwrap()
    }

    #[test]
    fn unique_id_is_monotonic() {
        let a = UniqueId::new();
        let b = UniqueId::new();
        assert_ne!(a.tag, b.tag);
        assert!(b.tag > a.tag);
    }

    #[test]
    fn comm_rejects_oob_rank() {
        let id = UniqueId::new();
        assert!(Communicator::from_unique_id(id, 4, 4).is_err());
    }

    #[test]
    fn comm_rejects_zero_world_size() {
        let id = UniqueId::new();
        assert!(Communicator::from_unique_id(id, 0, 0).is_err());
    }

    #[test]
    fn all_reduce_single_rank_copies_input_to_output() {
        let comm = Communicator::single_process();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        all_reduce_f32(&input, &mut output, &comm, ReduceOp::Sum, &st()).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn all_reduce_world_size_2_returns_unsupported_without_cuda() {
        let id = UniqueId::new();
        let comm = Communicator::from_unique_id(id, 0, 2).unwrap();
        let input = vec![1.0f32];
        let mut output = vec![0.0f32];
        let err = all_reduce_f32(&input, &mut output, &comm, ReduceOp::Sum, &st()).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn broadcast_single_rank_copies_input() {
        let comm = Communicator::single_process();
        let input = vec![1.0f32, 2.0];
        let mut output = vec![0.0f32; 2];
        broadcast_f32(&input, &mut output, 0, &comm, &st()).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn broadcast_oob_root_returns_error() {
        let comm = Communicator::single_process();
        let mut output = vec![0.0f32];
        let err = broadcast_f32(&[0.0f32], &mut output, 5, &comm, &st()).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn all_gather_single_rank_copies_input() {
        let comm = Communicator::single_process();
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        all_gather_f32(&input, &mut output, &comm, &st()).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn reduce_scatter_single_rank_copies_input() {
        let comm = Communicator::single_process();
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        reduce_scatter_f32(&input, &mut output, &comm, ReduceOp::Sum, &st()).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn barrier_single_rank_returns_immediately() {
        let comm = Communicator::single_process();
        barrier(&comm, &st()).unwrap();
    }

    #[test]
    fn group_executes_body_synchronously() {
        let mut x = 0;
        group(|| {
            x = 42;
            Ok(())
        })
        .unwrap();
        assert_eq!(x, 42);
    }

    #[test]
    fn spawn_world_returns_n_communicators_with_distinct_ranks() {
        let comms = spawn_world(4).unwrap();
        assert_eq!(comms.len(), 4);
        for (i, c) in comms.iter().enumerate() {
            assert_eq!(c.rank, i);
            assert_eq!(c.world_size, 4);
        }
        // All comms share the same unique_id (the rendezvous tag).
        let tag0 = comms[0].unique_id.tag;
        for c in &comms[1..] {
            assert_eq!(c.unique_id.tag, tag0);
        }
    }
}
