//! Caching device allocator (Plan 3c6dc6e4).
//!
//! Inspired by PyTorch's CUDACachingAllocator but **deliberately
//! minimalist** — ~500 LOC vs ~5000. Skips expandable segments,
//! external streams, IPC handles and other production-grade
//! corner-cases that aren't required for v1.
//!
//! # Design
//! - Power-of-two **size bins** (4 KiB → 256 MiB) feeding **free
//!   lists** of free `Block`s.
//! - Each `Block` belongs to a **`Segment`** (a single
//!   `cuMemAlloc` call). Free blocks within a segment are
//!   coalesced via a doubly-linked list (`prev`, `next` indices).
//! - Multi-stream safety: a freshly freed block gets a
//!   `record_stream` annotation so it cannot be re-used until the
//!   pending stream's events have completed (we model this via
//!   "free-pending" lists, not real cudaEvents — a TODO under the
//!   `cuda` feature).
//! - OOM fallback: if a fresh `cuMemAlloc` fails, the allocator
//!   first **trims** all segments with no in-use block and retries
//!   once before propagating `CudaError::Oom`.
//! - Stats: `Allocator::stats()` reports allocated bytes, reserved
//!   bytes, peak, alloc count, free count, hits, misses, OOM count.

#![allow(clippy::needless_range_loop)]

use crate::error::CudaError;
use crate::stream::Stream;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// A handle to a block of device memory served by [`Allocator`].
///
/// The address (`addr`) is opaque — without `--features cuda` it's
/// a synthetic `u64` derived from a monotonic counter so tests can
/// uniquely identify blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevPtr {
    /// Opaque device address (real `CUdeviceptr` under cuda feature).
    pub addr: u64,
    /// Size of the served block, in bytes.
    pub size: usize,
}

/// Block state in a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
enum BlockState {
    Free,
    InUse,
    /// Free but pending — has a `record_stream` annotation; cannot
    /// be re-used until that stream's events drain.
    FreePending,
}

#[derive(Debug, Clone)]
struct Block {
    addr: u64,
    size: usize,
    segment_id: usize,
    prev: Option<usize>,
    next: Option<usize>,
    state: BlockState,
    /// Streams that recorded against this block while it was in use.
    record_streams: Vec<u64>,
}

#[derive(Debug)]
struct Segment {
    /// Base address of the segment — recorded for stats / future trim
    /// implementation that physically returns memory.
    #[allow(dead_code)]
    base_addr: u64,
    /// Total bytes in this segment.
    total: usize,
    /// Indices of all blocks in this segment (in address order via prev/next).
    blocks: Vec<usize>,
    /// Stream this segment is bound to (per-stream pools).
    stream: u64,
}

/// Allocator runtime statistics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AllocatorStats {
    /// Bytes currently in use by callers.
    pub allocated_bytes: u64,
    /// Bytes reserved (sum of segment sizes).
    pub reserved_bytes: u64,
    /// Peak `allocated_bytes` ever seen.
    pub peak_allocated_bytes: u64,
    /// Total successful `alloc` calls.
    pub alloc_count: u64,
    /// Total `free` calls.
    pub free_count: u64,
    /// Allocations satisfied from a free list (no `cuMemAlloc`).
    pub cache_hits: u64,
    /// Allocations that required a fresh `cuMemAlloc`.
    pub cache_misses: u64,
    /// OOM events (after one trim retry).
    pub oom_count: u64,
    /// Trim cycles run.
    pub trim_count: u64,
}

/// Round `n` up to the nearest power of two ≥ 4096.
fn round_up_pow2(n: usize) -> usize {
    let min = 4096usize;
    if n <= min {
        return min;
    }
    n.next_power_of_two()
}

/// Map a block size to its bin index. Bin 0 = 4 KiB, bin 1 = 8 KiB, ...
fn bin_index(size: usize) -> usize {
    let s = round_up_pow2(size);
    s.trailing_zeros() as usize - 12 // log2(4096)=12
}

/// Caching device-memory allocator.
#[derive(Debug, Default)]
pub struct Allocator {
    inner: Mutex<AllocatorInner>,
}

#[derive(Debug, Default)]
struct AllocatorInner {
    /// Bin index → queue of free block indices.
    bins: HashMap<u64, Vec<VecDeque<usize>>>, // key: stream raw, value: bins
    blocks: Vec<Block>,
    segments: Vec<Segment>,
    next_synth_addr: u64,
    stats: AllocatorStats,
    /// Free-pending blocks per stream (drained on `synchronize_stream`).
    pending: HashMap<u64, Vec<usize>>,
}

impl Allocator {
    /// Build an empty allocator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a snapshot of allocator stats.
    pub fn stats(&self) -> AllocatorStats {
        self.inner.lock().unwrap().stats.clone()
    }

    /// Allocate `size` bytes on the given `stream`. Round-up to the
    /// nearest power-of-two bin; serve from the free list when possible
    /// or fall through to a fresh segment allocation.
    pub fn alloc(&self, size: usize, stream: &Stream) -> Result<DevPtr, CudaError> {
        if size == 0 {
            return Ok(DevPtr { addr: 0, size: 0 });
        }
        let mut inner = self.inner.lock().unwrap();
        let stream_key = stream.raw();
        let rounded = round_up_pow2(size);
        let bin = bin_index(size);
        // Ensure the per-stream bins vector exists and is wide enough.
        let bins = inner
            .bins
            .entry(stream_key)
            .or_insert_with(|| (0..32).map(|_| VecDeque::new()).collect::<Vec<_>>());
        if bin >= bins.len() {
            bins.resize(bin + 1, VecDeque::new());
        }
        // 1. Try the free list for this bin.
        if let Some(idx) = bins[bin].pop_front() {
            let blk = &mut inner.blocks[idx];
            blk.state = BlockState::InUse;
            blk.record_streams.clear();
            let ptr = DevPtr {
                addr: blk.addr,
                size: blk.size,
            };
            inner.stats.allocated_bytes += rounded as u64;
            if inner.stats.allocated_bytes > inner.stats.peak_allocated_bytes {
                inner.stats.peak_allocated_bytes = inner.stats.allocated_bytes;
            }
            inner.stats.alloc_count += 1;
            inner.stats.cache_hits += 1;
            return Ok(ptr);
        }
        // 2. No cached block — try to split a larger free block in any
        //    segment bound to this stream.
        if let Some(idx) = inner.find_splittable(stream_key, rounded) {
            inner.split_and_use(idx, rounded, stream_key);
            inner.stats.allocated_bytes += rounded as u64;
            if inner.stats.allocated_bytes > inner.stats.peak_allocated_bytes {
                inner.stats.peak_allocated_bytes = inner.stats.allocated_bytes;
            }
            inner.stats.alloc_count += 1;
            inner.stats.cache_hits += 1;
            return Ok(DevPtr {
                addr: inner.blocks[idx].addr,
                size: inner.blocks[idx].size,
            });
        }
        // 3. Fresh segment.
        match inner.alloc_segment(rounded, stream_key) {
            Ok(idx) => {
                inner.stats.allocated_bytes += rounded as u64;
                if inner.stats.allocated_bytes > inner.stats.peak_allocated_bytes {
                    inner.stats.peak_allocated_bytes = inner.stats.allocated_bytes;
                }
                inner.stats.alloc_count += 1;
                inner.stats.cache_misses += 1;
                Ok(DevPtr {
                    addr: inner.blocks[idx].addr,
                    size: inner.blocks[idx].size,
                })
            },
            Err(_first_err) => {
                // 4. Trim free segments and retry once.
                inner.trim_free_segments();
                inner.stats.trim_count += 1;
                match inner.alloc_segment(rounded, stream_key) {
                    Ok(idx) => {
                        inner.stats.allocated_bytes += rounded as u64;
                        if inner.stats.allocated_bytes > inner.stats.peak_allocated_bytes {
                            inner.stats.peak_allocated_bytes = inner.stats.allocated_bytes;
                        }
                        inner.stats.alloc_count += 1;
                        inner.stats.cache_misses += 1;
                        Ok(DevPtr {
                            addr: inner.blocks[idx].addr,
                            size: inner.blocks[idx].size,
                        })
                    },
                    Err(e) => {
                        inner.stats.oom_count += 1;
                        Err(e)
                    },
                }
            },
        }
    }

    /// Return a block to the free list of its segment's stream. Coalesces
    /// adjacent free blocks.
    pub fn free(&self, ptr: DevPtr) -> Result<(), CudaError> {
        if ptr.size == 0 {
            return Ok(());
        }
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .blocks
            .iter()
            .position(|b| b.addr == ptr.addr && b.state == BlockState::InUse)
            .ok_or(CudaError::Unsupported {
                msg: format!("free: block @ {:#x} not in-use", ptr.addr),
            })?;
        // Determine final state: free or free-pending if any record_streams.
        let final_state = if inner.blocks[idx].record_streams.is_empty() {
            BlockState::Free
        } else {
            BlockState::FreePending
        };
        let (segment_id, blk_size) = {
            let blk = &mut inner.blocks[idx];
            blk.state = final_state.clone();
            (blk.segment_id, blk.size)
        };
        inner.stats.allocated_bytes = inner.stats.allocated_bytes.saturating_sub(blk_size as u64);
        inner.stats.free_count += 1;
        if final_state == BlockState::FreePending {
            // Key the pending entry by EACH recorded stream — the block
            // can only be released once every recorded stream has
            // synchronised. We use refcounting via duplicate entries:
            // each synchronize_stream call drops one entry, and the
            // block is released when its count reaches 0.
            let recorded: Vec<u64> = inner.blocks[idx].record_streams.clone();
            let _ = segment_id;
            for sk in &recorded {
                inner.pending.entry(*sk).or_default().push(idx);
            }
            return Ok(());
        }
        inner.coalesce_and_return(idx);
        Ok(())
    }

    /// Annotate `ptr` so it cannot be re-used until `stream`'s events
    /// have drained. Mimics `cudaStreamRecord` semantics — if `ptr` is
    /// later freed, it goes onto the per-stream pending list.
    pub fn record_stream(&self, ptr: DevPtr, stream: &Stream) -> Result<(), CudaError> {
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .blocks
            .iter()
            .position(|b| b.addr == ptr.addr && b.state == BlockState::InUse)
            .ok_or(CudaError::Unsupported {
                msg: "record_stream on non-live block".into(),
            })?;
        inner.blocks[idx].record_streams.push(stream.raw());
        Ok(())
    }

    /// Drain pending blocks for `stream` (real cuda path: wait for the
    /// stream to reach the recorded event). Returns the count of
    /// blocks fully released — i.e. blocks for which every recorded
    /// stream has now synchronised.
    pub fn synchronize_stream(&self, stream: &Stream) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let key = stream.raw();
        let drained: Vec<usize> = inner.pending.remove(&key).unwrap_or_default();
        let mut released = 0usize;
        for idx in drained {
            // Pop this stream off the recorded list. If the list is now
            // empty, the block can be coalesced into the free list.
            if let Some(pos) = inner.blocks[idx]
                .record_streams
                .iter()
                .position(|&s| s == key)
            {
                inner.blocks[idx].record_streams.remove(pos);
            }
            if inner.blocks[idx].record_streams.is_empty()
                && inner.blocks[idx].state == BlockState::FreePending
            {
                inner.blocks[idx].state = BlockState::Free;
                inner.coalesce_and_return(idx);
                released += 1;
            }
        }
        released
    }

    /// Manually trim free segments. Returns bytes freed.
    pub fn empty_cache(&self) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        let before = inner.stats.reserved_bytes;
        inner.trim_free_segments();
        inner.stats.trim_count += 1;
        before - inner.stats.reserved_bytes
    }
}

impl AllocatorInner {
    fn find_splittable(&self, stream_key: u64, want: usize) -> Option<usize> {
        // Scan free blocks within a segment bound to this stream and pick the smallest fit.
        let mut best: Option<(usize, usize)> = None;
        for (i, b) in self.blocks.iter().enumerate() {
            if b.state == BlockState::Free
                && b.size >= want
                && self.segments[b.segment_id].stream == stream_key
            {
                let extra = b.size - want;
                if best.is_none() || extra < best.unwrap().1 {
                    best = Some((i, extra));
                }
            }
        }
        best.map(|(i, _)| i)
    }

    fn split_and_use(&mut self, idx: usize, want: usize, _stream_key: u64) {
        // Split off the right side as a new free block if there's enough
        // remainder; otherwise just take the whole block.
        let (cur_size, cur_addr, segment_id, next) = {
            let b = &self.blocks[idx];
            (b.size, b.addr, b.segment_id, b.next)
        };
        if cur_size > want {
            let new_idx = self.blocks.len();
            self.blocks.push(Block {
                addr: cur_addr + want as u64,
                size: cur_size - want,
                segment_id,
                prev: Some(idx),
                next,
                state: BlockState::Free,
                record_streams: Vec::new(),
            });
            self.blocks[idx].size = want;
            self.blocks[idx].next = Some(new_idx);
            if let Some(n) = next {
                self.blocks[n].prev = Some(new_idx);
            }
            self.segments[segment_id].blocks.push(new_idx);
            // Insert the new block into the free list of its bin.
            let stream_key = self.segments[segment_id].stream;
            let bin = bin_index(self.blocks[new_idx].size);
            let bins = self.bins.get_mut(&stream_key).unwrap();
            if bin >= bins.len() {
                bins.resize(bin + 1, VecDeque::new());
            }
            bins[bin].push_back(new_idx);
        }
        self.blocks[idx].state = BlockState::InUse;
        self.blocks[idx].record_streams.clear();
    }

    fn alloc_segment(&mut self, size: usize, stream_key: u64) -> Result<usize, CudaError> {
        // Synthetic alloc — no real GPU. Real cuda path:
        //   cuMemAllocAsync(&dptr, size, stream)
        let base_addr = if self.next_synth_addr == 0 {
            self.next_synth_addr = 0x1_0000_0000;
            self.next_synth_addr
        } else {
            self.next_synth_addr += size as u64 + 0x1000;
            self.next_synth_addr
        };
        let blk_idx = self.blocks.len();
        let seg_idx = self.segments.len();
        self.blocks.push(Block {
            addr: base_addr,
            size,
            segment_id: seg_idx,
            prev: None,
            next: None,
            state: BlockState::InUse,
            record_streams: Vec::new(),
        });
        self.segments.push(Segment {
            base_addr,
            total: size,
            blocks: vec![blk_idx],
            stream: stream_key,
        });
        self.stats.reserved_bytes += size as u64;
        Ok(blk_idx)
    }

    fn coalesce_and_return(&mut self, idx: usize) {
        // Merge with prev if adjacent and free.
        let mut cur = idx;
        loop {
            let prev = self.blocks[cur].prev;
            match prev {
                Some(p) if self.blocks[p].state == BlockState::Free => {
                    // p will absorb cur.
                    let cur_size = self.blocks[cur].size;
                    let cur_next = self.blocks[cur].next;
                    self.blocks[p].size += cur_size;
                    self.blocks[p].next = cur_next;
                    if let Some(n) = cur_next {
                        self.blocks[n].prev = Some(p);
                    }
                    // Remove `cur` from its bin's free list (if any) and
                    // mark it as a tombstone (Free + size 0). We re-bin p.
                    self.unbin(cur);
                    self.blocks[cur].size = 0;
                    cur = p;
                },
                _ => break,
            }
        }
        // Merge with next if adjacent and free.
        loop {
            let next = self.blocks[cur].next;
            match next {
                Some(n) if self.blocks[n].state == BlockState::Free => {
                    let n_size = self.blocks[n].size;
                    let n_next = self.blocks[n].next;
                    self.blocks[cur].size += n_size;
                    self.blocks[cur].next = n_next;
                    if let Some(nn) = n_next {
                        self.blocks[nn].prev = Some(cur);
                    }
                    self.unbin(n);
                    self.blocks[n].size = 0;
                },
                _ => break,
            }
        }
        // Re-insert into bin.
        let stream_key = self.segments[self.blocks[cur].segment_id].stream;
        let bin = bin_index(self.blocks[cur].size);
        let bins = self
            .bins
            .entry(stream_key)
            .or_insert_with(|| (0..32).map(|_| VecDeque::new()).collect::<Vec<_>>());
        if bin >= bins.len() {
            bins.resize(bin + 1, VecDeque::new());
        }
        bins[bin].push_back(cur);
    }

    fn unbin(&mut self, idx: usize) {
        let stream_key = self.segments[self.blocks[idx].segment_id].stream;
        if let Some(bins) = self.bins.get_mut(&stream_key) {
            for q in bins.iter_mut() {
                if let Some(pos) = q.iter().position(|&x| x == idx) {
                    q.remove(pos);
                }
            }
        }
    }

    fn trim_free_segments(&mut self) {
        // Keep only segments whose blocks are all InUse or FreePending.
        let n = self.segments.len();
        let mut keep = vec![true; n];
        for (si, seg) in self.segments.iter().enumerate() {
            let all_free = seg.blocks.iter().all(|&b| {
                matches!(self.blocks[b].state, BlockState::Free) && self.blocks[b].size > 0
            });
            if all_free {
                keep[si] = false;
                self.stats.reserved_bytes -= seg.total as u64;
            }
        }
        // Remove blocks belonging to trimmed segments from bins (best-effort:
        // we leave the block records but mark them sized 0 to prevent re-use).
        for si in 0..n {
            if !keep[si] {
                for &b in &self.segments[si].blocks {
                    self.blocks[b].state = BlockState::Free;
                    self.blocks[b].size = 0;
                }
            }
        }
        // We don't physically remove segments — it would invalidate the
        // segment_id indices stored in blocks. For v1, leave as tombstones.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn s(idx: u32) -> Stream {
        Stream::new(Device { index: idx }).unwrap()
    }

    #[test]
    fn round_up_pow2_basics() {
        assert_eq!(round_up_pow2(1), 4096);
        assert_eq!(round_up_pow2(4096), 4096);
        assert_eq!(round_up_pow2(4097), 8192);
        assert_eq!(round_up_pow2(1_000_000), 1_048_576);
    }

    #[test]
    fn bin_index_4kib_is_zero() {
        assert_eq!(bin_index(1), 0);
        assert_eq!(bin_index(4096), 0);
        assert_eq!(bin_index(8192), 1);
    }

    #[test]
    fn alloc_zero_size_returns_null_ptr() {
        let a = Allocator::new();
        let p = a.alloc(0, &s(0)).unwrap();
        assert_eq!(p.addr, 0);
        assert_eq!(p.size, 0);
    }

    #[test]
    fn alloc_then_free_reuses_block() {
        let a = Allocator::new();
        let st = s(0);
        let p1 = a.alloc(1024, &st).unwrap();
        a.free(p1).unwrap();
        let p2 = a.alloc(1024, &st).unwrap();
        assert_eq!(p1.addr, p2.addr, "second alloc should hit the cache");
        let s = a.stats();
        assert_eq!(s.alloc_count, 2);
        assert_eq!(s.free_count, 1);
        assert_eq!(s.cache_hits, 1);
        assert_eq!(s.cache_misses, 1);
    }

    #[test]
    fn alloc_distinct_streams_use_distinct_segments() {
        let a = Allocator::new();
        let p0 = a.alloc(8192, &s(0)).unwrap();
        let p1 = a.alloc(8192, &s(1)).unwrap();
        assert_ne!(p0.addr, p1.addr);
        assert_eq!(a.stats().cache_misses, 2);
    }

    #[test]
    fn record_stream_defers_reuse_until_synchronize() {
        let a = Allocator::new();
        let st = s(0);
        let other = s(1);
        let p = a.alloc(4096, &st).unwrap();
        a.record_stream(p, &other).unwrap();
        a.free(p).unwrap();
        // Re-alloc same size on same stream should NOT return same addr
        // because the block is pending.
        let p2 = a.alloc(4096, &st).unwrap();
        assert_ne!(p.addr, p2.addr);
        // After synchronize, the pending block becomes free.
        let drained = a.synchronize_stream(&other);
        assert_eq!(drained, 1);
    }

    #[test]
    fn peak_tracks_max_in_flight_bytes() {
        let a = Allocator::new();
        let st = s(0);
        let p1 = a.alloc(4096, &st).unwrap();
        let p2 = a.alloc(4096, &st).unwrap();
        let peak = a.stats().peak_allocated_bytes;
        a.free(p1).unwrap();
        a.free(p2).unwrap();
        // After free, allocated_bytes drops; peak stays.
        let after = a.stats();
        assert!(after.allocated_bytes < peak);
        assert_eq!(after.peak_allocated_bytes, peak);
        assert!(peak >= 8192);
    }

    #[test]
    fn empty_cache_trims_fully_free_segments() {
        let a = Allocator::new();
        let st = s(0);
        let p = a.alloc(4096, &st).unwrap();
        a.free(p).unwrap();
        let trimmed = a.empty_cache();
        assert!(trimmed >= 4096);
    }

    #[test]
    fn double_free_returns_error() {
        let a = Allocator::new();
        let st = s(0);
        let p = a.alloc(4096, &st).unwrap();
        a.free(p).unwrap();
        let err = a.free(p).unwrap_err();
        assert!(matches!(err, CudaError::Unsupported { .. }));
    }

    #[test]
    fn coalesce_merges_adjacent_free_blocks_after_split() {
        let a = Allocator::new();
        let st = s(0);
        // Pull a large segment then ask for something smaller — splits off a free remainder.
        let p_big = a.alloc(1 << 20, &st).unwrap();
        a.free(p_big).unwrap();
        // Now ask for half — should split off a free remainder.
        let p_half = a.alloc(1 << 19, &st).unwrap();
        a.free(p_half).unwrap();
        // After freeing both halves, coalesce should reunite them so a
        // 1MB request hits the cache (no new segment).
        let before_misses = a.stats().cache_misses;
        let _p_full = a.alloc(1 << 20, &st).unwrap();
        let after_misses = a.stats().cache_misses;
        assert_eq!(after_misses, before_misses, "request should hit cache");
    }
}
