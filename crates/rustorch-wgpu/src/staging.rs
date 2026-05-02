//! Staging buffer ring for non-stalling CPU → GPU uploads.
//!
//! `queue.write_buffer` is convenient but copies through wgpu's
//! internal staging path on every call, which can stall when many
//! small uploads are submitted back-to-back. This module implements
//! a **fence-tracked rotating ring** of `MAP_WRITE` staging buffers:
//!
//! - On `upload(payload)`, pick the oldest slot whose previous submit
//!   has completed (or wait until one frees up).
//! - Map the slot, copy the payload, unmap, and `copy_buffer_to_buffer`
//!   into the destination via the supplied encoder.
//! - The next call sees the previous submission and only reuses the
//!   slot after polling the device.
//!
//! Payloads larger than the slot size **spill** to a one-shot staging
//! buffer that is dropped immediately — this keeps the ring's working
//! set bounded and avoids unbounded memory growth on big uploads.

use std::cell::RefCell;

const DEFAULT_SLOT_BYTES: u64 = 4 * 1024 * 1024; // 4 MiB
const DEFAULT_NUM_SLOTS: usize = 4;

/// One slot in the ring: a `MAP_WRITE | COPY_SRC` buffer plus the
/// submission index that last touched it (so we know when the GPU
/// is done reading it back).
struct Slot {
    buffer: wgpu::Buffer,
    last_submission: Option<wgpu::SubmissionIndex>,
}

/// Rotating ring of staging buffers.
///
/// Cheap to construct (`StagingRing::with_defaults`) — buffers are
/// allocated lazily on the first `upload`. Use one per backend.
pub struct StagingRing {
    slots: RefCell<Vec<Slot>>,
    /// Per-slot byte size. Uploads larger than this spill (see below).
    slot_bytes: u64,
    /// Number of slots in the ring. Higher = more in-flight uploads
    /// before stalls; lower = less GPU memory.
    num_slots: usize,
    /// Round-robin cursor for the next slot to use.
    next: RefCell<usize>,
    /// Counter of payloads that exceeded `slot_bytes` and bypassed
    /// the ring via a one-shot staging allocation.
    spilled: RefCell<u64>,
}

impl Default for StagingRing {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl StagingRing {
    /// Build a ring with [`DEFAULT_NUM_SLOTS`] × [`DEFAULT_SLOT_BYTES`].
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SLOT_BYTES, DEFAULT_NUM_SLOTS)
    }

    /// Build a ring with custom sizing. `num_slots` is clamped to ≥ 1.
    pub fn new(slot_bytes: u64, num_slots: usize) -> Self {
        let n = num_slots.max(1);
        StagingRing {
            slots: RefCell::new(Vec::with_capacity(n)),
            slot_bytes,
            num_slots: n,
            next: RefCell::new(0),
            spilled: RefCell::new(0),
        }
    }

    /// Bytes per slot.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Number of slots in the ring.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Number of slots currently allocated. Lazy: starts at 0 and
    /// grows up to `num_slots` as `upload` is called.
    pub fn allocated_slots(&self) -> usize {
        self.slots.borrow().len()
    }

    /// Total number of uploads that bypassed the ring because they
    /// exceeded `slot_bytes`. Useful for telemetry: if this number
    /// is high, the slot size is undersized.
    pub fn spilled_count(&self) -> u64 {
        *self.spilled.borrow()
    }

    fn ensure_slot(&self, device: &wgpu::Device) -> usize {
        let mut slots = self.slots.borrow_mut();
        let mut next = self.next.borrow_mut();
        if slots.len() < self.num_slots {
            slots.push(Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rustorch-wgpu staging ring slot"),
                    size: self.slot_bytes,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                last_submission: None,
            });
            let idx = slots.len() - 1;
            *next = (idx + 1) % self.num_slots;
            return idx;
        }
        let idx = *next;
        *next = (idx + 1) % self.num_slots;
        idx
    }

    /// Upload `data` into `dst[dst_offset..dst_offset+data.len()]`
    /// using a slot from the ring. Records the work into `encoder`;
    /// the caller is responsible for submitting that encoder later
    /// and calling [`StagingRing::record_submission`] with the
    /// returned `SubmissionIndex`.
    ///
    /// Payloads larger than [`StagingRing::slot_bytes`] use a
    /// one-shot staging buffer and bypass the ring (counted as a
    /// "spill" — see [`StagingRing::spilled_count`]).
    pub fn upload(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dst: &wgpu::Buffer,
        dst_offset: u64,
        data: &[u8],
    ) {
        let n = data.len() as u64;
        if n == 0 {
            return;
        }
        if n > self.slot_bytes {
            // Spill: build a single-shot staging buffer sized exactly.
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rustorch-wgpu staging spill"),
                size: n,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            });
            staging
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(data);
            staging.unmap();
            encoder.copy_buffer_to_buffer(&staging, 0, dst, dst_offset, n);
            *self.spilled.borrow_mut() += 1;
            return;
        }

        let idx = self.ensure_slot(device);
        let needs_wait = self.slots.borrow()[idx].last_submission.is_some();
        if needs_wait {
            // Conservative: block until previous submits drain. wgpu
            // 22 exposes a more granular WaitForSubmissionIndex —
            // upgrade target for a follow-up slice once the API
            // stabilises across our toolchain.
            device.poll(wgpu::Maintain::Wait);
        }

        let slots = self.slots.borrow();
        let slot = &slots[idx];
        let buf = &slot.buffer;
        let slice = buf.slice(..n);
        slice.map_async(wgpu::MapMode::Write, |_| {});
        device.poll(wgpu::Maintain::Wait);
        slice.get_mapped_range_mut().copy_from_slice(data);
        buf.unmap();
        encoder.copy_buffer_to_buffer(buf, 0, dst, dst_offset, n);
    }

    /// After the encoder has been submitted, call this with the
    /// returned `SubmissionIndex` so the ring knows when the most
    /// recently used slot becomes safe to reuse.
    pub fn record_submission(&self, idx: wgpu::SubmissionIndex) {
        let mut slots = self.slots.borrow_mut();
        // Tag every slot whose last_submission is None — these were
        // touched by the upload(s) preceding this submit.
        for slot in slots.iter_mut() {
            if slot.last_submission.is_none() {
                slot.last_submission = Some(idx.clone());
            }
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::backend::WgpuBackend;

    #[test]
    fn ring_lazy_allocates_slots() {
        let ring = StagingRing::with_defaults();
        assert_eq!(ring.allocated_slots(), 0);
        assert_eq!(ring.num_slots(), DEFAULT_NUM_SLOTS);
    }

    #[test]
    fn ring_uploads_round_trip() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let ring = StagingRing::new(64, 2);
        let dst = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-test-dst"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let payload: Vec<u8> = (0..16_u8).collect();
        let mut enc = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("staging-test"),
            });
        ring.upload(&backend.device, &mut enc, &dst, 0, &payload);
        let sub = backend.queue.submit(Some(enc.finish()));
        ring.record_submission(sub);

        let readback = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-test-readback"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&dst, 0, &readback, 0, 16);
        backend.queue.submit(Some(enc.finish()));
        let slice = readback.slice(..16);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        backend.device.poll(wgpu::Maintain::Wait);
        let got: Vec<u8> = slice.get_mapped_range().to_vec();
        readback.unmap();
        assert_eq!(got, payload);
    }

    #[test]
    fn ring_spills_when_payload_exceeds_slot() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let ring = StagingRing::new(64, 2);
        let dst = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let payload = vec![0xAA_u8; 200];
        let mut enc = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ring.upload(&backend.device, &mut enc, &dst, 0, &payload);
        backend.queue.submit(Some(enc.finish()));
        assert_eq!(ring.spilled_count(), 1);
    }

    #[test]
    fn ring_empty_payload_is_noop() {
        let backend = WgpuBackend::new_blocking().expect("init");
        let ring = StagingRing::new(64, 2);
        let dst = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ring.upload(&backend.device, &mut enc, &dst, 0, &[]);
        backend.queue.submit(Some(enc.finish()));
        assert_eq!(ring.spilled_count(), 0);
        assert_eq!(ring.allocated_slots(), 0);
    }

    #[test]
    fn ring_rotates_slots_across_uploads() {
        let backend = WgpuBackend::new_blocking().expect("init");
        // 2 slots, payload smaller than slot → second upload should
        // allocate the second slot, third upload should reuse the
        // first one.
        let ring = StagingRing::new(64, 2);
        let dst = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        for i in 0..3_u8 {
            let payload = vec![i; 8];
            let mut enc = backend
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            ring.upload(&backend.device, &mut enc, &dst, 0, &payload);
            let sub = backend.queue.submit(Some(enc.finish()));
            ring.record_submission(sub);
        }
        // After 3 uploads on a 2-slot ring, both slots are allocated.
        assert_eq!(ring.allocated_slots(), 2);
        assert_eq!(ring.spilled_count(), 0);
    }
}
