//! Forward hooks for `Module`s (P1.6).
//!
//! Mirrors `torch.nn.Module.register_forward_pre_hook` and
//! `register_forward_hook`. A pre-hook can observe the input before
//! `forward` runs; a post-hook can observe the (input, output) pair
//! after. Hooks are typically used for activation logging, feature
//! extraction, or debugging.
//!
//! v1 hooks are **observers only** — they cannot mutate the input or
//! output to redirect the forward pass. Mutating hooks are pending a
//! follow-up.
//!
//! Example:
//! ```ignore
//! let mut model = HookedModule::new(Linear::new(8, 4));
//! let h = model.add_forward_hook(|input, output| {
//!     println!("forward: in shape {:?} → out shape {:?}",
//!              input.tensor().shape(), output.tensor().shape());
//! });
//! let y = model.forward(&x)?;
//! model.remove_hook(h);
//! ```

use crate::module::{Module, ModuleError};
use rustorch_autograd::Variable;

/// Removable handle returned from `add_*_hook`. Pass back to
/// `remove_hook` to unregister.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HookHandle {
    /// Internal id (per-Module unique).
    id: usize,
    /// Whether this handle refers to a pre or post hook.
    kind: HookKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum HookKind {
    Pre,
    Post,
}

type PreHook = Box<dyn Fn(&Variable) + Send + Sync>;
type PostHook = Box<dyn Fn(&Variable, &Variable) + Send + Sync>;

/// Wrap any [`Module`] with observer hooks. Forward = run pre-hooks,
/// inner forward, post-hooks.
pub struct HookedModule<M: Module> {
    inner: M,
    pre_hooks: Vec<(usize, PreHook)>,
    post_hooks: Vec<(usize, PostHook)>,
    next_id: usize,
}

impl<M: Module> HookedModule<M> {
    /// Wrap the given module. No hooks registered initially.
    pub fn new(inner: M) -> Self {
        HookedModule {
            inner,
            pre_hooks: Vec::new(),
            post_hooks: Vec::new(),
            next_id: 0,
        }
    }

    /// Borrow the inner module.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Register a forward-pre hook. Returns a [`HookHandle`] for removal.
    pub fn add_forward_pre_hook<F>(&mut self, hook: F) -> HookHandle
    where
        F: Fn(&Variable) + Send + Sync + 'static,
    {
        let id = self.next_id;
        self.next_id += 1;
        self.pre_hooks.push((id, Box::new(hook)));
        HookHandle {
            id,
            kind: HookKind::Pre,
        }
    }

    /// Register a forward-post hook. Returns a [`HookHandle`] for removal.
    pub fn add_forward_hook<F>(&mut self, hook: F) -> HookHandle
    where
        F: Fn(&Variable, &Variable) + Send + Sync + 'static,
    {
        let id = self.next_id;
        self.next_id += 1;
        self.post_hooks.push((id, Box::new(hook)));
        HookHandle {
            id,
            kind: HookKind::Post,
        }
    }

    /// Remove a previously-registered hook by handle. Returns true if
    /// found. Idempotent — calling twice with the same handle returns
    /// false the second time.
    pub fn remove_hook(&mut self, handle: HookHandle) -> bool {
        match handle.kind {
            HookKind::Pre => {
                if let Some(idx) = self.pre_hooks.iter().position(|(id, _)| *id == handle.id) {
                    let _ = self.pre_hooks.swap_remove(idx);
                    return true;
                }
            },
            HookKind::Post => {
                if let Some(idx) = self.post_hooks.iter().position(|(id, _)| *id == handle.id) {
                    let _ = self.post_hooks.swap_remove(idx);
                    return true;
                }
            },
        }
        false
    }

    /// Number of currently-registered pre hooks.
    pub fn n_pre_hooks(&self) -> usize {
        self.pre_hooks.len()
    }

    /// Number of currently-registered post hooks.
    pub fn n_post_hooks(&self) -> usize {
        self.post_hooks.len()
    }
}

impl<M: Module> Module for HookedModule<M> {
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        for (_, h) in &self.pre_hooks {
            h(input);
        }
        let output = self.inner.forward(input)?;
        for (_, h) in &self.post_hooks {
            h(input, &output);
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<Variable> {
        self.inner.parameters()
    }

    fn named_parameters(&self) -> Vec<(String, Variable)> {
        self.inner.named_parameters()
    }

    fn train(&mut self) {
        self.inner.train();
    }

    fn eval(&mut self) {
        self.inner.eval();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Relu};
    use rustorch_core::tensor::tensor_impl::Tensor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn input(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Variable {
        Variable::new(Tensor::from_vec(shape.into(), data).unwrap())
    }

    #[test]
    fn pre_hook_observes_input() {
        let mut model = HookedModule::new(Linear::new(2, 2));
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        let _h = model.add_forward_pre_hook(move |x| {
            assert_eq!(x.tensor().shape(), &[1, 2]);
            c2.fetch_add(1, Ordering::SeqCst);
        });
        let x = input(vec![1usize, 2], vec![1.0_f32, 2.0]);
        let _ = model.forward(&x).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn post_hook_observes_input_output() {
        let mut model = HookedModule::new(Linear::new(3, 4));
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        let _h = model.add_forward_hook(move |x, y| {
            assert_eq!(x.tensor().shape(), &[1, 3]);
            assert_eq!(y.tensor().shape(), &[1, 4]);
            c2.fetch_add(1, Ordering::SeqCst);
        });
        let x = input(vec![1usize, 3], vec![0.1_f32; 3]);
        let _ = model.forward(&x).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn multiple_hooks_called_in_order() {
        let mut model = HookedModule::new(Relu);
        let order = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let o1 = order.clone();
        let _h1 = model.add_forward_pre_hook(move |_| o1.lock().unwrap().push(1));
        let o2 = order.clone();
        let _h2 = model.add_forward_pre_hook(move |_| o2.lock().unwrap().push(2));
        let x = input(vec![3usize], vec![-1.0_f32, 0.0, 1.0]);
        let _ = model.forward(&x).unwrap();
        assert_eq!(*order.lock().unwrap(), vec![1, 2]);
    }

    #[test]
    fn remove_hook_unregisters() {
        let mut model = HookedModule::new(Relu);
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        let h = model.add_forward_pre_hook(move |_| {
            c2.fetch_add(1, Ordering::SeqCst);
        });
        let x = input(vec![1usize], vec![1.0_f32]);
        let _ = model.forward(&x).unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        assert!(model.remove_hook(h));
        let _ = model.forward(&x).unwrap();
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "hook should not fire after removal"
        );

        // Idempotent: remove twice returns false.
        assert!(!model.remove_hook(h));
    }

    #[test]
    fn no_hooks_means_passthrough() {
        let model = HookedModule::new(Linear::new(2, 2));
        let x = input(vec![1usize, 2], vec![0.5_f32, -0.5]);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[1, 2]);
        assert_eq!(model.n_pre_hooks(), 0);
        assert_eq!(model.n_post_hooks(), 0);
    }

    #[test]
    fn hooks_dont_break_parameters() {
        let mut model = HookedModule::new(Linear::new(3, 2));
        let _h1 = model.add_forward_pre_hook(|_| {});
        let _h2 = model.add_forward_hook(|_, _| {});
        // Linear has weight + bias = 2 parameters
        assert_eq!(model.parameters().len(), 2);
        assert_eq!(model.named_parameters().len(), 2);
    }
}
