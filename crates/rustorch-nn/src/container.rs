//! Container modules — store + iterate over child modules.
//!
//! - [`ModuleList`] — `Vec<Box<dyn Module>>`. No `forward`; users
//!   iterate manually. Useful for stacking N transformer blocks where
//!   you need access to each layer's output.
//! - [`ModuleDict`] — named children via `HashMap<String, Box<dyn
//!   Module>>`. Useful for branched architectures (encoder/decoder,
//!   multiple heads).
//!
//! Both expose `parameters()` aggregating across children, and
//! `train()`/`eval()` propagating mode flips.

use crate::module::{Module, ModuleError};
use rustorch_autograd::Variable;
use std::collections::HashMap;

// ------------------------------ ModuleList ------------------------------

/// Ordered list of modules. No `forward` method — users iterate.
#[derive(Default)]
pub struct ModuleList {
    modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    /// Empty list.
    pub fn new() -> Self {
        ModuleList {
            modules: Vec::new(),
        }
    }

    /// Append a module (builder-style).
    #[must_use]
    pub fn push<M: Module + 'static>(mut self, m: M) -> Self {
        self.modules.push(Box::new(m));
        self
    }

    /// Number of children.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Borrow an immutable child.
    pub fn get(&self, idx: usize) -> Option<&dyn Module> {
        self.modules.get(idx).map(|b| b.as_ref())
    }

    /// Iterate over the children (immutably).
    pub fn iter(&self) -> impl Iterator<Item = &dyn Module> {
        self.modules.iter().map(|b| b.as_ref())
    }

    /// Apply a closure that takes the input and successively produces
    /// the next output, given each child module. Equivalent to
    /// `modules.iter().fold(input, |x, m| m.forward(&x))` but exposes
    /// the per-layer output if the closure wants to inspect it.
    pub fn fold(&self, input: &Variable) -> Result<Variable, ModuleError> {
        let mut x = input.clone();
        for m in &self.modules {
            x = m.forward(&x)?;
        }
        Ok(x)
    }

    /// Aggregate parameters across all children (delegates to
    /// `Module::parameters` impl below).
    pub fn collect_parameters(&self) -> Vec<Variable> {
        self.modules.iter().flat_map(|m| m.parameters()).collect()
    }
}

impl Module for ModuleList {
    /// `ModuleList` doesn't define a forward chain — call [`Self::fold`]
    /// to chain through children, or iterate manually with [`Self::iter`].
    /// `forward` returns the input unchanged.
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Variable> {
        self.collect_parameters()
    }

    fn train(&mut self) {
        for m in &mut self.modules {
            m.train();
        }
    }

    fn eval(&mut self) {
        for m in &mut self.modules {
            m.eval();
        }
    }
}

// ------------------------------ ModuleDict ------------------------------

/// Named map of modules. Useful for branched architectures.
#[derive(Default)]
pub struct ModuleDict {
    modules: HashMap<String, Box<dyn Module>>,
    // Stable insertion order for parameters() ordering.
    order: Vec<String>,
}

impl ModuleDict {
    /// Empty dict.
    pub fn new() -> Self {
        ModuleDict {
            modules: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Insert a named child (builder-style). Re-inserting overwrites.
    #[must_use]
    pub fn insert<S: Into<String>, M: Module + 'static>(mut self, key: S, m: M) -> Self {
        let k = key.into();
        if !self.modules.contains_key(&k) {
            self.order.push(k.clone());
        }
        self.modules.insert(k, Box::new(m));
        self
    }

    /// Number of children.
    pub fn len(&self) -> usize {
        self.modules.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    /// Look up a child by name.
    pub fn get(&self, key: &str) -> Option<&dyn Module> {
        self.modules.get(key).map(|b| b.as_ref())
    }

    /// Names in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }
}

impl Module for ModuleDict {
    /// `ModuleDict` doesn't define a forward chain — call children by
    /// name. `forward` returns the input unchanged.
    fn forward(&self, input: &Variable) -> Result<Variable, ModuleError> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<Variable> {
        // Stable order via `self.order`.
        self.order
            .iter()
            .filter_map(|k| self.modules.get(k))
            .flat_map(|m| m.parameters())
            .collect()
    }

    fn train(&mut self) {
        for m in self.modules.values_mut() {
            m.train();
        }
    }

    fn eval(&mut self) {
        for m in self.modules.values_mut() {
            m.eval();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, Relu};

    #[test]
    fn module_list_collects_params_across_children() {
        let list = ModuleList::new()
            .push(Linear::new(4, 4))
            .push(Relu)
            .push(Linear::new(4, 2));
        // 2 Linears * (weight + bias) = 4 params
        assert_eq!(list.parameters().len(), 4);
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn module_list_fold_chains_forward() {
        use rustorch_autograd::Variable;
        use rustorch_core::tensor::tensor_impl::Tensor;
        let list = ModuleList::new()
            .push(Linear::new(3, 3))
            .push(Relu)
            .push(Linear::new(3, 2));
        let x = Variable::new(Tensor::from_vec([1usize, 3], vec![1.0_f32, 2.0, 3.0]).unwrap());
        let y = list.fold(&x).unwrap();
        assert_eq!(y.tensor().shape(), &[1, 2]);
    }

    #[test]
    fn module_list_get_by_index() {
        let list = ModuleList::new().push(Linear::new(2, 2)).push(Relu);
        assert!(list.get(0).is_some());
        assert!(list.get(1).is_some());
        assert!(list.get(2).is_none());
    }

    #[test]
    fn module_dict_keys_in_insertion_order() {
        let dict = ModuleDict::new()
            .insert("encoder", Linear::new(8, 4))
            .insert("decoder", Linear::new(4, 8))
            .insert("head", Linear::new(8, 2));
        let keys: Vec<&str> = dict.keys().collect();
        assert_eq!(keys, vec!["encoder", "decoder", "head"]);
    }

    #[test]
    fn module_dict_get_returns_named_child() {
        let dict = ModuleDict::new()
            .insert("layer1", Linear::new(4, 4))
            .insert("act", Relu);
        assert!(dict.get("layer1").is_some());
        assert!(dict.get("missing").is_none());
    }

    #[test]
    fn module_dict_parameters_count() {
        let dict = ModuleDict::new()
            .insert("a", Linear::new(2, 3))
            .insert("b", Linear::new(3, 1));
        // 2 Linears * (weight + bias) = 4
        assert_eq!(dict.parameters().len(), 4);
    }

    #[test]
    fn empty_containers_are_empty() {
        let l = ModuleList::new();
        let d = ModuleDict::new();
        assert!(l.is_empty());
        assert!(d.is_empty());
        assert_eq!(l.parameters().len(), 0);
        assert_eq!(d.parameters().len(), 0);
    }
}
