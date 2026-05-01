//! Module ↔ tensor-map conversion (P1.6 + P1.8 glue).
//!
//! [`state_dict`] walks a [`Module`] tree and returns a flat
//! BTreeMap keyed by dotted path (e.g. `"layers.0.weight"`).
//! [`load_state_dict`] is the inverse: copy tensors from a map back
//! into the module's parameters.
//!
//! Order of operations is determined by [`Module::named_parameters`],
//! which composite modules (Sequential, ModuleList, ModuleDict) override
//! to recurse into children with prefixed names.

use crate::module::Module;
use rustorch_autograd::Variable;
use rustorch_core::tensor::tensor_impl::Tensor;
use std::collections::BTreeMap;

/// Errors during state-dict load.
#[derive(Debug, thiserror::Error)]
pub enum StateDictError {
    /// Strict mode: a key expected by the module is absent from `sd`.
    #[error("missing key (strict): {0}")]
    MissingKey(String),
    /// Strict mode: a key in `sd` doesn't correspond to any parameter.
    #[error("unexpected key (strict): {0}")]
    UnexpectedKey(String),
    /// Source tensor shape doesn't match the destination parameter.
    #[error("shape mismatch for {key}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Parameter name.
        key: String,
        /// Expected shape (from module).
        expected: Vec<usize>,
        /// Actual shape (from sd).
        got: Vec<usize>,
    },
}

/// Walk a Module tree and return a flat dotted-path → tensor map.
pub fn state_dict<M: Module + ?Sized>(module: &M) -> BTreeMap<String, Tensor> {
    module
        .named_parameters()
        .into_iter()
        .map(|(name, v)| (name, v.tensor().clone()))
        .collect()
}

/// Report from a non-strict load: keys present in the module but absent
/// from `sd`, and keys in `sd` that didn't match any parameter.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Names expected by the module but missing from the input map.
    pub missing: Vec<String>,
    /// Names provided in the input map but not consumed by the module.
    pub unexpected: Vec<String>,
}

/// Copy tensors from `sd` into the module's parameters in-place.
///
/// `strict=true` returns `Err` on any missing or unexpected key.
/// `strict=false` returns a [`LoadReport`] listing them. Shape
/// mismatches always return `Err`.
pub fn load_state_dict<M: Module + ?Sized>(
    module: &M,
    sd: &BTreeMap<String, Tensor>,
    strict: bool,
) -> Result<LoadReport, StateDictError> {
    let mut report = LoadReport::default();
    let module_params: BTreeMap<String, Variable> = module.named_parameters().into_iter().collect();

    for (name, dst) in &module_params {
        match sd.get(name) {
            Some(src) => {
                let dst_shape = dst.tensor().shape().to_vec();
                let src_shape = src.shape().to_vec();
                if dst_shape != src_shape {
                    return Err(StateDictError::ShapeMismatch {
                        key: name.clone(),
                        expected: dst_shape,
                        got: src_shape,
                    });
                }
                // In-place copy: replace the Variable's data buffer.
                dst.set_data(src.clone());
            },
            None => {
                if strict {
                    return Err(StateDictError::MissingKey(name.clone()));
                }
                report.missing.push(name.clone());
            },
        }
    }

    // Detect unexpected keys (in sd but not in module).
    for sd_key in sd.keys() {
        if !module_params.contains_key(sd_key) {
            if strict {
                return Err(StateDictError::UnexpectedKey(sd_key.clone()));
            }
            report.unexpected.push(sd_key.clone());
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LayerNorm, Linear, ModuleDict, ModuleList, Relu, Sequential};

    #[test]
    fn linear_state_dict_has_weight_and_bias_keys() {
        let l = Linear::new(3, 2);
        let sd = state_dict(&l);
        let keys: Vec<&str> = sd.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["bias", "weight"]); // BTreeMap → alphabetical
        assert_eq!(sd["weight"].shape(), &[3, 2]);
        assert_eq!(sd["bias"].shape(), &[2]);
    }

    #[test]
    fn linear_no_bias_state_dict_only_weight() {
        let l = Linear::no_bias(3, 2);
        let sd = state_dict(&l);
        assert_eq!(sd.len(), 1);
        assert!(sd.contains_key("weight"));
        assert!(!sd.contains_key("bias"));
    }

    #[test]
    fn sequential_state_dict_uses_dot_indexed_paths() {
        let net = Sequential::new()
            .add(Linear::new(4, 4))
            .add(Relu)
            .add(Linear::new(4, 2));
        let sd = state_dict(&net);
        let keys: Vec<&str> = sd.keys().map(String::as_str).collect();
        // BTreeMap-sorted: 0.bias, 0.weight, 2.bias, 2.weight
        assert_eq!(keys, vec!["0.bias", "0.weight", "2.bias", "2.weight"]);
    }

    #[test]
    fn module_list_state_dict_indexed_paths() {
        let list = ModuleList::new()
            .push(Linear::new(2, 2))
            .push(Linear::new(2, 2));
        let sd = state_dict(&list);
        let keys: Vec<&str> = sd.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["0.bias", "0.weight", "1.bias", "1.weight"]);
    }

    #[test]
    fn module_dict_state_dict_named_paths() {
        let dict = ModuleDict::new()
            .insert("encoder", Linear::new(8, 4))
            .insert("decoder", Linear::new(4, 8));
        let sd = state_dict(&dict);
        let keys: Vec<&str> = sd.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            vec![
                "decoder.bias",
                "decoder.weight",
                "encoder.bias",
                "encoder.weight"
            ]
        );
    }

    #[test]
    fn nested_state_dict_layer_norm_in_sequential() {
        let net = Sequential::new()
            .add(Linear::new(4, 4))
            .add(LayerNorm::new(4));
        let sd = state_dict(&net);
        let keys: Vec<&str> = sd.keys().map(String::as_str).collect();
        // 0.{bias,weight} for Linear; 1.{beta,gamma} for LayerNorm
        assert_eq!(keys, vec!["0.bias", "0.weight", "1.beta", "1.gamma"]);
    }

    #[test]
    fn round_trip_state_dict_preserves_values() {
        let l1 = Linear::new(3, 2);
        let mut sd_before = state_dict(&l1);
        // Patch the weight to a known value for verification.
        let new_w =
            Tensor::from_vec([3usize, 2], vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        sd_before.insert("weight".to_string(), new_w.clone());

        let l2 = Linear::new(3, 2);
        let report = load_state_dict(&l2, &sd_before, true).unwrap();
        assert_eq!(report.missing.len(), 0);
        assert_eq!(report.unexpected.len(), 0);

        let sd_after = state_dict(&l2);
        assert_eq!(
            sd_after["weight"].as_slice::<f32>().unwrap(),
            new_w.as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn strict_load_errors_on_missing_key() {
        let l = Linear::new(3, 2);
        let mut sd = state_dict(&l);
        sd.remove("weight"); // make it missing
        let l2 = Linear::new(3, 2);
        assert!(matches!(
            load_state_dict(&l2, &sd, true),
            Err(StateDictError::MissingKey(_))
        ));
    }

    #[test]
    fn non_strict_load_returns_missing_report() {
        let l = Linear::new(3, 2);
        let mut sd = state_dict(&l);
        sd.remove("bias");
        let l2 = Linear::new(3, 2);
        let report = load_state_dict(&l2, &sd, false).unwrap();
        assert_eq!(report.missing, vec!["bias".to_string()]);
        assert!(report.unexpected.is_empty());
    }

    #[test]
    fn strict_load_errors_on_unexpected_key() {
        let l = Linear::new(3, 2);
        let mut sd = state_dict(&l);
        sd.insert(
            "phantom".to_string(),
            Tensor::from_vec([1usize], vec![0.0_f32]).unwrap(),
        );
        let l2 = Linear::new(3, 2);
        assert!(matches!(
            load_state_dict(&l2, &sd, true),
            Err(StateDictError::UnexpectedKey(_))
        ));
    }

    #[test]
    fn shape_mismatch_returns_err() {
        let l = Linear::new(3, 2);
        let mut sd = state_dict(&l);
        sd.insert(
            "weight".to_string(),
            Tensor::from_vec([2usize, 3], vec![0.0_f32; 6]).unwrap(),
        );
        let l2 = Linear::new(3, 2);
        assert!(matches!(
            load_state_dict(&l2, &sd, false),
            Err(StateDictError::ShapeMismatch { .. })
        ));
    }
}
