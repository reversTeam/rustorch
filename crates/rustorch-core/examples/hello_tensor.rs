//! `hello_tensor` — minimal demo exercising the P0.3 prototype API.
//!
//! Run:    cargo run --release -p rustorch-core --example hello_tensor
//! WASM:   cargo build --target wasm32-unknown-unknown -p rustorch-core
//!         (this example is not run on wasm; only the library is built.)

use rustorch_core::{ops, Result, Tensor};

fn main() -> Result<()> {
    println!(
        "rustorch-core prototype — version {}",
        rustorch_core::VERSION
    );

    // 1. construct
    let a = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::ones([2usize, 3]);
    println!("a = {a}");
    println!("b = {b}");

    // 2. element-wise add
    let c = ops::add(&a, &b)?;
    println!("a + b = {c}");

    // 3. broadcast: scalar + matrix
    let s = Tensor::scalar(10.0);
    let d = ops::add(&s, &a)?;
    println!("scalar(10) + a = {d}");

    // 4. matmul
    let lhs = Tensor::from_vec([2usize, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let rhs = Tensor::from_vec([3usize, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;
    let out = ops::matmul(&lhs, &rhs)?;
    println!("lhs @ rhs = {out}");

    // 5. error path: shape mismatch surfaces a typed Error, no panic.
    // lhs is (2, 3); we pick a (4, 2) matrix → inner dims 3 vs 4 don't match.
    let bad = Tensor::from_vec([4usize, 2], vec![0.0; 8])?;
    match ops::matmul(&lhs, &bad) {
        Err(e) => println!("(expected) error: {e}"),
        Ok(_) => unreachable!("should have errored"),
    }

    Ok(())
}
