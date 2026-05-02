//! Procedural macros for rustorch-fusion.
//!
//! `fuse_call!` is a compact way to invoke the hand-written fused
//! kernels in `rustorch-fusion`. The macro is intentionally minimal
//! — it only expands a few well-known pipeline shapes; arbitrary op
//! chains stay as plain function calls.
//!
//! ## Supported forms
//!
//! ```ignore
//! // y = activation(x @ w + b) — single fused kernel
//! fuse_call!(matmul_bias_relu(x, w, bias) -> y; m=4, k=8, n=6);
//! fuse_call!(matmul_bias_gelu(x, w, bias) -> y; m=4, k=8, n=6);
//! fuse_call!(matmul_bias_silu(x, w, bias) -> y; m=4, k=8, n=6);
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::{parse_macro_input, Expr, Ident, LitInt, Result, Token};

struct FuseCall {
    op_name: Ident,
    x: Expr,
    w: Expr,
    bias: Expr,
    out: Expr,
    m: LitInt,
    k: LitInt,
    n: LitInt,
}

impl Parse for FuseCall {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        // op_name(x, w, bias) -> out; m=N, k=N, n=N
        let op_name: Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let x: Expr = content.parse()?;
        content.parse::<Token![,]>()?;
        let w: Expr = content.parse()?;
        content.parse::<Token![,]>()?;
        let bias: Expr = content.parse()?;
        input.parse::<Token![->]>()?;
        let out: Expr = input.parse()?;
        input.parse::<Token![;]>()?;
        // m=N, k=N, n=N (in any order)
        let mut m: Option<LitInt> = None;
        let mut k: Option<LitInt> = None;
        let mut n: Option<LitInt> = None;
        loop {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let v: LitInt = input.parse()?;
            match key.to_string().as_str() {
                "m" => m = Some(v),
                "k" => k = Some(v),
                "n" => n = Some(v),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown shape arg `{other}`, expected m / k / n"),
                    ))
                },
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
        }
        let m = m.ok_or_else(|| syn::Error::new(op_name.span(), "missing `m=N`"))?;
        let k = k.ok_or_else(|| syn::Error::new(op_name.span(), "missing `k=N`"))?;
        let n = n.ok_or_else(|| syn::Error::new(op_name.span(), "missing `n=N`"))?;
        Ok(Self {
            op_name,
            x,
            w,
            bias,
            out,
            m,
            k,
            n,
        })
    }
}

/// `fuse_call!(matmul_bias_<activation>(x, w, bias) -> out; m=M, k=K, n=N)`
/// expands to a call into rustorch-fusion's hand-written fused
/// matmul+bias+activation kernel.
#[proc_macro]
pub fn fuse_call(input: TokenStream) -> TokenStream {
    let FuseCall {
        op_name,
        x,
        w,
        bias,
        out,
        m,
        k,
        n,
    } = parse_macro_input!(input as FuseCall);
    let activation = match op_name.to_string().as_str() {
        "matmul_bias_relu" => quote! { ::rustorch_fusion::Activation::Relu },
        "matmul_bias_gelu" => quote! { ::rustorch_fusion::Activation::Gelu },
        "matmul_bias_silu" => quote! { ::rustorch_fusion::Activation::Silu },
        "matmul_bias" => quote! { ::rustorch_fusion::Activation::None },
        other => {
            let err = format!(
                "unknown fuse_call op `{other}` — supported: matmul_bias[_relu|_gelu|_silu]"
            );
            return syn::Error::new(op_name.span(), err)
                .to_compile_error()
                .into();
        },
    };
    let expanded = quote! {{
        ::rustorch_fusion::fused_matmul_bias_activation(
            &#x,
            &#w,
            Some(&#bias),
            &mut #out,
            #m,
            #k,
            #n,
            #activation,
        )
    }};
    expanded.into()
}
