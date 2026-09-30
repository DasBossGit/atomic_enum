#![forbid(future_incompatible, unsafe_code)]
#![warn(
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_import_braces,
    unused_qualifications
)]

//! An attribute to create an atomic wrapper around a C-style enum.
//!
//! Internally, the generated wrapper uses an appropriate Atomic to store the value.
//! The atomic operations have the same semantics as the equivalent operations
//! of its underlying Atomic type.
//!
//! The enum implies the `Copy` and `Clone` traits.
//!
//! # Example
//!
//! ```
//! # use atomic_enum::atomic_enum;
//! # use std::sync::atomic::Ordering;
//! #[atomic_enum]
//! #[derive(PartialEq)]
//! enum CatState {
//!     Dead = 0,
//!     BothDeadAndAlive,
//!     Alive,
//! }
//!
//! let state = AtomicCatState::new(CatState::Dead);
//! state.store(CatState::Alive, Ordering::Relaxed);
//!
//! assert_eq!(state.load(Ordering::Relaxed), CatState::Alive);
//! ```
//!
//! This attribute does not use or generate any unsafe code.
//!
//! The crate can be used in a `#[no_std]` environment.

use ::proc_macro2::TokenTree;
use ::simsearch::Index;
use ::syn::punctuated::Punctuated;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, quote_spanned};
use syn::{Ident, ItemEnum, Variant, Visibility, parse_macro_input, spanned::Spanned};

fn atomic_enum_definition(
    vis: &Visibility,
    ident: &Ident,
    atomic_ident: &Ident,
    size: TypeSize,
    derive: Option<TokenStream2>,
) -> TokenStream2 {
    let atomic_ident_docs = format!(
        "A wrapper around [`{0}`] which can be safely shared between threads.

This type uses an `{atomic_ty}` to store the enum value.

[`{0}`]: enum.{0}.html",
        ident,
        atomic_ty = match size {
            TypeSize::U8 => "AtomicU8",
            TypeSize::U16 => "AtomicU16",
            TypeSize::U32 => "AtomicU32",
            TypeSize::U64 => "AtomicU64",
            TypeSize::U128 => "AtomicU128",
            TypeSize::Usize => "AtomicUsize",
            TypeSize::I8 => "AtomicI8",
            TypeSize::I16 => "AtomicI16",
            TypeSize::I32 => "AtomicI32",
            TypeSize::I64 => "AtomicI64",
            TypeSize::I128 => "AtomicI128",
            TypeSize::Isize => "AtomicIsize",
        }
    );

    let atomic_type = match size {
        TypeSize::U8 => quote! { core::sync::atomic::AtomicU8 },
        TypeSize::U16 => quote! { core::sync::atomic::AtomicU16 },
        TypeSize::U32 => quote! { core::sync::atomic::AtomicU32 },
        TypeSize::U64 => quote! { core::sync::atomic::AtomicU64 },
        TypeSize::U128 => quote! { core::sync::atomic::AtomicU128 },
        TypeSize::Usize => quote! { core::sync::atomic::AtomicUsize },
        TypeSize::I8 => quote! { core::sync::atomic::AtomicI8 },
        TypeSize::I16 => quote! { core::sync::atomic::AtomicI16 },
        TypeSize::I32 => quote! { core::sync::atomic::AtomicI32 },
        TypeSize::I64 => quote! { core::sync::atomic::AtomicI64 },
        TypeSize::I128 => quote! { core::sync::atomic::AtomicI128 },
        TypeSize::Isize => quote! { core::sync::atomic::AtomicIsize },
    };

    let derive_clause = if let Some(derives) = derive {
        let derives = derives
            .into_iter()
            .filter_map(|tt| {
                if let TokenTree::Group(inner) = &tt {
                    let stream = inner.stream().into_iter().collect::<Vec<_>>();
                    if stream.len() == 0 {
                        None
                    } else {
                        Some(stream)
                    }
                } else {
                    Some(vec![tt])
                }
            })
            .flatten()
            .collect::<Vec<_>>();
        let derives = TokenStream2::from_iter(derives.into_iter());

        quote! { #[derive(#derives)] }
    } else {
        quote! {}
    };

    quote! {
        #[doc = #atomic_ident_docs]
        #derive_clause
        #vis struct #atomic_ident(#atomic_type);
    }
}

fn atomic_enum_new(ident: &Ident, atomic_ident: &Ident, repr: TypeSize) -> TokenStream2 {
    let atomic_ident_docs = format!(
        "Creates a new atomic [`{0}`].

[`{0}`]: enum.{0}.html",
        ident
    );

    let ty = match repr {
        TypeSize::U8 => quote! { u8 },
        TypeSize::U16 => quote! { u16 },
        TypeSize::U32 => quote! { u32 },
        TypeSize::U64 => quote! { u64 },
        TypeSize::U128 => quote! { u128 },
        TypeSize::Usize => quote! { usize },
        TypeSize::I8 => quote! { i8 },
        TypeSize::I16 => quote! { i16 },
        TypeSize::I32 => quote! { i32 },
        TypeSize::I64 => quote! { i64 },
        TypeSize::I128 => quote! { i128 },
        TypeSize::Isize => quote! { isize },
    };

    let atomic_ty = match repr {
        TypeSize::U8 => quote! { core::sync::atomic::AtomicU8 },
        TypeSize::U16 => quote! { core::sync::atomic::AtomicU16 },
        TypeSize::U32 => quote! { core::sync::atomic::AtomicU32 },
        TypeSize::U64 => quote! { core::sync::atomic::AtomicU64 },
        TypeSize::U128 => quote! { core::sync::atomic::AtomicU128 },
        TypeSize::Usize => quote! { core::sync::atomic::AtomicUsize },
        TypeSize::I8 => quote! { core::sync::atomic::AtomicI8 },
        TypeSize::I16 => quote! { core::sync::atomic::AtomicI16 },
        TypeSize::I32 => quote! { core::sync::atomic::AtomicI32 },
        TypeSize::I64 => quote! { core::sync::atomic::AtomicI64 },
        TypeSize::I128 => quote! { core::sync::atomic::AtomicI128 },
        TypeSize::Isize => quote! { core::sync::atomic::AtomicIsize },
    };

    quote! {
        #[doc = #atomic_ident_docs]
        pub const fn new(v: #ident) -> #atomic_ident {
            #atomic_ident(#atomic_ty::new(v as #ty))
        }
    }
}

fn atomic_enum_into_inner(ident: &Ident, from_repr: &TokenStream2) -> TokenStream2 {
    quote! {
        /// Consumes the atomic and returns the contained value.
        ///
        /// This is safe because passing self by value guarantees that no other threads are concurrently accessing the atomic data.
        pub const fn into_inner(self) -> #ident {
            unsafe { #from_repr(self.0.into_inner()) }
        }
    }
}

fn atomic_enum_set(ident: &Ident, to_repr: &TokenStream2) -> TokenStream2 {
    quote! {
        /// Sets the value of the atomic without performing an atomic operation.
        ///
        /// This is safe because the mutable reference guarantees that no other threads are concurrently accessing the atomic data.
        pub fn set(&mut self, v: #ident) {
            *self.0.get_mut() = v as #to_repr;
        }
    }
}

fn atomic_enum_get(ident: &Ident, from_repr: &TokenStream2) -> TokenStream2 {
    quote! {
        /// Gets the value of the atomic without performing an atomic operation.
        ///
        /// This is safe because the mutable reference guarantees that no other threads are concurrently accessing the atomic data.
        pub fn get(&mut self) -> #ident {
            unsafe { #from_repr(*self.0.get_mut()) }
        }
    }
}

fn atomic_enum_swap_mut(ident: &Ident) -> TokenStream2 {
    quote! {
        /// Stores a value into the atomic, returning the previous value, without performing an atomic operation.
        ///
        /// This is safe because the mutable reference guarantees that no other threads are concurrently accessing the atomic data.
        pub fn swap_mut(&mut self, v: #ident) -> #ident {
            let r = self.get();
            self.set(v);
            r
        }
    }
}

fn atomic_enum_load(ident: &Ident, from_repr: &TokenStream2) -> TokenStream2 {
    quote! {
        /// Loads a value from the atomic.
        ///
        /// `load` takes an `Ordering` argument which describes the memory ordering of this operation. Possible values are `SeqCst`, `Acquire` and `Relaxed`.
        ///
        /// # Panics
        ///
        /// Panics if order is `Release` or `AcqRel`.
        pub fn load(&self, order: core::sync::atomic::Ordering) -> #ident {
            unsafe { #from_repr(self.0.load(order)) }
        }
    }
}

fn atomic_enum_store(ident: &Ident, to_repr: &TokenStream2) -> TokenStream2 {
    quote! {
        /// Stores a value into the atomic.
        ///
        /// `store` takes an `Ordering` argument which describes the memory ordering of this operation. Possible values are `SeqCst`, `Release` and `Relaxed`.
        ///
        /// # Panics
        ///
        /// Panics if order is `Acquire` or `AcqRel`.
        pub fn store(&self, val: #ident, order: core::sync::atomic::Ordering) {
            self.0.store(val as #to_repr, order)
        }
    }
}

#[cfg(feature = "cas")]
fn atomic_enum_swap(
    ident: &Ident,
    to_repr: &TokenStream2,
    from_repr: &TokenStream2,
) -> TokenStream2 {
    quote! {
        /// Stores a value into the atomic, returning the previous value.
        ///
        /// `swap` takes an `Ordering` argument which describes the memory ordering of this operation.
        /// All ordering modes are possible. Note that using `Acquire` makes the store part of this operation `Relaxed`,
        /// and using `Release` makes the load part `Relaxed`.
        pub fn swap(&self, val: #ident, order: core::sync::atomic::Ordering) -> #ident {
            unsafe { #from_repr(self.0.swap(val as #to_repr, order)) }
        }
    }
}

#[cfg(feature = "cas")]
fn atomic_enum_compare_exchange(
    ident: &Ident,
    to_repr: &TokenStream2,
    from_repr: &TokenStream2,
) -> TokenStream2 {
    quote! {
        /// Stores a value into the atomic if the current value is the same as the `current` value.
        ///
        /// The return value is a result indicating whether the new value was written and containing the previous value.
        /// On success this value is guaranteed to be equal to `current`.
        ///
        /// `compare_exchange` takes two `Ordering` arguments to describe the memory ordering of this operation. The first
        /// describes the required ordering if the operation succeeds while the second describes the required ordering when
        /// the operation fails. Using `Acquire` as success ordering makes the store part of this operation `Relaxed`, and
        /// using `Release` makes the successful load `Relaxed`. The failure ordering can only be `SeqCst`, `Acquire` or
        /// `Relaxed` and must be equivalent to or weaker than the success ordering.
        pub fn compare_exchange(
            &self,
            current: #ident,
            new: #ident,
            success: core::sync::atomic::Ordering,
            failure: core::sync::atomic::Ordering
        ) -> core::result::Result<#ident, #ident> {
            self.0
                .compare_exchange(
                    current as #to_repr,
                    new as #to_repr,
                    success,
                    failure
                )
                .map(|v| unsafe { #from_repr(v) })
                .map_err(|v| unsafe { #from_repr(v) })
        }
    }
}

#[cfg(feature = "cas")]
fn atomic_enum_compare_exchange_weak(
    ident: &Ident,
    to_repr: &TokenStream2,
    from_repr: &TokenStream2,
) -> TokenStream2 {
    quote! {
        /// Stores a value into the atomic if the current value is the same as the `current` value.
        ///
        /// Unlike `compare_exchange`, this function is allowed to spuriously fail even when the comparison succeeds,
        /// which can result in more efficient code on some platforms. The return value is a result indicating whether
        /// the new value was written and containing the previous value.
        ///
        /// `compare_exchange_weak` takes two `Ordering` arguments to describe the memory ordering of this operation.
        /// The first describes the required ordering if the operation succeeds while the second describes the required
        /// ordering when the operation fails. Using `Acquire` as success ordering makes the store part of this operation
        /// `Relaxed`, and using `Release` makes the successful load `Relaxed`. The failure ordering can only be `SeqCst`,
        /// `Acquire` or `Relaxed` and must be equivalent to or weaker than the success ordering.
        pub fn compare_exchange_weak(
            &self,
            current: #ident,
            new: #ident,
            success: core::sync::atomic::Ordering,
            failure: core::sync::atomic::Ordering
        ) -> core::result::Result<#ident, #ident> {
            self.0
                .compare_exchange_weak(
                    current as #to_repr,
                    new as #to_repr,
                    success,
                    failure
                )
                .map(|v| unsafe { #from_repr(v) })
                .map_err(|v| unsafe { #from_repr(v) })
        }
    }
}

fn from_impl(ident: &Ident, atomic_ident: &Ident) -> TokenStream2 {
    quote! {
        impl From<#ident> for #atomic_ident {
            fn from(val: #ident) -> #atomic_ident {
                #atomic_ident::new(val)
            }
        }
    }
}

fn debug_impl(ident: &Ident, atomic_ident: &Ident) -> TokenStream2 {
    quote! {
        impl core::fmt::Debug for #atomic_ident where for<'__atomic_enum_debug__> #ident: core::fmt::Debug {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                core::fmt::Debug::fmt(&self.load(core::sync::atomic::Ordering::SeqCst), f)
            }
        }
    }
}
fn default_impl(ident: &Ident, atomic_ident: &Ident) -> TokenStream2 {
    quote! {
        impl Default for #atomic_ident where for<'__atomic_enum_default__> #ident: Default {
            fn default() -> Self {
                #atomic_ident::new(Default::default())
            }
        }
    }
}

enum Identifer {
    Lit(syn::Lit),
    Ident(Ident),
    Path(syn::Path),
}

impl ToString for Identifer {
    fn to_string(&self) -> String {
        match self {
            Self::Lit(lit) => match lit {
                ::syn::Lit::Str(lit_str) => lit_str.value(),
                ::syn::Lit::ByteStr(lit_byte_str) => String::from_utf8(lit_byte_str.value())
                    .ok()
                    .unwrap_or_else(|| String::from_utf8_lossy(&lit_byte_str.value()).into_owned()),
                ::syn::Lit::CStr(lit_cstr) => lit_cstr.value().to_str().ok().map_or_else(
                    || lit_cstr.value().to_string_lossy().into_owned(),
                    |s| s.to_string(),
                ),
                ::syn::Lit::Byte(lit_byte) => (lit_byte.value() as char).to_string(),
                ::syn::Lit::Char(lit_char) => lit_char.value().to_string(),
                ::syn::Lit::Int(lit_int) => lit_int.token().to_string(),
                ::syn::Lit::Float(lit_float) => lit_float.token().to_string(),
                ::syn::Lit::Bool(lit_bool) => lit_bool.value.to_string(),
                ::syn::Lit::Verbatim(literal) => literal.to_string(),
                _ => String::new(),
            },
            Self::Ident(ident) => ident.to_string(),
            Self::Path(path) => quote! { #path }.to_string(),
        }
    }
}

impl syn::parse::Parse for Identifer {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if let Ok(ident) = input.parse::<Ident>() {
            Ok(Identifer::Ident(ident))
        } else if let Ok(lit) = input.parse::<syn::Lit>() {
            Ok(Identifer::Lit(lit))
        } else if let Ok(path) = input.parse::<syn::Path>() {
            Ok(Identifer::Path(path))
        } else {
            Err(syn::Error::new(
                input.span(),
                "Expected identifier, literal or path",
            ))
        }
    }
}

enum Assignment {
    Colon,
    Equal,
}

impl syn::parse::Parse for Assignment {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(syn::Token![=]) {
            let _eq_token: syn::Token![=] = input.parse()?;
            Ok(Assignment::Equal)
        } else if input.peek(syn::Token![:]) {
            let _dc_token: syn::Token![:] = input.parse()?;
            Ok(Assignment::Colon)
        } else {
            Err(input.error("Expected '=' or '::'"))
        }
    }
}

impl Assignment {
    fn peek(input: syn::parse::ParseStream) -> bool {
        input.peek(syn::Token![=]) || input.peek(syn::Token![:])
    }
}

enum OptionType {
    KeyValue(String, TokenStream2),
    Flag(String),
}

impl OptionType {
    fn key(&self) -> &str {
        match self {
            OptionType::KeyValue(key, _) => key,
            OptionType::Flag(key) => key,
        }
    }

    fn into_value(self) -> Option<TokenStream2> {
        match self {
            OptionType::KeyValue(_, value) => Some(value),
            OptionType::Flag(_) => None,
        }
    }
}

impl syn::parse::Parse for OptionType {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Any Ident, Literal or Path is acceptable as a key
        if input.is_empty() {
            return Err(syn::Error::new(
                input.span(),
                "Expected identifier, literal or path as option key",
            ));
        }

        let key = input.parse::<Identifer>()?.to_string();
        if Assignment::peek(input) {
            while Assignment::peek(input) {
                let _assignment: Assignment = input.parse()?;
            }

            let mut value = Vec::new();

            if input.is_empty() {
                return Err(syn::Error::new(
                    input.span(),
                    "Expected value after assignment",
                ));
            }

            while !input.is_empty() && !Assignment::peek(input) && !input.peek(syn::Token![,]) {
                let tt: TokenTree = input.parse()?;
                value.push(tt);
            }
            let value = TokenStream2::from_iter(value.into_iter());
            Ok(OptionType::KeyValue(key, value))
        } else {
            Ok(OptionType::Flag(key))
        }
    }
}

struct AtomicWrapperOptions {
    atomic_name: Option<Ident>,
    derive: Option<TokenStream2>,
    underlying_type_size: Option<TypeSize>,
    no_debug: bool,
    no_default: bool,
}

impl AtomicWrapperOptions {
    const ATOMIC_NAMES: &[&'static str] = &[
        "atomic_name",
        "name",
        "ident",
        "identifier",
        "atomic_ident",
        "atomic_identifier",
    ];
    const DERIVE_NAMES: &[&'static str] = &[
        "derive",
        "derives",
        "auto_derive",
        "auto-derive",
        "autoderive",
    ];
    const SIZE_NAMES: &[&'static str] = &[
        "size",
        "repr",
        "representation",
        "underlying_type",
        "underlyingtype",
    ];
    const NO_DEBUG: &[&'static str] = &["no_debug", "nodebug", "no-debug"];
    const NO_DEFAULT: &[&'static str] = &["no_default", "nodefault", "no-default"];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Keys {
    AtomicName,
    Derive,
    Size,
    NoDebug,
    NoDefault,
}

impl syn::parse::Parse for AtomicWrapperOptions {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut fuzzy_engine = Index::<Keys>::new();
        fuzzy_engine.insert_parts(Keys::AtomicName, AtomicWrapperOptions::ATOMIC_NAMES);
        fuzzy_engine.insert_parts(Keys::Derive, AtomicWrapperOptions::DERIVE_NAMES);
        fuzzy_engine.insert_parts(Keys::Size, AtomicWrapperOptions::SIZE_NAMES);
        fuzzy_engine.insert_parts(Keys::NoDebug, AtomicWrapperOptions::NO_DEBUG);
        fuzzy_engine.insert_parts(Keys::NoDefault, AtomicWrapperOptions::NO_DEFAULT);

        let mut atomic_name: Option<Ident> = None;
        let mut derive: Option<TokenStream2> = None;
        let mut underlying_type_size: Option<TypeSize> = None;
        let mut no_debug: bool = false;
        let mut no_default: bool = false;

        // Parse arguments as Punctuated<Expr, Comma>

        let punctuated = input.parse_terminated(OptionType::parse, syn::Token![,])?;

        for option in punctuated {
            let key_alphanumeric = option
                .key()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>();
            let key_str = option.key().to_string();

            let matches = fuzzy_engine.search(&key_alphanumeric).into_iter().next();

            if let Some(key) = matches {
                match (key.id, option.into_value()) {
                    (Keys::NoDebug, None) => {
                        no_debug = true;
                    }
                    (Keys::NoDebug, Some(_)) => {
                        return Err(syn::Error::new(
                            input.span(),
                            format!("Unexpected value for option key: {}", key_str),
                        ));
                    }
                    (Keys::NoDefault, None) => {
                        no_default = true;
                    }
                    (Keys::NoDefault, Some(_)) => {
                        return Err(syn::Error::new(
                            input.span(),
                            format!("Unexpected value for option key: {}", key_str),
                        ));
                    }
                    (Keys::AtomicName, None) => {
                        return Err(syn::Error::new(
                            input.span(),
                            format!("Expected value for option key: {}", key_str),
                        ));
                    }
                    (Keys::AtomicName, Some(tt)) => {
                        if atomic_name.is_some() {
                            return Err(syn::Error::new(
                                input.span(),
                                format!("Duplicate option key: {}", key_str),
                            ));
                        }

                        // Parse tt as Identifier
                        let span = tt.span();
                        if let Ok(ident) = syn::parse2::<Identifer>(tt) {
                            let value = ident.to_string();
                            atomic_name = Some(Ident::new(&value, span));
                        } else {
                            return Err(syn::Error::new(
                                span,
                                format!("Expected a valid identifier for option key: {}", key_str),
                            ));
                        }
                    }
                    (Keys::Derive, None) => {
                        return Err(syn::Error::new(
                            input.span(),
                            format!("Expected value for option key: {}", key_str),
                        ));
                    }
                    (Keys::Derive, Some(tt)) => {
                        derive.as_mut().map(|prev| prev.extend(quote! { , }));
                        derive.get_or_insert_with(|| TokenStream2::new()).extend(tt);
                    }
                    (Keys::Size, None) => {
                        return Err(syn::Error::new(
                            input.span(),
                            format!("Expected value for option key: {}", key_str),
                        ));
                    }
                    (Keys::Size, Some(tt)) => {
                        if underlying_type_size.is_some() {
                            /* return Err(syn::Error::new(
                                input.span(),
                                format!("Duplicate option key: {}", key_str),
                            )); */
                        }
                        // Parse tt as Identifier
                        let span = tt.span();
                        if let Ok(ident) = syn::parse2::<Identifer>(tt) {
                            let value = ident.to_string().to_lowercase();
                            let type_size = match value.as_str() {
                                "u8" => TypeSize::U8,
                                "u16" => TypeSize::U16,
                                "u32" => TypeSize::U32,
                                "u64" => TypeSize::U64,
                                "u128" => TypeSize::U128,
                                "usize" => TypeSize::Usize,
                                "i8" => TypeSize::I8,
                                "i16" => TypeSize::I16,
                                "i32" => TypeSize::I32,
                                "i64" => TypeSize::I64,
                                "i128" => TypeSize::I128,
                                "isize" => TypeSize::Isize,
                                _ => {
                                    return Err(syn::Error::new(
                                        span,
                                        format!(
                                            "Invalid underlying type size for option key: {}",
                                            key_str
                                        ),
                                    ));
                                }
                            };
                            underlying_type_size = Some(type_size);
                        } else {
                            return Err(syn::Error::new(
                                span,
                                format!("Expected a valid identifier for option key: {}", key_str),
                            ));
                        }
                    }
                }
            } else {
                return Err(syn::Error::new(
                    input.span(),
                    format!("Unknown option key: {}", option.key()),
                ));
            }
        }

        Ok(AtomicWrapperOptions {
            atomic_name,
            derive,
            underlying_type_size,
            no_debug,
            no_default,
        })
    }
}

fn disc_as_usize(expr: Option<&(syn::token::Eq, ::syn::Expr)>) -> Option<usize> {
    if let Some((_, expr)) = expr {
        if let syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(lit_int),
            ..
        }) = expr
        {
            if let Ok(value) = lit_int.base10_parse::<usize>() {
                return Some(value);
            }
        }
    }
    None
}

fn min_type_size(variants: &Punctuated<Variant, syn::token::Comma>) -> TypeSize {
    let max_discriminant = variants.len() - 1;

    let variants_disc = variants
        .iter()
        .enumerate()
        .map(|(idx, v)| disc_as_usize(v.discriminant.as_ref()).unwrap_or(idx))
        .max()
        .unwrap_or(max_discriminant)
        .max(max_discriminant);

    if variants_disc <= u8::MAX as usize {
        TypeSize::U8
    } else if variants_disc <= u16::MAX as usize {
        TypeSize::U16
    } else if variants_disc <= u32::MAX as usize {
        if size_of::<usize>() == 4 {
            TypeSize::Usize
        } else {
            TypeSize::U32
        }
    } else {
        if size_of::<usize>() == 8 {
            TypeSize::Usize
        } else {
            TypeSize::U64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TypeSize {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
}

#[proc_macro_derive(AtomicEnum, attributes(atomic_enum))]
/// Creates an atomic wrapper around a C-style enum.
///
/// The generated type is a wrapper around a suitable atomic integer type that transparently
/// converts between the stored integer and the enum type. This attribute
/// also automatically derives the `Copy`, `Clone`, `AsRef<underlying type>`, `AsRef<underlying int type>`, `Deref<underlying type>` and `Deref<underlying int type>` traits for
/// the enum type.
///
/// The name of the atomic type is the name of the enum type, prefixed with
/// `Atomic`.
///
/// ```
/// # use atomic_enum::atomic_enum;
/// #[atomic_enum]
/// enum State {
///     On,
///     Off,
/// }
///
/// let state = AtomicState::new(State::Off);
/// ```
///
/// The name can be overridden by passing an identifier
/// as an argument to the attribute.
///
/// ```
/// # use atomic_enum::atomic_enum;
/// #[atomic_enum(StateAtomic)]
/// enum State {
///     On,
///     Off,
/// }
///
/// let state = StateAtomic::new(State::Off);
/// ```
pub fn atomic_enum(input: TokenStream) -> TokenStream {
    // Parse the input
    let input = parse_macro_input!(input as ItemEnum);
    let ItemEnum {
        attrs,
        vis,
        ident,
        generics,
        variants,
        ..
    } = &input;

    if variants.len() < 2 {
        let span = ident.span();
        let err = quote_spanned! { span => compile_error!("Expected an enum with at least two variants."); };
        return err.into();
    }

    let repr = attrs
        .iter()
        .find_map(|attr| {
            if attr.path().is_ident("repr") {
                // Try parse as repr attribute (excluding C style, packed, align)
                let mut repr_size = None;
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("u8") {
                        repr_size = Some(TypeSize::U8);
                    } else if meta.path.is_ident("u16") {
                        repr_size = Some(TypeSize::U16);
                    } else if meta.path.is_ident("u32") {
                        repr_size = Some(TypeSize::U32);
                    } else if meta.path.is_ident("u64") {
                        repr_size = Some(TypeSize::U64);
                    } else if meta.path.is_ident("u128") {
                        repr_size = Some(TypeSize::U128);
                    } else if meta.path.is_ident("usize") {
                        repr_size = Some(TypeSize::Usize);
                    } else if meta.path.is_ident("i8") {
                        repr_size = Some(TypeSize::I8);
                    } else if meta.path.is_ident("i16") {
                        repr_size = Some(TypeSize::I16);
                    } else if meta.path.is_ident("i32") {
                        repr_size = Some(TypeSize::I32);
                    } else if meta.path.is_ident("i64") {
                        repr_size = Some(TypeSize::I64);
                    } else if meta.path.is_ident("i128") {
                        repr_size = Some(TypeSize::I128);
                    } else if meta.path.is_ident("isize") {
                        repr_size = Some(TypeSize::Isize);
                    }
                    Ok(())
                });
                repr_size
            } else {
                None
            }
        })
        .or_else(|| Some(min_type_size(&variants)))
        // Default to u64 or usize
        .unwrap_or(TypeSize::Usize);

    // We only support C-style enums: No generics, no fields
    if !generics.params.is_empty() {
        let span = generics.span();
        let err = quote_spanned! {span => compile_error!("Expected an enum without generics."); };
        return err.into();
    }

    for variant in variants.iter() {
        if !matches!(variant.fields, syn::Fields::Unit) {
            let span = variant.fields.span();
            let err =
                quote_spanned! {span=> compile_error!("Expected a variant without fields."); };
            return err.into();
        }
    }

    // Define the enum
    let mut output = TokenStream2::new();

    // Define the atomic wrapper
    let args = input
        .attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("atomic_enum") {
                Some(attr)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if args.len() > 1 {
        let span = args[1].span();
        let err = quote_spanned! {span=> compile_error!("Duplicate 'atomic_enum' attribute."); };
        return err.into();
    }

    let options = if let Some(attr) = args.into_iter().next() {
        if attr.meta.require_path_only().is_ok() {
            AtomicWrapperOptions {
                atomic_name: None,
                derive: None,
                underlying_type_size: None,
                no_debug: false,
                no_default: false,
            }
        } else {
            match attr.parse_args_with(|input: syn::parse::ParseStream| {
                if input.is_empty() {
                    return Ok(AtomicWrapperOptions {
                        atomic_name: None,
                        derive: None,
                        underlying_type_size: None,
                        no_debug: false,
                        no_default: false,
                    });
                }
                input.parse::<AtomicWrapperOptions>()
            }) {
                Ok(opts) => opts,
                Err(err) => return err.to_compile_error().into(),
            }
        }
    } else {
        AtomicWrapperOptions {
            atomic_name: None,
            derive: None,
            underlying_type_size: None,
            no_debug: false,
            no_default: false,
        }
    };

    let repr = if let Some(size) = options.underlying_type_size {
        size
    } else {
        repr
    };

    let atomic_ident = options
        .atomic_name
        .unwrap_or_else(|| Ident::new(&format!("Atomic{}", ident), ident.span()));

    output.extend(atomic_enum_definition(
        &vis,
        &ident,
        &atomic_ident,
        repr,
        options.derive,
    ));

    // Write the impl block for the atomic wrapper
    let atomic_enum_new = atomic_enum_new(&ident, &atomic_ident, repr);

    let from_repr = quote! { core::mem::transmute::<_, #ident> };

    let to_repr = match repr {
        TypeSize::U8 => quote! { u8 },
        TypeSize::U16 => quote! { u16 },
        TypeSize::U32 => quote! { u32 },
        TypeSize::U64 => quote! { u64 },
        TypeSize::U128 => quote! { u128 },
        TypeSize::Usize => quote! { usize },
        TypeSize::I8 => quote! { i8 },
        TypeSize::I16 => quote! { i16 },
        TypeSize::I32 => quote! { i32 },
        TypeSize::I64 => quote! { i64 },
        TypeSize::I128 => quote! { i128 },
        TypeSize::Isize => quote! { isize },
    };

    let atomic_enum_into_inner = atomic_enum_into_inner(&ident, &from_repr);
    let atomic_enum_set = atomic_enum_set(&ident, &to_repr);
    let atomic_enum_get = atomic_enum_get(&ident, &from_repr);
    let atomic_enum_swap_mut = atomic_enum_swap_mut(&ident);
    let atomic_enum_load = atomic_enum_load(&ident, &from_repr);
    let atomic_enum_store = atomic_enum_store(&ident, &to_repr);

    output.extend(quote! {
        impl #atomic_ident {
            #atomic_enum_new
            #atomic_enum_into_inner
            #atomic_enum_set
            #atomic_enum_get
            #atomic_enum_swap_mut
            #atomic_enum_load
            #atomic_enum_store
        }
    });

    #[cfg(feature = "cas")]
    {
        let atomic_enum_swap = atomic_enum_swap(&ident, &to_repr, &from_repr);
        let atomic_enum_compare_exchange =
            atomic_enum_compare_exchange(&ident, &to_repr, &from_repr);
        let atomic_enum_compare_exchange_weak =
            atomic_enum_compare_exchange_weak(&ident, &to_repr, &from_repr);

        output.extend(quote! {
            impl #atomic_ident {
                #atomic_enum_swap
                #atomic_enum_compare_exchange
                #atomic_enum_compare_exchange_weak
            }
        });
    }

    // Implement the from and debug traits
    output.extend(from_impl(&ident, &atomic_ident));
    if !options.no_debug {
        output.extend(debug_impl(&ident, &atomic_ident));
    }
    if !options.no_default {
        output.extend(default_impl(&ident, &atomic_ident));
    }

    output.into()
}
