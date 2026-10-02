#![allow(unexpected_cfgs)] // multics is deliberately always false

use ::std::sync::atomic::AtomicU64;

use ::atomic_enum::AtomicEnum;

#[derive(Debug, AtomicEnum)]
enum MyEnum {
    Foo,
    #[cfg(target_os = "multics")]
    Bar,
    #[cfg(not(target_os = "multics"))]
    Baz,
}

// Foo and Baz should both be constructible.  Bar should not be, but that can only be verified from
// a doc test.
#[test]
fn construction() {
    let _ = AtomicMyEnum::new(MyEnum::Foo);
    let _ = AtomicMyEnum::new(MyEnum::Baz);
}

/* #[test] */
#[allow(unused)]
fn testscfgrs1318b653655843528864b5ad9f107016() {
    let a64 = AtomicU64::new(0);

    a64.try_update(
        ::std::sync::atomic::Ordering::SeqCst,
        ::std::sync::atomic::Ordering::SeqCst,
        |v| None,
    );
}
