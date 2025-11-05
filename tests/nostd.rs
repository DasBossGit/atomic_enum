// Purpose: This integration test checks that atomic_enum can be used in
// a no_std environment.

#![no_std]

use core::sync::atomic::Ordering;

use ::atomic_enum::AtomicEnum;

#[derive(Debug, PartialEq, Eq, AtomicEnum, Copy, Clone)]
enum FooBar {
    Foo,
    Bar,
}

#[cfg(feature = "cas")]
#[test]
fn test_no_std_use() {
    let fb = AtomicFooBar::new(FooBar::Foo);
    let prev = fb
        .compare_exchange(
            FooBar::Foo,
            FooBar::Bar,
            Ordering::SeqCst,
            Ordering::Relaxed,
        )
        .unwrap();
    assert_eq!(prev, FooBar::Foo);

    let prev_fail = fb.compare_exchange(
        FooBar::Foo,
        FooBar::Bar,
        Ordering::SeqCst,
        Ordering::Relaxed,
    );
    assert!(prev_fail.is_err());
}

mod result_conflict {
    use core::sync::atomic::Ordering;

    use ::atomic_enum::AtomicEnum;

    #[allow(dead_code)]
    pub type Result<T> = core::result::Result<T, ()>;

    #[derive(Debug, PartialEq, Eq, AtomicEnum)]
    enum FooBar {
        Foo,
        Bar,
    }

    #[test]
    fn test_result_conflict() {
        let fb = AtomicFooBar::new(FooBar::Foo);
        assert_eq!(fb.load(Ordering::SeqCst), FooBar::Foo);
    }
}

#[test]
fn test_load_store() {
    let original = FooBar::Foo;
    let fb = AtomicFooBar::new(original);
    assert_eq!(fb.load(Ordering::SeqCst), original);

    let new = FooBar::Bar;
    fb.store(new, Ordering::SeqCst);
    assert_eq!(fb.load(Ordering::SeqCst), new);
}
