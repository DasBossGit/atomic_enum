use atomic_enum::atomic_enum;

#[derive(Debug)]
#[repr(usize)]
#[atomic_enum(name = Test)]
enum MyEnum {
    Foo,
    Bar,
    Baz = 1024,
}

// Foo and Baz should both be constructible.  Bar should not be, but that can only be verified from
// a doc test.
#[test]
fn construction() {
    let _ = Test::new(MyEnum::Foo);
    let _ = Test::new(MyEnum::Baz);

    let inner = {
        let this = &mut Test::new(MyEnum::Baz);
        let v = MyEnum::Foo;
        *this.0.get_mut() = Test::to_usize(v);
    };
}
