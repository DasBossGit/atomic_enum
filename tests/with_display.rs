use core::sync::atomic::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};

use ::atomic_enum::AtomicEnum;

#[derive(Debug, PartialEq, Eq, AtomicEnum)]
enum DisplayableEnum {
    Foo,
    Bar,
}

impl Display for DisplayableEnum {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DisplayableEnum::Foo => write!(f, "Foo"),
            DisplayableEnum::Bar => write!(f, "Bar"),
        }
    }
}

#[test]
fn test_displayable_enum() {
    let e = AtomicDisplayableEnum::new(DisplayableEnum::Foo);
    assert_eq!(format!("{}", e.load(Ordering::SeqCst)), "Foo");
}
