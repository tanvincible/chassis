//! Errors from `VectorIndex` and `IndexReader` (ADR-0020): a kind a program can match on, and a
//! message that says what was wrong, with the value, and what to do instead. The message is
//! written for whoever reads it, a person or an agent, in words that fit every language Chassis
//! is used from.

use std::fmt;

/// What went wrong, for a program to act on. Bindings map it: C's `chassis_last_error_code`,
/// Python's exception classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An argument is out of range or malformed: dimensions, an option, an id, a vector with a
    /// component that isn't a finite number.
    InvalidArgument,
    /// A vector or query has another number of components than the index's vectors, or the file
    /// holds vectors of other dimensions than asked for.
    DimensionMismatch,
    /// The file was created with another metric or precision than the one asked for.
    OptionsMismatch,
    /// There is no index at the path, or its directory doesn't exist.
    NotFound,
    /// The file is not a Chassis index.
    NotAnIndex,
    /// The id is already in use.
    IdInUse,
    /// Another writer has the index open.
    Locked,
    /// The handle is a reader, which only searches.
    ReadOnly,
    /// The file is damaged.
    Corrupt,
    /// The index can hold no more.
    Full,
    /// The operating system refused: permissions, a full disk, a failed read or write.
    Io,
    /// Anything else.
    Other,
}

/// An error from `VectorIndex` or `IndexReader`. It converts into `anyhow::Error` and any
/// `Box<dyn Error>`, so `?` works where those are returned.
pub struct Error {
    kind: ErrorKind,
    inner: anyhow::Error,
}

impl Error {
    /// What went wrong.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

/// `Result` with Chassis's `Error`.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    /// The message, then each cause after it: "Can't open …: Permission denied (os error 13)".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#}", self.inner)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {self}", self.kind)
    }
}

// The message already holds every cause, so none is reported again as a source.
impl std::error::Error for Error {}

impl From<anyhow::Error> for Error {
    fn from(inner: anyhow::Error) -> Self {
        let raised = inner.chain().find_map(|cause| cause.downcast_ref::<Raised>());
        let kind = match raised {
            Some(raised) => raised.kind,
            None if inner.chain().any(|cause| cause.is::<std::io::Error>()) => ErrorKind::Io,
            None => ErrorKind::Other,
        };
        Self { kind, inner }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        anyhow::Error::from(e).into()
    }
}

/// An error raised with its kind, which the conversion to `Error` finds however much context
/// was added around it.
#[derive(Debug)]
pub(crate) struct Raised {
    kind: ErrorKind,
    message: String,
}

impl fmt::Display for Raised {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Raised {}

/// An `anyhow::Error` that will convert to an `Error` of `kind`.
pub(crate) fn raised(kind: ErrorKind, message: String) -> anyhow::Error {
    anyhow::Error::new(Raised { kind, message })
}

/// Returns an error of a kind, with a formatted message, from a function returning either
/// `anyhow::Result` or Chassis's `Result`.
macro_rules! fail {
    ($kind:ident, $($message:tt)+) => {
        return Err($crate::error::raised(
            $crate::error::ErrorKind::$kind,
            format!($($message)+),
        )
        .into())
    };
}
pub(crate) use fail;

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;

    fn raise() -> anyhow::Result<()> {
        fail!(Locked, "{} is open", "x.chassis");
    }

    #[test]
    fn test_a_kind_survives_context_and_the_message_keeps_every_cause() {
        let error: Error = raise().context("Opening failed").unwrap_err().into();
        assert_eq!(error.kind(), ErrorKind::Locked);
        assert_eq!(error.to_string(), "Opening failed: x.chassis is open");
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let error: Error = anyhow::Error::from(io).context("Can't open x").into();
        assert_eq!(
            (error.kind(), error.to_string().as_str()),
            (ErrorKind::Io, "Can't open x: denied")
        );
        let error: Error = anyhow::anyhow!("something").into();
        assert_eq!(error.kind(), ErrorKind::Other);
        // It goes into anyhow with `?`.
        let into: anyhow::Result<()> = (|| Err(Error::from(anyhow::anyhow!("x")))?)();
        assert!(into.is_err());
    }
}
