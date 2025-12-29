/// Panics execution indicating functionality is not supported
/// [*] Use when backend cannot implement specific feature
#[macro_export]
macro_rules! agnesslog_unsupported {
    ($($arg:tt)*) => {
        panic!("Agness Unsupported {}", format_args!($($arg)*));
    }
}

/// Prints warning message only in debug builds
/// [*] Compiled out in release builds for performance
#[macro_export]
macro_rules! agnesslog_warn {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            eprintln!("[Agness Warn] {}", format_args!($($arg)*));
        }
    }
}

/// Prints verbose debug information
/// [!] Massive performance impact due to syscalls
/// [*] Only active if agness_verbose feature is enabled
#[macro_export]
macro_rules! agnesslog_verbose {
    ($($arg:tt)*) => {
        #[cfg(feature = "agness_verbose")]
        {
            println!("[Agness Verbose] {}", format_args!($($arg)*));
        }
    }
}