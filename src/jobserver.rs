//! Cargo jobserver integration.
//!
//! Cargo hands every build script a jobserver: a pool of tokens, one per job
//! cargo may run at once (`-j`, by default the number of CPUs). `rustc` jobs
//! and build scripts already draw from it. Taking a token for every nvcc
//! process makes CUDA compilation share that one budget, so a build saturates
//! the machine without oversubscribing it, whatever else cargo is running.
//!
//! A build script owns one implicit token while it runs. The first nvcc job
//! uses it; every further job acquires a token from the pool.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Tokens for nvcc jobs, drawn from cargo's jobserver when one is present.
pub(crate) struct JobTokens {
    client: Option<jobserver::Client>,
    implicit_free: AtomicBool,
}

/// One job's claim on the CPU, released when dropped.
pub(crate) enum JobToken<'a> {
    /// The build script's own implicit token.
    Implicit(&'a AtomicBool),
    /// A token taken from the jobserver pool.
    Acquired(#[allow(dead_code)] jobserver::Acquired),
    /// No jobserver: concurrency is bounded by the thread pool alone.
    Unbounded,
}

impl Drop for JobToken<'_> {
    fn drop(&mut self) {
        if let JobToken::Implicit(free) = self {
            free.store(true, Ordering::Release);
        }
    }
}

impl JobTokens {
    /// The process-wide token source.
    ///
    /// Reads the jobserver from `CARGO_MAKEFLAGS`/`MAKEFLAGS` once.
    pub(crate) fn get() -> &'static JobTokens {
        static TOKENS: OnceLock<JobTokens> = OnceLock::new();
        TOKENS.get_or_init(|| {
            // SAFETY: called once, before this crate spawns any child process
            // or opens other descriptors that could alias the jobserver's.
            let client = unsafe { jobserver::Client::from_env() };
            JobTokens {
                client,
                implicit_free: AtomicBool::new(true),
            }
        })
    }

    /// Whether a jobserver bounds this build.
    pub(crate) fn is_active(&self) -> bool {
        self.client.is_some()
    }

    /// Block until this job may run.
    pub(crate) fn acquire(&self) -> io::Result<JobToken<'_>> {
        let Some(client) = &self.client else {
            return Ok(JobToken::Unbounded);
        };
        if self
            .implicit_free
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return Ok(JobToken::Implicit(&self.implicit_free));
        }
        client.acquire().map(JobToken::Acquired)
    }

    /// Take up to `max` more tokens without blocking, for a job that can use
    /// more than one core (nvcc `--threads`). Never blocks, so jobs holding
    /// one token can't deadlock waiting for a second.
    pub(crate) fn try_acquire_extra(&self, max: usize) -> Vec<JobToken<'_>> {
        let Some(client) = &self.client else {
            return Vec::new();
        };
        let mut extra = Vec::new();
        while extra.len() < max {
            match client.try_acquire() {
                Ok(Some(token)) => extra.push(JobToken::Acquired(token)),
                _ => break,
            }
        }
        extra
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(client: Option<jobserver::Client>) -> JobTokens {
        JobTokens {
            client,
            implicit_free: AtomicBool::new(true),
        }
    }

    #[test]
    fn without_a_jobserver_jobs_are_unbounded() {
        let t = tokens(None);
        assert!(!t.is_active());
        assert!(matches!(t.acquire().unwrap(), JobToken::Unbounded));
        assert!(t.try_acquire_extra(4).is_empty());
    }

    #[test]
    fn first_job_uses_the_implicit_token_then_draws_from_the_pool() {
        // A pool of 2 tokens plus the implicit one: 3 concurrent jobs.
        let t = tokens(Some(jobserver::Client::new(2).unwrap()));
        let a = t.acquire().unwrap();
        assert!(matches!(a, JobToken::Implicit(_)));
        let b = t.acquire().unwrap();
        let c = t.acquire().unwrap();
        assert!(matches!(b, JobToken::Acquired(_)));
        assert!(matches!(c, JobToken::Acquired(_)));
        // The pool is empty: no extra threads for anyone.
        assert!(t.try_acquire_extra(3).is_empty());
        drop(a);
        // The implicit token is free again.
        assert!(matches!(t.acquire().unwrap(), JobToken::Implicit(_)));
    }

    /// A client opened the way a build script opens cargo's: from
    /// `CARGO_MAKEFLAGS`. On Linux that reopens the pipe through `/dev/fd`,
    /// which supports non-blocking `try_acquire`; `Client::new` does not.
    #[cfg(target_os = "linux")]
    fn client_as_build_script(limit: usize) -> (jobserver::Client, jobserver::Client) {
        let parent = jobserver::Client::new(limit).unwrap();
        // `configure` sets exactly what cargo passes to a build script.
        let mut probe = std::process::Command::new("true");
        parent.configure(&mut probe);
        let auth = probe
            .get_envs()
            .find(|(k, _)| *k == "CARGO_MAKEFLAGS")
            .and_then(|(_, v)| v)
            .expect("configure sets CARGO_MAKEFLAGS")
            .to_owned();
        std::env::set_var("CARGO_MAKEFLAGS", &auth);
        // SAFETY: the parent client keeps the descriptors open.
        let child = unsafe { jobserver::Client::from_env_ext(false) }.client;
        std::env::remove_var("CARGO_MAKEFLAGS");
        (parent, child.unwrap())
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn extra_tokens_are_bounded_by_the_pool_and_the_request() {
        let (_parent, client) = client_as_build_script(3);
        let t = tokens(Some(client));
        let _job = t.acquire().unwrap();
        let first = t.try_acquire_extra(2);
        assert_eq!(first.len(), 2);
        assert_eq!(t.try_acquire_extra(5).len(), 1);
        drop(first);
        // Released extras return to the pool.
        assert_eq!(t.try_acquire_extra(5).len(), 3);
    }
}
