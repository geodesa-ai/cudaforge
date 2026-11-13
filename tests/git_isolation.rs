//! A dependency fetch must only ever touch its own clone.
//!
//! `git bisect run` inside a linked worktree exports `GIT_DIR=<repo>/.git/worktrees/<wt>` to the
//! command it runs; cargo passes it to build scripts. Git honours `GIT_DIR` over the working
//! directory, so every git call cudaforge made "in" the cached clone ran against the enclosing
//! worktree instead: `rev-parse HEAD` read the worktree's HEAD (cache miss), `sparse-checkout set`
//! rewrote the worktree's sparse-checkout and turned on `extensions.worktreeConfig` in the main
//! repository's config, and `fetch origin <sha>` asked the worktree's remote for the dependency
//! commit. These tests run the fetch with that environment and check the enclosing repository is
//! byte-for-byte untouched, for a cache hit, a failed fetch, and a half-created cache entry.
//!
//! One test function: it sets process environment (`GIT_DIR`, `CUDAFORGE_HOME`) and must not run
//! concurrently with another test in this binary.

use cudaforge::ExternalDependency;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const GIT_ENV: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_PREFIX",
];

fn git(dir: &Path, args: &[&str]) -> String {
    let mut cmd = Command::new("git");
    for var in GIT_ENV {
        cmd.env_remove(var);
    }
    let out = cmd
        .args([
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?} in {}: {}",
        dir.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn have_git() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Every file of a git directory that a stray `config`/`sparse-checkout` write would change.
fn snapshot(git_dirs: &[PathBuf]) -> Vec<(PathBuf, Option<Vec<u8>>)> {
    let mut files = Vec::new();
    for dir in git_dirs {
        for name in [
            "config",
            "config.worktree",
            "info/sparse-checkout",
            "HEAD",
            "index",
        ] {
            let p = dir.join(name);
            files.push((p.clone(), fs::read(&p).ok()));
        }
    }
    files
}

#[test]
fn fetch_never_touches_the_enclosing_repository() {
    if !have_git() {
        eprintln!("git not found: skipped");
        return;
    }
    let root = std::env::temp_dir().join(format!("cudaforge-gitiso-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();

    // The dependency's upstream: one commit with an include directory.
    let upstream = root.join("upstream");
    fs::create_dir_all(upstream.join("include")).unwrap();
    fs::write(upstream.join("include/dep.h"), "#pragma once\n").unwrap();
    git(&root, &["init", "-q", upstream.to_str().unwrap()]);
    git(&upstream, &["config", "uploadpack.allowFilter", "true"]);
    git(
        &upstream,
        &["config", "uploadpack.allowAnySHA1InWant", "true"],
    );
    git(&upstream, &["add", "."]);
    git(&upstream, &["commit", "-q", "-m", "dep"]);
    let dep_commit = git(&upstream, &["rev-parse", "HEAD"]);
    let url = format!("file://{}", upstream.display());

    // The enclosing repository with a linked worktree: the shape `git bisect run` runs in.
    let main = root.join("main");
    fs::create_dir_all(&main).unwrap();
    git(&root, &["init", "-q", main.to_str().unwrap()]);
    fs::write(main.join("README"), "main\n").unwrap();
    git(&main, &["add", "."]);
    git(&main, &["commit", "-q", "-m", "main"]);
    let wt = root.join("wt");
    git(
        &main,
        &["worktree", "add", "-q", "--detach", wt.to_str().unwrap()],
    );
    let wt_git_dir = PathBuf::from(git(&wt, &["rev-parse", "--absolute-git-dir"]));
    let watched = [main.join(".git"), wt_git_dir.clone()];
    let before = snapshot(&watched);

    let cache = root.join("cudaforge-home");
    std::env::set_var("CUDAFORGE_HOME", &cache);
    let out_dir = root.join("out");
    fs::create_dir_all(&out_dir).unwrap();
    let dep =
        |commit: &str| ExternalDependency::git("dep", &url, commit, vec!["include"], vec![], false);

    // 1. First fetch with the leaked GIT_DIR: clones into the cache, not into the worktree.
    std::env::set_var("GIT_DIR", &wt_git_dir);
    let dir = dep(&dep_commit)
        .fetch(&out_dir)
        .expect("fetch with leaked GIT_DIR");
    assert!(
        dir.join("include/dep.h").exists(),
        "dependency checked out in its own clone"
    );
    assert_eq!(
        snapshot(&watched),
        before,
        "clone touched the enclosing repository"
    );

    // 2. Cache hit with the leaked GIT_DIR (the reported failure began here: rev-parse read the
    //    worktree's HEAD, so the hit was missed and sparse-checkout/fetch ran on the worktree).
    let again = dep(&dep_commit)
        .fetch(&out_dir)
        .expect("cache hit with leaked GIT_DIR");
    assert_eq!(again, dir);
    assert_eq!(
        snapshot(&watched),
        before,
        "cache hit touched the enclosing repository"
    );

    // 3. A commit upstream does not have: the fetch fails, and nothing outside the clone changed.
    let missing = "0123456789abcdef0123456789abcdef01234567";
    let err = ExternalDependency::git("dep", &url, missing, vec!["include"], vec![], false)
        .fetch(&out_dir);
    assert!(err.is_err(), "fetching a missing commit must fail");
    assert_eq!(
        snapshot(&watched),
        before,
        "failed fetch touched the enclosing repository"
    );

    // 4. A half-created cache entry (directory without .git, as an interrupted clone leaves) INSIDE
    //    the enclosing worktree, with no GIT_DIR: git would discover the worktree upward. The
    //    fetch must re-clone instead of running git in it.
    std::env::remove_var("GIT_DIR");
    let nested_cache = wt.join("nested-cudaforge-home");
    std::env::set_var("CUDAFORGE_HOME", &nested_cache);
    let half = nested_cache
        .join("git/checkouts")
        .join(format!("dep-{}", &dep_commit[..16]));
    fs::create_dir_all(half.join("include")).unwrap();
    let dir = dep(&dep_commit)
        .fetch(&out_dir)
        .expect("fetch over a half-created entry");
    assert!(dir.join(".git").is_dir() && dir.join("include/dep.h").exists());
    assert_eq!(
        snapshot(&watched),
        before,
        "half-created entry: git ran in the enclosing worktree"
    );

    std::env::remove_var("CUDAFORGE_HOME");
    let _ = fs::remove_dir_all(&root);
}
