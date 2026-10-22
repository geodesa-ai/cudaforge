//! Dependency closures of nvcc compiles.
//!
//! An object depends on its `.cu` file and on every header that file includes, directly or not,
//! for the host pass and for the device pass of each requested architecture. `nvcc -M` with the
//! compile's own arguments lists exactly that set, including headers that are only reached under
//! `#if __CUDA_ARCH__ >= ...` for the requested architectures. The build cache records the set
//! with content hashes, and the build script tells cargo to rerun when any file of it changes.

use crate::hash::{DepHash, HashMemo};
use std::collections::HashSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The files `nvcc -M` reports for compiling `source` with `args` (the compile's arguments
/// without its mode and output flags), in nvcc's order without duplicates.
pub(crate) fn nvcc_closure(
    nvcc: &Path,
    args: &[OsString],
    source: &Path,
) -> std::result::Result<Vec<PathBuf>, String> {
    let output = Command::new(nvcc)
        .args(args)
        .arg("-M")
        .arg(source)
        .output()
        .map_err(|e| format!("running nvcc -M: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "nvcc -M failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .next()
                .unwrap_or("")
        ));
    }
    let text = String::from_utf8(output.stdout).map_err(|_| "nvcc -M output is not UTF-8")?;
    let mut seen = HashSet::new();
    let deps: Vec<PathBuf> = parse_make_deps(&text)
        .into_iter()
        .filter(|p| seen.insert(p.clone()))
        .collect();
    if !deps.iter().any(|d| d == source) {
        return Err(format!(
            "nvcc -M did not list {} among its dependencies",
            source.display()
        ));
    }
    Ok(deps)
}

/// Content hashes of `deps`. Fails, naming the file, if one cannot be read or its path is not
/// UTF-8; the object then has no recorded closure and is rebuilt every time.
pub(crate) fn hash_closure(
    deps: &[PathBuf],
    memo: &HashMemo,
) -> std::result::Result<Vec<DepHash>, String> {
    deps.iter()
        .map(|p| {
            let path = p
                .to_str()
                .ok_or_else(|| format!("dependency path is not UTF-8: {}", p.display()))?;
            let hash = memo
                .hash(p)
                .ok_or_else(|| format!("cannot read dependency {}", p.display()))?;
            Ok(DepHash {
                path: path.to_string(),
                hash,
            })
        })
        .collect()
}

/// Prerequisites of make rules (`target: dep dep \` ...), with `\ `, `\#`, `\\` and `$$`
/// unescaped. nvcc can print one rule per compilation pass; all prerequisites are returned.
pub(crate) fn parse_make_deps(text: &str) -> Vec<PathBuf> {
    let joined = text.replace("\\\r\n", " ").replace("\\\n", " ");
    let mut deps = Vec::new();
    for line in joined.lines() {
        let mut in_deps = false;
        for word in make_words(line) {
            if in_deps {
                deps.push(PathBuf::from(word));
            } else if word.ends_with(':') {
                in_deps = true;
            }
        }
    }
    deps
}

fn make_words(line: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if matches!(chars.peek(), Some(' ' | '#' | '\\')) => {
                word.push(chars.next().unwrap_or(c));
            }
            '$' if chars.peek() == Some(&'$') => {
                chars.next();
                word.push('$');
            }
            c if c.is_whitespace() => {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
            }
            c => word.push(c),
        }
    }
    if !word.is_empty() {
        words.push(word);
    }
    words
}

/// Directories whose files must not become `rerun-if-changed` inputs: the build's own outputs,
/// which are always newer than the build script's last run and would rerun it forever.
pub(crate) fn output_dirs(out_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![absolute(out_dir)];
    if let Some(env_out) = std::env::var_os("OUT_DIR").map(PathBuf::from) {
        // <target>/<profile>/build/<pkg>-<hash>/out -> <target>/<profile>
        if let Some(profile) = env_out.ancestors().nth(3) {
            dirs.push(absolute(profile));
        }
        dirs.push(absolute(&env_out));
    }
    dirs
}

fn absolute(path: &Path) -> PathBuf {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    std::fs::canonicalize(&path).unwrap_or(path)
}

/// A path in `OUT_DIR` that never exists. When some compile's closure could not be determined,
/// the build script declares it as an input, so cargo reruns the script on every build: a
/// header-only edit can then never be missed, even though that object's headers are unknown.
pub(crate) const CLOSURE_MISSING: &str = ".cudaforge-closure-missing";

/// Print `cargo:rerun-if-changed` for every file of the closures that exists and is not under
/// one of `excluded`, once each, plus the [`CLOSURE_MISSING`] sentinel in `closure_missing_in`
/// when a closure is missing.
pub(crate) fn emit_rerun_if_changed<'a>(
    deps: impl IntoIterator<Item = &'a str>,
    excluded: &[PathBuf],
    closure_missing_in: Option<&Path>,
) {
    for line in rerun_lines(deps, excluded, closure_missing_in) {
        println!("{line}");
    }
}

fn rerun_lines<'a>(
    deps: impl IntoIterator<Item = &'a str>,
    excluded: &[PathBuf],
    closure_missing_in: Option<&Path>,
) -> Vec<String> {
    let mut lines: Vec<String> = rerun_inputs(deps, excluded)
        .into_iter()
        .map(|p| format!("cargo:rerun-if-changed={p}"))
        .collect();
    if let Some(dir) = closure_missing_in {
        lines.push(format!(
            "cargo:rerun-if-changed={}",
            dir.join(CLOSURE_MISSING).display()
        ));
    }
    lines
}

fn rerun_inputs<'a>(deps: impl IntoIterator<Item = &'a str>, excluded: &[PathBuf]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for dep in deps {
        if !seen.insert(dep) {
            continue;
        }
        let path = Path::new(dep);
        if !path.exists() {
            continue;
        }
        let abs = absolute(path);
        if excluded.iter().any(|e| abs.starts_with(e)) {
            continue;
        }
        out.push(dep.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn parses_nvcc_make_output() {
        let text = "k.o : src/k.cu \\\n    src/a\\ b.cuh \\\n    /usr/include/stdio.h\nk.o: src/k.cu src/dev.cuh\n";
        assert_eq!(
            parse_make_deps(text),
            [
                PathBuf::from("src/k.cu"),
                PathBuf::from("src/a b.cuh"),
                PathBuf::from("/usr/include/stdio.h"),
                PathBuf::from("src/k.cu"),
                PathBuf::from("src/dev.cuh"),
            ]
        );
        assert!(parse_make_deps("").is_empty());
    }

    #[test]
    fn a_missing_closure_makes_the_build_script_rerun_every_time() {
        let out = std::env::temp_dir().join(format!("cudaforge-sentinel-{}", std::process::id()));
        let none = rerun_lines(std::iter::empty(), &[], None);
        assert!(none.is_empty());
        let lines = rerun_lines(std::iter::empty(), &[], Some(&out));
        assert_eq!(
            lines,
            [format!(
                "cargo:rerun-if-changed={}",
                out.join(CLOSURE_MISSING).display()
            )]
        );
        // The sentinel never exists, so cargo always considers the input changed.
        assert!(!out.join(CLOSURE_MISSING).exists());
    }

    #[test]
    fn unreadable_dependency_is_reported_by_name() {
        let root =
            std::env::temp_dir().join(format!("cudaforge-unreadable-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.cuh"), "x").unwrap();
        let memo = HashMemo::default();
        let ok = hash_closure(&[root.join("a.cuh")], &memo).unwrap();
        assert_eq!(ok.len(), 1);
        let err = hash_closure(&[root.join("a.cuh"), root.join("gone.cuh")], &memo).unwrap_err();
        assert!(err.contains("gone.cuh"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rerun_inputs_skip_outputs_missing_files_and_duplicates() {
        let root = std::env::temp_dir().join(format!("cudaforge-deps-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("out")).unwrap();
        fs::write(root.join("a.cuh"), "").unwrap();
        fs::write(root.join("out/gen.h"), "").unwrap();
        let a = root.join("a.cuh").to_string_lossy().into_owned();
        let generated = root.join("out/gen.h").to_string_lossy().into_owned();
        let missing = root.join("gone.h").to_string_lossy().into_owned();
        let inputs = rerun_inputs(
            [a.as_str(), generated.as_str(), missing.as_str(), a.as_str()],
            &[absolute(&root.join("out"))],
        );
        assert_eq!(inputs, [a]);
        let _ = fs::remove_dir_all(&root);
    }
}
