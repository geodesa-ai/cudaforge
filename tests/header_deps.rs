//! Header-only edits must rebuild the objects that include the header (issue: cudaforge's
//! incremental cache used to hash only the `.cu` file, so a changed header relinked stale
//! objects). Needs `nvcc`; skipped without it. Compiles for sm_80 (no GPU needed).

use cudaforge::KernelBuilder;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn have_nvcc() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cudaforge-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::create_dir_all(dir.join("out")).unwrap();
    dir
}

fn build(dir: &Path) {
    KernelBuilder::new()
        .source_files(vec![dir.join("src/k.cu")])
        .include_path(dir.join("src"))
        .compute_cap(80)
        .out_dir(dir.join("out"))
        .build_lib(dir.join("out/libk.a"))
        .expect("build_lib");
}

fn object(dir: &Path) -> PathBuf {
    fs::read_dir(dir.join("out"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "o"))
        .expect("object file")
}

fn mtime(p: &Path) -> SystemTime {
    fs::metadata(p).unwrap().modified().unwrap()
}

/// The SASS of the archive, so that the test checks what the kernel computes, not file times.
fn sass(dir: &Path) -> String {
    let out = Command::new("cuobjdump")
        .arg("-sass")
        .arg(dir.join("out/libk.a"))
        .output()
        .expect("cuobjdump");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn wait_for_mtime_tick() {
    std::thread::sleep(std::time::Duration::from_millis(20));
}

#[test]
fn header_only_edits_rebuild_and_no_op_builds_do_not() {
    if !have_nvcc() {
        eprintln!("skipped: nvcc not found");
        return;
    }
    let dir = scratch("header-deps");
    fs::write(
        dir.join("src/k.cu"),
        concat!(
            "#include \"h.cuh\"\n",
            "#if defined(__CUDA_ARCH__)\n#include \"dev_only.cuh\"\n#else\n#define DEV 0\n#endif\n",
            "extern \"C\" __global__ void k(int *o) { o[0] = K * 1000 + DEV; }\n",
        ),
    )
    .unwrap();
    fs::write(dir.join("src/h.cuh"), "#define K 1\n").unwrap();
    fs::write(dir.join("src/dev_only.cuh"), "#define DEV 7\n").unwrap();

    build(&dir);
    let obj = object(&dir);
    let first = sass(&dir);
    assert!(
        first.contains("0x3ef"),
        "K=1, DEV=7 -> 1007 = 0x3ef:\n{first}"
    );

    // No-op: nothing recompiles.
    let t0 = mtime(&obj);
    wait_for_mtime_tick();
    build(&dir);
    assert_eq!(mtime(&obj), t0, "a no-op build recompiled");

    // Header-only edit (the .cu is unchanged): the object must be rebuilt.
    fs::write(dir.join("src/h.cuh"), "#define K 2\n").unwrap();
    build(&dir);
    let second = sass(&dir);
    assert!(
        second.contains("0x7d7"),
        "K=2, DEV=7 -> 2007 = 0x7d7:\n{second}"
    );

    // A header reached only in the device pass (`#if __CUDA_ARCH__`).
    fs::write(dir.join("src/dev_only.cuh"), "#define DEV 9\n").unwrap();
    build(&dir);
    let third = sass(&dir);
    assert!(
        third.contains("0x7d9"),
        "K=2, DEV=9 -> 2009 = 0x7d9:\n{third}"
    );

    // And no-op again afterwards.
    let t1 = mtime(&object(&dir));
    wait_for_mtime_tick();
    build(&dir);
    assert_eq!(mtime(&object(&dir)), t1, "a no-op build recompiled");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn recorded_closure_lists_the_headers_and_not_the_outputs() {
    if !have_nvcc() {
        eprintln!("skipped: nvcc not found");
        return;
    }
    let dir = scratch("closure");
    fs::write(
        dir.join("src/k.cu"),
        "#include \"h.cuh\"\n__global__ void k() {}\n",
    )
    .unwrap();
    fs::write(dir.join("src/h.cuh"), "#pragma once\n").unwrap();
    build(&dir);
    let cache = cudaforge::BuildCache::load(&dir.join("out"));
    let obj = object(&dir);
    let deps = cache
        .deps_of(&dir.join("src/k.cu"), &obj)
        .expect("closure recorded");
    let paths: Vec<&str> = deps.iter().map(|d| d.path.as_str()).collect();
    assert!(paths.iter().any(|p| p.ends_with("src/k.cu")), "{paths:?}");
    assert!(paths.iter().any(|p| p.ends_with("src/h.cuh")), "{paths:?}");
    assert!(
        paths.iter().all(|p| !p.contains("/out/")),
        "outputs must not be inputs: {paths:?}"
    );
    let _ = fs::remove_dir_all(&dir);
}
