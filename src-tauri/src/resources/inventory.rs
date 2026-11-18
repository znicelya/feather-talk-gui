use super::{catalog::*, fail};
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
pub(crate) fn contained(root: &Path, path: &Path) -> Result<PathBuf> {
    let canonical = path.canonicalize().map_err(fail)?;
    if !canonical.starts_with(root.canonicalize().map_err(fail)?) {
        return Err(fail(format!(
            "path escapes resource root: {}",
            path.display()
        )));
    }
    Ok(canonical)
}
pub(crate) fn verify_file(root: &Path, rel: &Path, spec: &FileSpec) -> Result<()> {
    let path = contained(root, &root.join(rel))?;
    let mut file = fs::File::open(&path).map_err(fail)?;
    if !file.metadata().map_err(fail)?.is_file()
        || file.metadata().map_err(fail)?.len() != spec.size
    {
        return Err(fail(format!("size mismatch: {}", rel.display())));
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(fail)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    if format!("{:x}", hash.finalize()) != spec.sha256 {
        return Err(fail(format!("SHA-256 mismatch: {}", rel.display())));
    }
    Ok(())
}
pub(crate) fn verify(root: &Path, c: &Component) -> Result<()> {
    for (p, f) in c.files() {
        verify_file(root, &p, &f)?;
    }
    let raw = fs::read(root.join("manifest.json")).map_err(fail)?;
    let actual: AssetManifest = serde_json::from_slice(&raw).map_err(fail)?;
    if actual != c.manifest {
        return Err(fail("manifest differs from catalog"));
    }
    for p in &c.executables {
        executable(&root.join(p), c.target.as_ref().unwrap())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if fs::metadata(root.join(p))
                .map_err(fail)?
                .permissions()
                .mode()
                & 0o111
                == 0
            {
                return Err(fail("executable permission missing"));
            }
        }
    }
    Ok(())
}
/// Read binary headers rather than trusting a filename or catalog architecture.
fn executable(path: &Path, target: &TargetPlatform) -> Result<()> {
    let mut f = fs::File::open(path).map_err(fail)?;
    let mut h = [0u8; 64];
    f.read_exact(&mut h).map_err(fail)?;
    let (os, arch) = if &h[..2] == b"MZ" {
        let offset = u32::from_le_bytes(h[60..64].try_into().unwrap()) as u64;
        f.seek(SeekFrom::Start(offset)).map_err(fail)?;
        let mut pe = [0; 26];
        f.read_exact(&mut pe).map_err(fail)?;
        if &pe[..4] != b"PE\0\0" || u16::from_le_bytes([pe[24], pe[25]]) != 0x20b {
            return Err(fail("invalid PE executable"));
        }
        (
            "windows",
            match u16::from_le_bytes([pe[4], pe[5]]) {
                0x8664 => "x86_64",
                0xaa64 => "aarch64",
                _ => "unsupported",
            },
        )
    } else if &h[..4] == b"\x7fELF" && h[4] == 2 && h[5] == 1 {
        if ![2, 3].contains(&u16::from_le_bytes([h[16], h[17]])) {
            return Err(fail("ELF is not executable"));
        }
        (
            "linux",
            match u16::from_le_bytes([h[18], h[19]]) {
                62 => "x86_64",
                183 => "aarch64",
                _ => "unsupported",
            },
        )
    } else if h[..4] == [0xcf, 0xfa, 0xed, 0xfe] {
        if u32::from_le_bytes(h[12..16].try_into().unwrap()) != 2 {
            return Err(fail("Mach-O is not executable"));
        }
        (
            "macos",
            match u32::from_le_bytes(h[4..8].try_into().unwrap()) {
                0x01000007 => "x86_64",
                0x0100000c => "aarch64",
                _ => "unsupported",
            },
        )
    } else {
        return Err(fail("unsupported executable format"));
    };
    if os != target.os || arch != target.architecture {
        return Err(fail(format!(
            "executable target {os}/{arch} differs from {}/{}",
            target.os, target.architecture
        )));
    }
    Ok(())
}
