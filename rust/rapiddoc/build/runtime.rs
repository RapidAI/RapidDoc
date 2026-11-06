use sha2::{Digest, Sha256};
use std::{
    env,
    error::Error,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
const VERSION: &str = "1.23.2";

fn asset(target: &str, gpu: bool) -> Result<(&'static str, &'static str)> {
    Ok(match (target, gpu) {
        ("x86_64-pc-windows-msvc", false) => ("onnxruntime-win-x64-1.23.2.zip", "0b38df9af21834e41e73d602d90db5cb06dbd1ca618948b8f1d66d607ac9f3cd"),
        ("x86_64-pc-windows-msvc", true) => ("onnxruntime-win-x64-gpu-1.23.2.zip", "e77afdbbc2b8cb6da4e5a50d89841b48c44f3e47dce4fb87b15a2743786d0bb9"),
        ("x86_64-unknown-linux-gnu", false) => ("onnxruntime-linux-x64-1.23.2.tgz", "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b"),
        ("x86_64-unknown-linux-gnu", true) => ("onnxruntime-linux-x64-gpu-1.23.2.tgz", "2083e361072a79ce16a90dcd5f5cb3ab92574a82a3ce0ac01e5cfa3158176f53"),
        ("aarch64-unknown-linux-gnu", false) => ("onnxruntime-linux-aarch64-1.23.2.tgz", "7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c"),
        ("aarch64-pc-windows-msvc", false) => ("onnxruntime-win-arm64-1.23.2.zip", "1cfe88b6435df3b5fb0e9f6bd7d6f5df1e887b6174de7f6e2a47bab956f3f168"),
        ("aarch64-apple-darwin", false) => ("onnxruntime-osx-arm64-1.23.2.tgz", "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856"),
        ("x86_64-apple-darwin", false) => ("onnxruntime-osx-x86_64-1.23.2.tgz", "d10359e16347b57d9959f7e80a225a5b4a66ed7d7e007274a15cae86836485a6"),
        _ => return Err(format!("download-runtime does not support {target}; disable it and supply an external ORT library").into()),
    })
}
fn checksum(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn runtime_file(path: &Path) -> bool {
    let name = path.file_name().unwrap().to_string_lossy();
    name.ends_with(".dll")
        || name.ends_with(".dylib")
        || name.ends_with(".so")
        || name.contains(".so.")
}
fn copy_changed(source: &Path, target: &Path) -> Result<()> {
    if target.is_file() && checksum(source)? == checksum(target)? {
        return Ok(());
    }
    fs::copy(source, target)?;
    Ok(())
}
fn deliver_entry(bundle: &Path, archive_path: &Path, mut data: impl Read) -> Result<()> {
    let Some(name) = archive_path.file_name() else {
        return Ok(());
    };
    let text = name.to_string_lossy();
    let library = archive_path
        .parent()
        .and_then(Path::file_name)
        .is_some_and(|v| v == "lib")
        && runtime_file(archive_path);
    let notice = archive_path.components().count() == 2
        && (text.starts_with("LICENSE") || text.starts_with("ThirdPartyNotices"));
    if library || notice {
        // Flatten selected trusted archive files; never extract arbitrary paths
        // or the large PDB/static library/header payload to the build cache.
        let target = bundle.join(name);
        let temporary = bundle.join(format!("{}.{}.part", text, std::process::id()));
        let mut file = fs::File::create(&temporary)?;
        std::io::copy(&mut data, &mut file)?;
        drop(file);
        copy_changed(&temporary, &target)?;
        fs::remove_file(temporary)?;
    }
    Ok(())
}
pub fn install() -> Result<()> {
    let target = env::var("TARGET")?;
    let (name, expected) = asset(&target, env::var_os("CARGO_FEATURE_CUDA").is_some())?;
    let out = PathBuf::from(env::var_os("OUT_DIR").ok_or("Missing OUT_DIR")?);
    let archive = out.join(name);
    // An explicitly supplied archive supports offline builds; it must match
    // the exact same upstream checksum as an online download.
    if let Some(local) = env::var_os("RAPIDDOC_ORT_ARCHIVE") {
        println!("cargo:rerun-if-changed={}", Path::new(&local).display());
        fs::copy(local, &archive)?;
    } else if !archive.is_file() || checksum(&archive)? != expected {
        println!(
            "cargo:warning=Downloading {name} (ORT {VERSION}); subsequent builds reuse the archive"
        );
        let url =
            format!("https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/{name}");
        let temporary = out.join(format!("{name}.{}.part", std::process::id()));
        let mut file = fs::File::create(&temporary)?;
        let mut response = ureq::get(&url).call()?;
        std::io::copy(&mut response.body_mut().as_reader(), &mut file)?;
        file.flush()?;
        drop(file);
        if checksum(&temporary)? != expected {
            return Err("Downloaded ORT archive SHA-256 mismatch".into());
        }
        fs::rename(temporary, &archive)?;
    }
    if checksum(&archive)? != expected {
        return Err("ORT archive SHA-256 mismatch".into());
    }
    let bundle = out.join("ort-runtime");
    fs::create_dir_all(&bundle)?;
    if name.ends_with(".zip") {
        let mut zip = zip::ZipArchive::new(fs::File::open(&archive)?)?;
        for index in 0..zip.len() {
            let mut entry = zip.by_index(index)?;
            if entry.is_file() {
                let path = entry.enclosed_name().ok_or("Invalid ORT archive path")?;
                deliver_entry(&bundle, &path, &mut entry)?;
            }
        }
    } else {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(fs::File::open(&archive)?));
        for entry in tar.entries()? {
            let mut entry = entry?;
            if entry.header().entry_type().is_file() {
                let path = entry.path()?.into_owned();
                deliver_entry(&bundle, &path, &mut entry)?;
            }
        }
    }
    // Cargo examples/tests live in subdirectories of the profile directory.
    // Keep a deployable bundle there too, instead of exposing only OUT_DIR.
    let profile = out.ancestors().nth(3).ok_or("Unexpected Cargo OUT_DIR")?;
    for destination in [
        profile.join("runtime"),
        profile.join("examples/runtime"),
        profile.join("deps/runtime"),
    ] {
        fs::create_dir_all(&destination)?;
        for entry in fs::read_dir(&bundle)? {
            let path = entry?.path();
            let name = path.file_name().unwrap().to_string_lossy();
            if runtime_file(&path)
                || name.starts_with("LICENSE")
                || name.starts_with("ThirdPartyNotices")
            {
                copy_changed(&path, &destination.join(path.file_name().unwrap()))?;
            }
        }
    }
    // The SDK also finds this directory when the executable is moved during
    // development. Deployments should ship its contents alongside the binary.
    println!(
        "cargo:rustc-env=RAPIDDOC_BUNDLED_ORT_DIR={}",
        bundle.display()
    );
    Ok(())
}
