// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "params.rs"
// ============================================================================
//! Saving and loading model parameters by name.
//!
//! A checkpoint is a safetensors file with one tensor per [`Parameter`], named by the
//! parameter's own name, plus two metadata entries that identify the format and its
//! version. Loading is strict and all or nothing: every parameter must be in the file
//! with the same shape, the file must hold nothing else, and nothing is changed unless
//! all of that holds.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use midman_foundation::{Error, Result};
use midman_tensor::Parameter;

use crate::safetensors::{NamedTensor, TensorFile};

/// The metadata key that holds [`FORMAT_NAME`].
pub const FORMAT_KEY: &str = "format";

/// The value of [`FORMAT_KEY`] in every file this crate writes.
pub const FORMAT_NAME: &str = "midman-checkpoint";

/// The metadata key that holds the checkpoint version.
pub const VERSION_KEY: &str = "checkpoint_version";

/// The checkpoint version this build writes and reads.
pub const CHECKPOINT_VERSION: u32 = 1;

/// How many names an error message lists before it says "and N more".
const NAMES_SHOWN: usize = 5;

/// Builds a checkpoint file holding `params` and the caller's `metadata`.
///
/// The format and version entries are added here, so `metadata` must not use those two
/// keys. Values are copied.
///
/// # Errors
///
/// Returns an error if `metadata` uses a reserved key or two parameters share a name.
pub fn parameters_to_file(
    params: &[&Parameter],
    metadata: &BTreeMap<String, String>,
) -> Result<TensorFile> {
    for reserved in [FORMAT_KEY, VERSION_KEY] {
        if metadata.contains_key(reserved) {
            return Err(Error::invalid(format!(
                "metadata key `{reserved}` is set by the checkpoint writer"
            )));
        }
    }
    let mut file = TensorFile::new();
    for (key, value) in metadata {
        file.set_metadata(key.as_str(), value.as_str());
    }
    file.set_metadata(FORMAT_KEY, FORMAT_NAME);
    file.set_metadata(VERSION_KEY, CHECKPOINT_VERSION.to_string());
    for param in params {
        file.push(param.name(), param.shape(), param.data())?;
    }
    Ok(file)
}

/// Writes a checkpoint of `params` to `path`, replacing the file atomically.
///
/// See [`parameters_to_file`] and [`TensorFile::save`].
///
/// # Errors
///
/// Returns an error for any of the reasons those two functions do.
pub fn save_parameters(
    path: impl AsRef<Path>,
    params: &[&Parameter],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    parameters_to_file(params, metadata)?.save(path)
}

/// Checks that `file` is a checkpoint this build can read.
///
/// # Errors
///
/// Returns an error if the format entry is missing or names another format, if the
/// version entry is missing or is not a whole number, or if the version is not
/// [`CHECKPOINT_VERSION`].
pub fn check_header(file: &TensorFile) -> Result<()> {
    match file.metadata().get(FORMAT_KEY) {
        Some(name) if name == FORMAT_NAME => {}
        Some(other) => {
            return Err(Error::parse(
                "checkpoint",
                0,
                format!("`{FORMAT_KEY}` is `{other}`, not `{FORMAT_NAME}`"),
            ));
        }
        None => {
            return Err(Error::parse(
                "checkpoint",
                0,
                format!("the file has no `{FORMAT_KEY}` entry, so it is not a {FORMAT_NAME} file"),
            ));
        }
    }
    let Some(text) = file.metadata().get(VERSION_KEY) else {
        return Err(Error::parse(
            "checkpoint",
            0,
            format!("the file has no `{VERSION_KEY}` entry"),
        ));
    };
    let found: u32 = text.parse().map_err(|_| {
        Error::parse("checkpoint", 0, format!("`{VERSION_KEY}` is `{text}`, not a whole number"))
    })?;
    if found != CHECKPOINT_VERSION {
        return Err(Error::UnsupportedVersion {
            what: "checkpoint".to_string(),
            found,
            supported: CHECKPOINT_VERSION,
        });
    }
    Ok(())
}

/// Copies the values in `file` into `params`, matching by name.
///
/// Nothing is changed unless the whole load can succeed.
///
/// # Errors
///
/// Returns an error if the header check fails (see [`check_header`]), if two
/// parameters share a name, if a parameter is missing from the file or has a different
/// shape there, or if the file holds a tensor that no parameter wants.
pub fn load_parameters(file: &TensorFile, params: &mut [&mut Parameter]) -> Result<()> {
    check_header(file)?;

    let mut wanted: HashSet<&str> = HashSet::new();
    for param in params.iter() {
        if !wanted.insert(param.name()) {
            return Err(Error::invalid(format!("two parameters are named `{}`", param.name())));
        }
    }
    let by_name: HashMap<&str, &NamedTensor> =
        file.tensors().iter().map(|tensor| (tensor.name(), tensor)).collect();

    let mut missing: Vec<&str> = Vec::new();
    for param in params.iter() {
        match by_name.get(param.name()) {
            None => missing.push(param.name()),
            Some(tensor) if tensor.shape() != param.shape() => {
                return Err(Error::shape(
                    "load_parameters",
                    format!(
                        "`{}` has shape {:?} in the checkpoint but {:?} in the model",
                        param.name(),
                        tensor.shape(),
                        param.shape()
                    ),
                ));
            }
            Some(_) => {}
        }
    }
    if !missing.is_empty() {
        return Err(Error::invalid(format!(
            "the checkpoint is missing {} tensor(s) the model needs: {}",
            missing.len(),
            list_names(&missing)
        )));
    }
    let unexpected: Vec<&str> = file
        .tensors()
        .iter()
        .map(NamedTensor::name)
        .filter(|name| !wanted.contains(name))
        .collect();
    if !unexpected.is_empty() {
        return Err(Error::invalid(format!(
            "the checkpoint holds {} tensor(s) the model does not have: {}",
            unexpected.len(),
            list_names(&unexpected)
        )));
    }

    for param in params.iter_mut() {
        if let Some(tensor) = by_name.get(param.name()) {
            param.set_data(tensor.data())?;
        }
    }
    Ok(())
}

fn list_names(names: &[&str]) -> String {
    let shown: Vec<String> = names.iter().take(NAMES_SHOWN).map(|n| format!("`{n}`")).collect();
    let mut text = shown.join(", ");
    if names.len() > NAMES_SHOWN {
        text.push_str(&format!(" and {} more", names.len() - NAMES_SHOWN));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use midman_tensor::Tensor;

    fn param(name: &str, shape: &[usize], start: f32) -> Parameter {
        let len: usize = shape.iter().product();
        let values: Vec<f32> = (0..len).map(|i| start + i as f32).collect();
        Parameter::new(name, Tensor::from_vec(values, shape).unwrap())
    }

    fn pair() -> (Parameter, Parameter) {
        (param("a", &[2, 2], 1.0), param("b", &[3], 10.0))
    }

    fn file_of(params: &[&Parameter]) -> TensorFile {
        parameters_to_file(params, &BTreeMap::new()).unwrap()
    }

    #[test]
    fn writes_the_format_and_version_entries_and_keeps_the_callers_metadata() {
        let (a, b) = pair();
        let mut metadata = BTreeMap::new();
        metadata.insert("step".to_string(), "40".to_string());
        let file = parameters_to_file(&[&a, &b], &metadata).unwrap();
        assert_eq!(file.metadata().get(FORMAT_KEY).map(String::as_str), Some("midman-checkpoint"));
        assert_eq!(file.metadata().get(VERSION_KEY).map(String::as_str), Some("1"));
        assert_eq!(file.metadata().get("step").map(String::as_str), Some("40"));
        assert_eq!(file.tensors().len(), 2);
        assert!(check_header(&file).is_ok());
    }

    #[test]
    fn the_writer_owns_the_reserved_metadata_keys() {
        let (a, _) = pair();
        for key in [FORMAT_KEY, VERSION_KEY] {
            let mut metadata = BTreeMap::new();
            metadata.insert(key.to_string(), "x".to_string());
            assert!(parameters_to_file(&[&a], &metadata).is_err(), "{key}");
        }
    }

    #[test]
    fn duplicate_parameter_names_are_rejected_on_save_and_on_load() {
        let (a, _) = pair();
        let twin = param("a", &[2, 2], 5.0);
        assert!(parameters_to_file(&[&a, &twin], &BTreeMap::new()).is_err());
        let file = file_of(&[&a]);
        let (mut x, mut y) = (param("a", &[2, 2], 0.0), param("a", &[2, 2], 0.0));
        assert!(load_parameters(&file, &mut [&mut x, &mut y]).is_err());
    }

    #[test]
    fn loading_copies_every_value_by_name_in_any_order() {
        let (a, b) = pair();
        let file = file_of(&[&a, &b]);
        let (mut a2, mut b2) = (param("a", &[2, 2], -1.0), param("b", &[3], -1.0));
        // The parameters are given in the opposite order to the file.
        load_parameters(&file, &mut [&mut b2, &mut a2]).unwrap();
        assert_eq!(a2.data(), a.data());
        assert_eq!(b2.data(), b.data());
    }

    #[test]
    fn a_failed_load_changes_nothing() {
        let (a, b) = pair();
        let mut broken = TensorFile::new();
        for (key, value) in file_of(&[&a, &b]).metadata() {
            broken.set_metadata(key.as_str(), value.as_str());
        }
        broken.push("a", &[2, 2], a.data()).unwrap();
        // `b` is missing, but `a` is fine and comes first.
        let (mut a2, mut b2) = (param("a", &[2, 2], -1.0), param("b", &[3], -1.0));
        let before = (a2.data().to_vec(), b2.data().to_vec());
        let err = load_parameters(&broken, &mut [&mut a2, &mut b2]).unwrap_err().to_string();
        assert!(err.contains("missing") && err.contains("`b`"), "{err}");
        assert_eq!((a2.data().to_vec(), b2.data().to_vec()), before);
    }

    #[test]
    fn a_shape_mismatch_names_both_shapes() {
        let (a, b) = pair();
        let file = file_of(&[&a, &b]);
        let (mut a2, mut b2) = (param("a", &[4], 0.0), param("b", &[3], 0.0));
        let err = load_parameters(&file, &mut [&mut a2, &mut b2]).unwrap_err().to_string();
        assert!(err.contains("`a`") && err.contains("[2, 2]") && err.contains("[4]"), "{err}");
    }

    #[test]
    fn extra_tensors_in_the_file_are_an_error() {
        let (a, b) = pair();
        let file = file_of(&[&a, &b]);
        let mut a2 = param("a", &[2, 2], 0.0);
        let err = load_parameters(&file, &mut [&mut a2]).unwrap_err().to_string();
        assert!(err.contains("does not have") && err.contains("`b`"), "{err}");
    }

    #[test]
    fn long_name_lists_are_cut_short() {
        let names: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        assert_eq!(list_names(&refs), "`p0`, `p1`, `p2`, `p3`, `p4` and 3 more");
        assert_eq!(list_names(&refs[..2]), "`p0`, `p1`");
    }

    #[test]
    fn the_header_check_rejects_foreign_missing_and_future_files() {
        let (a, _) = pair();
        let mut foreign = TensorFile::new();
        foreign.push("a", &[1], &[1.0]).unwrap();
        assert!(check_header(&foreign).is_err());

        let mut wrong_format = file_of(&[&a]);
        wrong_format.set_metadata(FORMAT_KEY, "pt");
        assert!(check_header(&wrong_format).is_err());

        let mut no_version = TensorFile::new();
        no_version.set_metadata(FORMAT_KEY, FORMAT_NAME);
        assert!(check_header(&no_version).is_err());

        let mut not_a_number = file_of(&[&a]);
        not_a_number.set_metadata(VERSION_KEY, "one");
        assert!(check_header(&not_a_number).is_err());

        let mut past = file_of(&[&a]);
        past.set_metadata(VERSION_KEY, "0");
        assert!(matches!(check_header(&past), Err(Error::UnsupportedVersion { found: 0, .. })));

        let mut future = file_of(&[&a]);
        future.set_metadata(VERSION_KEY, "2");
        match check_header(&future) {
            Err(Error::UnsupportedVersion { found: 2, supported: 1, .. }) => {}
            other => panic!("expected an unsupported version, got {other:?}"),
        }
        let mut target = param("a", &[2, 2], 0.0);
        assert!(load_parameters(&future, &mut [&mut target]).is_err());
    }
}
