// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-checkpoint.md, section "safetensors.rs"
// ============================================================================
//! Reading and writing the safetensors file format for `f32` tensors.
//!
//! A file is an 8-byte little-endian length `N`, then `N` bytes of JSON header, then the
//! raw tensor data. The header maps each tensor name to its dtype, shape and
//! `[begin, end)` byte range inside the data, and may carry a `__metadata__` object of
//! string pairs. Only the `F32` dtype is supported; a file with any other dtype is
//! rejected by name instead of being converted.
//!
//! The reader checks everything the format promises before it trusts it: the header
//! length, the header text, every shape against its byte range, and that the ranges
//! tile the data exactly with no gap, overlap or trailing bytes. Allocation is bounded
//! by the length of the file, never by what the header claims.

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use midman_foundation::{Error, Result};

use crate::json::{self, Value};

/// What the errors say was being read or written.
const WHAT: &str = "safetensors file";

/// The header key that holds the string metadata. It cannot be a tensor name.
const METADATA_KEY: &str = "__metadata__";

/// The only dtype this crate reads and writes.
const DTYPE: &str = "F32";

/// Bytes per element.
const ELEMENT_BYTES: usize = 4;

/// The format limits the header to 100 MB.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// How many `f32` values are converted to bytes at a time when writing.
const WRITE_CHUNK: usize = 16 * 1024;

/// One named tensor: a shape and its values in row-major order.
#[derive(Debug, Clone)]
pub struct NamedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl NamedTensor {
    /// The tensor's name, for example `blocks.0.attn.wq`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tensor's shape. An empty shape is a scalar with one value.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The values in row-major order. Their count is the product of the shape.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

/// The contents of a safetensors file: named `f32` tensors and string metadata.
#[derive(Debug, Clone, Default)]
pub struct TensorFile {
    tensors: Vec<NamedTensor>,
    metadata: BTreeMap<String, String>,
}

impl TensorFile {
    /// An empty file: no tensors and no metadata.
    pub fn new() -> TensorFile {
        TensorFile::default()
    }

    /// Adds a tensor, copying its values.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` does not hold exactly the product of `shape` values,
    /// if the name is already taken, or if the name is `__metadata__`, which the
    /// format reserves.
    pub fn push(&mut self, name: impl Into<String>, shape: &[usize], data: &[f32]) -> Result<()> {
        let name = name.into();
        if name == METADATA_KEY {
            return Err(Error::invalid(format!(
                "`{METADATA_KEY}` is reserved and cannot name a tensor"
            )));
        }
        if self.get(&name).is_some() {
            return Err(Error::invalid(format!("two tensors are named `{name}`")));
        }
        let expected = element_count(shape);
        if expected != Some(data.len()) {
            return Err(Error::shape(
                "TensorFile::push",
                format!("tensor `{name}` has shape {shape:?} but {} values", data.len()),
            ));
        }
        self.tensors.push(NamedTensor { name, shape: shape.to_vec(), data: data.to_vec() });
        Ok(())
    }

    /// The tensors, in the order they were added or, for a file that was read, in the
    /// order their data appears in the file.
    pub fn tensors(&self) -> &[NamedTensor] {
        &self.tensors
    }

    /// Finds a tensor by name.
    pub fn get(&self, name: &str) -> Option<&NamedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The string metadata, sorted by key.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Sets a metadata entry, replacing any earlier value for the key.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Serializes the file to memory. See [`TensorFile::write_to`] for the layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_to(&mut out)?;
        Ok(out)
    }

    /// Writes the file to `writer`.
    ///
    /// Tensors are written sorted by name, and the header is padded with spaces to a
    /// multiple of eight bytes, which is what the reference implementation does for a
    /// file of one dtype, so the same tensors give the same bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the header would exceed the format's 100 MB limit, or if
    /// writing fails.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        let mut sorted: Vec<&NamedTensor> = self.tensors.iter().collect();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));

        let mut header = String::from("{");
        let mut first = true;
        if !self.metadata.is_empty() {
            header.push_str("\"__metadata__\":{");
            for (i, (key, value)) in self.metadata.iter().enumerate() {
                if i > 0 {
                    header.push(',');
                }
                json::write_string(&mut header, key);
                header.push(':');
                json::write_string(&mut header, value);
            }
            header.push('}');
            first = false;
        }
        let mut begin: usize = 0;
        for tensor in &sorted {
            let end = tensor
                .data
                .len()
                .checked_mul(ELEMENT_BYTES)
                .and_then(|bytes| begin.checked_add(bytes))
                .ok_or_else(|| Error::invalid("the tensors are too large to address"))?;
            if !first {
                header.push(',');
            }
            first = false;
            json::write_string(&mut header, &tensor.name);
            header.push_str(&format!(":{{\"dtype\":\"{DTYPE}\",\"shape\":["));
            for (i, dim) in tensor.shape.iter().enumerate() {
                if i > 0 {
                    header.push(',');
                }
                header.push_str(&dim.to_string());
            }
            header.push_str(&format!("],\"data_offsets\":[{begin},{end}]}}"));
            begin = end;
        }
        header.push('}');
        let padding = (8 - header.len() % 8) % 8;
        header.push_str(&" ".repeat(padding));
        if header.len() as u64 > MAX_HEADER_BYTES {
            return Err(Error::invalid(format!(
                "the header is {} bytes, over the format's limit of {MAX_HEADER_BYTES}",
                header.len()
            )));
        }

        writer.write_all(&(header.len() as u64).to_le_bytes())?;
        writer.write_all(header.as_bytes())?;
        let mut buffer = Vec::with_capacity(WRITE_CHUNK * ELEMENT_BYTES);
        for tensor in sorted {
            for chunk in tensor.data.chunks(WRITE_CHUNK) {
                buffer.clear();
                for value in chunk {
                    buffer.extend_from_slice(&value.to_le_bytes());
                }
                writer.write_all(&buffer)?;
            }
        }
        Ok(())
    }

    /// Writes the file to `path` without ever leaving a half-written file there.
    ///
    /// The bytes go to a sibling file named `<path>.tmp`, are flushed to disk, and are
    /// then renamed over `path`, so a crash leaves either the old file or the new one.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written; the temporary file is removed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let mut tmp_name = path.as_os_str().to_owned();
        tmp_name.push(".tmp");
        let tmp = PathBuf::from(tmp_name);
        let result = self.write_then_rename(&tmp, path);
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result.map_err(|err| match err {
            Error::Io(message) => Error::Io(format!("{}: {message}", path.display())),
            other => other,
        })
    }

    fn write_then_rename(&self, tmp: &Path, path: &Path) -> Result<()> {
        let mut writer = BufWriter::new(fs::File::create(tmp)?);
        self.write_to(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Reads a file from `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or is not a valid `F32`
    /// safetensors file; see [`TensorFile::from_bytes`].
    pub fn load(path: impl AsRef<Path>) -> Result<TensorFile> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).map_err(|err| Error::Io(format!("{}: {err}", path.display())))?;
        TensorFile::from_bytes(&bytes)
    }

    /// Parses a whole file held in memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the 8-byte length prefix or the header is malformed, if the
    /// header is over 100 MB or does not start with `{`, if a tensor has a dtype
    /// other than `F32` or a field the format does not define, if a shape does not
    /// match its byte range, or if the ranges do not cover the data exactly.
    pub fn from_bytes(bytes: &[u8]) -> Result<TensorFile> {
        if bytes.len() < 8 {
            return Err(fail("the file is shorter than its 8-byte length prefix"));
        }
        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&bytes[..8]);
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES {
            return Err(fail(&format!(
                "the header claims {header_len} bytes, over the format's limit of {MAX_HEADER_BYTES}"
            )));
        }
        let header_len = header_len as usize;
        if header_len > bytes.len() - 8 {
            return Err(fail(&format!(
                "the header claims {header_len} bytes but the file has only {} after the prefix",
                bytes.len() - 8
            )));
        }
        let header = &bytes[8..8 + header_len];
        let data = &bytes[8 + header_len..];
        if header.first() != Some(&b'{') {
            return Err(fail("the header must start with `{`"));
        }
        let text =
            std::str::from_utf8(header).map_err(|_| fail("the header is not valid UTF-8"))?;
        let Value::Object(root) = json::parse(text)? else {
            return Err(fail("the header must be a JSON object"));
        };

        let mut metadata = BTreeMap::new();
        let mut entries: Vec<Entry> = Vec::new();
        for (key, value) in root {
            if key == METADATA_KEY {
                metadata = read_metadata(value)?;
            } else {
                entries.push(read_entry(key, value)?);
            }
        }

        entries.sort_by_key(|entry| (entry.begin, entry.end));
        let mut expected_begin: u64 = 0;
        for entry in &entries {
            if entry.begin != expected_begin {
                return Err(fail(&format!(
                    "tensor `{}` starts at byte {} where byte {expected_begin} was expected: \
                     the data ranges must tile the data with no gap or overlap",
                    entry.name, entry.begin
                )));
            }
            expected_begin = entry.end;
        }
        if expected_begin != data.len() as u64 {
            return Err(fail(&format!(
                "the tensors cover {expected_begin} bytes but the data section has {}",
                data.len()
            )));
        }

        let tensors = entries
            .into_iter()
            .map(|entry| {
                // The tiling check above keeps both ends inside `data`.
                let raw = &data[entry.begin as usize..entry.end as usize];
                let values = raw
                    .chunks_exact(ELEMENT_BYTES)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                NamedTensor { name: entry.name, shape: entry.shape, data: values }
            })
            .collect();
        Ok(TensorFile { tensors, metadata })
    }
}

/// One tensor's header entry, checked on its own.
struct Entry {
    name: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

fn fail(message: &str) -> Error {
    Error::parse(WHAT, 0, message)
}

/// The product of the dimensions, or `None` if it overflows.
fn element_count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn read_metadata(value: Value) -> Result<BTreeMap<String, String>> {
    let Value::Object(pairs) = value else {
        return Err(fail("`__metadata__` must be an object of strings"));
    };
    let mut metadata = BTreeMap::new();
    for (key, value) in pairs {
        let Value::String(text) = value else {
            return Err(fail(&format!("metadata value for `{key}` must be a string")));
        };
        metadata.insert(key, text);
    }
    Ok(metadata)
}

fn read_entry(name: String, value: Value) -> Result<Entry> {
    let Value::Object(fields) = value else {
        return Err(fail(&format!("the entry for `{name}` must be an object")));
    };
    let (mut dtype, mut shape, mut offsets) = (None, None, None);
    for (key, value) in fields {
        match key.as_str() {
            "dtype" => dtype = Some(value),
            "shape" => shape = Some(value),
            "data_offsets" => offsets = Some(value),
            other => {
                return Err(fail(&format!(
                    "the entry for `{name}` has an unknown field `{other}`"
                )));
            }
        }
    }
    let (Some(dtype), Some(shape), Some(offsets)) = (dtype, shape, offsets) else {
        return Err(fail(&format!(
            "the entry for `{name}` must have `dtype`, `shape` and `data_offsets`"
        )));
    };

    match dtype {
        Value::String(d) if d == DTYPE => {}
        Value::String(d) => {
            return Err(fail(&format!(
                "tensor `{name}` has dtype `{d}`; only {DTYPE} is supported"
            )));
        }
        _ => return Err(fail(&format!("the dtype of `{name}` must be a string"))),
    }

    let Value::Array(dims) = shape else {
        return Err(fail(&format!("the shape of `{name}` must be an array")));
    };
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        let Value::Number(n) = dim else {
            return Err(fail(&format!("the shape of `{name}` must hold only integers")));
        };
        let n = usize::try_from(n)
            .map_err(|_| fail(&format!("a dimension of `{name}` is too large")))?;
        shape.push(n);
    }

    let Value::Array(pair) = offsets else {
        return Err(fail(&format!("the data_offsets of `{name}` must be an array")));
    };
    let (begin, end) = match pair.as_slice() {
        [Value::Number(begin), Value::Number(end)] => (*begin, *end),
        _ => {
            return Err(fail(&format!("the data_offsets of `{name}` must be two integers")));
        }
    };

    let bytes = element_count(&shape)
        .and_then(|n| n.checked_mul(ELEMENT_BYTES))
        .ok_or_else(|| fail(&format!("the shape of `{name}` is too large")))?;
    if end.checked_sub(begin) != Some(bytes as u64) {
        return Err(fail(&format!(
            "tensor `{name}` has shape {shape:?}, which needs {bytes} bytes, \
             but its data_offsets [{begin}, {end}] do not span that"
        )));
    }
    Ok(Entry { name, shape, begin, end })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Builds a file from a header written by hand and raw data bytes.
    fn raw_file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    fn floats(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn sample() -> TensorFile {
        let mut file = TensorFile::new();
        file.push("w", &[2, 3], &[1.5, -2.0, 0.0, 3.25, 1e-30, 1e30]).unwrap();
        file.push("b", &[3], &[0.1, 0.2, 0.3]).unwrap();
        file.set_metadata("format", "test");
        file
    }

    #[test]
    fn round_trips_every_bit_including_nan_and_signed_zero() {
        let tricky = [
            f32::NAN,
            f32::from_bits(0x7fc0_1234),
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::MAX,
            f32::MIN,
        ];
        let mut file = TensorFile::new();
        file.push("t", &[tricky.len()], &tricky).unwrap();
        let back = TensorFile::from_bytes(&file.to_bytes().unwrap()).unwrap();
        assert_eq!(bits(back.get("t").unwrap().data()), bits(&tricky));
    }

    #[test]
    fn round_trips_names_shapes_metadata_and_unicode() {
        let mut file = sample();
        file.push("scalar", &[], &[7.0]).unwrap();
        file.push("empty", &[0], &[]).unwrap();
        file.push("blocks.0.attn.wq é日本", &[1, 1, 2], &[1.0, 2.0]).unwrap();
        file.set_metadata("note", "quote \" backslash \\ newline \n é");
        let back = TensorFile::from_bytes(&file.to_bytes().unwrap()).unwrap();
        assert_eq!(back.metadata(), file.metadata());
        assert_eq!(back.tensors().len(), file.tensors().len());
        for t in file.tensors() {
            let got = back.get(t.name()).unwrap();
            assert_eq!(got.shape(), t.shape());
            assert_eq!(bits(got.data()), bits(t.data()));
        }
    }

    #[test]
    fn writes_sorted_by_name_with_a_header_padded_to_eight_bytes() {
        let bytes = sample().to_bytes().unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(header_len % 8, 0);
        let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
        assert_eq!(
            header.trim_end(),
            "{\"__metadata__\":{\"format\":\"test\"},\
             \"b\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]},\
             \"w\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[12,36]}}"
        );
        assert_eq!(bytes.len(), 8 + header_len + 36);
    }

    #[test]
    fn the_header_is_padded_by_less_than_eight_bytes_whatever_its_length() {
        // Name lengths 1..=24 walk the unpadded header through every length modulo 8,
        // including the aligned case, which must get no padding at all.
        for len in 1..=24 {
            let mut file = TensorFile::new();
            file.push("a".repeat(len), &[1], &[1.0]).unwrap();
            let bytes = file.to_bytes().unwrap();
            let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let header = std::str::from_utf8(&bytes[8..8 + header_len]).unwrap();
            assert_eq!(header_len % 8, 0, "name length {len}");
            let padding = header.len() - header.trim_end().len();
            assert!(padding < 8, "name length {len} was padded by {padding} bytes");
        }
    }

    #[test]
    fn an_empty_file_is_the_empty_object_padded() {
        let bytes = TensorFile::new().to_bytes().unwrap();
        assert_eq!(bytes, raw_file("{}      ", &[]));
        let back = TensorFile::from_bytes(&bytes).unwrap();
        assert!(back.tensors().is_empty() && back.metadata().is_empty());
    }

    #[test]
    fn reading_orders_tensors_by_their_position_in_the_data() {
        // `z` is first in the data though last by name.
        let header = r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"z":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let file = TensorFile::from_bytes(&raw_file(header, &floats(&[1.0, 2.0]))).unwrap();
        let names: Vec<&str> = file.tensors().iter().map(|t| t.name()).collect();
        assert_eq!(names, ["z", "a"]);
        assert_eq!(file.get("a").unwrap().data(), [2.0]);
    }

    #[test]
    fn reading_accepts_any_whitespace_padding_after_the_header() {
        let header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}} \n\t  ";
        let file = TensorFile::from_bytes(&raw_file(header, &floats(&[9.0]))).unwrap();
        assert_eq!(file.get("a").unwrap().data(), [9.0]);
    }

    #[test]
    fn push_rejects_bad_tensors() {
        let mut file = TensorFile::new();
        assert!(file.push("a", &[2, 2], &[1.0, 2.0, 3.0]).is_err());
        assert!(file.push("a", &[usize::MAX, 2], &[]).is_err());
        assert!(file.push("__metadata__", &[1], &[1.0]).is_err());
        file.push("a", &[1], &[1.0]).unwrap();
        assert!(file.push("a", &[1], &[2.0]).is_err());
        assert_eq!(file.tensors().len(), 1);
    }

    fn one_tensor_header(dtype: &str, shape: &str, begin: u64, end: u64) -> String {
        format!(r#"{{"a":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[{begin},{end}]}}}}"#)
    }

    #[test]
    fn rejects_malformed_files() {
        let four = floats(&[1.0]);
        let good = one_tensor_header("F32", "[1]", 0, 4);
        assert!(TensorFile::from_bytes(&raw_file(&good, &four)).is_ok());

        let mut too_short_prefix = vec![0u8; 7];
        too_short_prefix[0] = 2;
        let mut not_utf8 = 3u64.to_le_bytes().to_vec();
        not_utf8.extend_from_slice(&[b'{', 0xff, b'}']);
        let mut huge_header = u64::MAX.to_le_bytes().to_vec();
        huge_header.extend_from_slice(b"{}");
        let mut header_past_end = 50u64.to_le_bytes().to_vec();
        header_past_end.extend_from_slice(b"{}");
        let mut limit = (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec();
        limit.extend_from_slice(b"{}");

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("short prefix", too_short_prefix),
            ("header length of u64::MAX", huge_header),
            ("header past the end of the file", header_past_end),
            ("header over the format limit", limit),
            ("header not starting with {", raw_file(" {}", &[])),
            ("header not UTF-8", not_utf8),
            ("header not an object", raw_file("[]", &[])),
            ("unsupported dtype", raw_file(&one_tensor_header("F16", "[1]", 0, 2), &[0, 0])),
            ("lowercase dtype", raw_file(&one_tensor_header("f32", "[1]", 0, 4), &four)),
            ("shape against range", raw_file(&one_tensor_header("F32", "[2]", 0, 4), &four)),
            ("end before begin", raw_file(&one_tensor_header("F32", "[1]", 4, 0), &four)),
            ("range past the data", raw_file(&one_tensor_header("F32", "[1]", 0, 4), &[0, 0])),
            ("trailing bytes", raw_file(&good, &floats(&[1.0, 2.0]))),
            (
                "gap before the first tensor",
                raw_file(&one_tensor_header("F32", "[1]", 4, 8), &floats(&[1.0, 2.0])),
            ),
            (
                "shape overflow",
                raw_file(
                    &one_tensor_header("F32", "[4294967296,4294967296,4294967296]", 0, 4),
                    &four,
                ),
            ),
            (
                "dimension too large for a u64",
                raw_file(&one_tensor_header("F32", "[18446744073709551616]", 0, 4), &four),
            ),
            ("negative dimension", raw_file(&one_tensor_header("F32", "[-1]", 0, 4), &four)),
            (
                "unknown field",
                raw_file(
                    r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":1}}"#,
                    &four,
                ),
            ),
            ("missing field", raw_file(r#"{"a":{"dtype":"F32","shape":[1]}}"#, &four)),
            ("entry not an object", raw_file(r#"{"a":1}"#, &four)),
            (
                "three offsets",
                raw_file(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4,8]}}"#, &four),
            ),
            ("metadata not strings", raw_file(r#"{"__metadata__":{"k":1}}"#, &[])),
            ("metadata not an object", raw_file(r#"{"__metadata__":["k"]}"#, &[])),
            (
                "duplicate tensor names",
                raw_file(
                    r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
                    &floats(&[1.0, 2.0]),
                ),
            ),
        ];
        for (what, bytes) in cases {
            assert!(TensorFile::from_bytes(&bytes).is_err(), "should reject: {what}");
        }
    }

    #[test]
    fn a_header_over_the_format_limit_is_refused_for_that_reason() {
        let mut bytes = (MAX_HEADER_BYTES + 1).to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let message = TensorFile::from_bytes(&bytes).unwrap_err().to_string();
        assert!(message.contains("limit"), "{message}");
        // Right at the limit it is the missing bytes, not the limit, that stop it.
        let mut at_limit = MAX_HEADER_BYTES.to_le_bytes().to_vec();
        at_limit.extend_from_slice(b"{}");
        let message = TensorFile::from_bytes(&at_limit).unwrap_err().to_string();
        assert!(!message.contains("limit"), "{message}");
    }

    #[test]
    fn rejects_overlapping_ranges_even_when_the_total_fits() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let err = TensorFile::from_bytes(&raw_file(header, &floats(&[1.0, 2.0, 3.0]))).unwrap_err();
        assert!(err.to_string().contains("no gap or overlap"), "{err}");
    }

    #[test]
    fn a_header_that_lies_about_size_cannot_force_a_large_allocation() {
        // Claims a billion elements but supplies four bytes: rejected before any
        // allocation of that size.
        let header = one_tensor_header("F32", "[1000000000]", 0, 4_000_000_000);
        let err = TensorFile::from_bytes(&raw_file(&header, &floats(&[1.0]))).unwrap_err();
        assert!(err.to_string().contains("cover"), "{err}");
    }

    #[test]
    fn errors_name_the_file_format_and_the_tensor() {
        let message =
            TensorFile::from_bytes(&raw_file(&one_tensor_header("BF16", "[1]", 0, 2), &[0, 0]))
                .unwrap_err()
                .to_string();
        assert!(message.contains("safetensors file"), "{message}");
        assert!(message.contains("`a`") && message.contains("BF16"), "{message}");
    }

    struct FailingWriter {
        budget: usize,
    }

    impl Write for FailingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if buf.len() > self.budget {
                return Err(std::io::Error::other("disk full"));
            }
            self.budget -= buf.len();
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_failing_writer_gives_an_io_error() {
        for budget in [0, 8, 40] {
            match sample().write_to(&mut FailingWriter { budget }) {
                Err(Error::Io(message)) => assert!(message.contains("disk full")),
                other => panic!("expected an I/O error, got {other:?}"),
            }
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("midman-checkpoint-{}-{n}-{name}", std::process::id()))
    }

    #[test]
    fn save_and_load_round_trip_and_leave_no_temporary_file() {
        let path = temp_path("roundtrip.safetensors");
        sample().save(&path).unwrap();
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        assert!(!Path::new(&tmp).exists());
        let back = TensorFile::load(&path).unwrap();
        assert_eq!(bits(back.get("w").unwrap().data()), bits(sample().get("w").unwrap().data()));
        // Saving again over the same path replaces it.
        let mut other = TensorFile::new();
        other.push("only", &[1], &[5.0]).unwrap();
        other.save(&path).unwrap();
        assert_eq!(TensorFile::load(&path).unwrap().tensors().len(), 1);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn saving_into_a_missing_directory_fails_with_the_path_and_leaves_nothing() {
        let path = temp_path("no-such-dir").join("inner").join("m.safetensors");
        let message = sample().save(&path).unwrap_err().to_string();
        assert!(message.contains("m.safetensors"), "{message}");
        assert!(!path.exists());
    }

    #[test]
    fn loading_a_missing_file_names_the_path() {
        let path = temp_path("absent.safetensors");
        let message = TensorFile::load(path).unwrap_err().to_string();
        assert!(message.contains("absent.safetensors"), "{message}");
    }
}
