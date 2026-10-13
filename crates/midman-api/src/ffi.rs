// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-api.md, section "ffi.rs"
// ============================================================================
//! The C ABI.
//!
//! Everything here is `extern "C"` with fixed-width scalar arguments, so the same
//! functions serve a browser (WebAssembly exports) and a native C or C++ caller.
//! The rules the boundary follows:
//!
//! - **Handles, not pointers.** A [`Lab`] lives in a table inside this library and
//!   the caller holds a small integer. A stale or invented handle is an error
//!   status, not undefined behaviour.
//! - **The caller owns its buffers, the library owns its results.** Input arrays
//!   are written by the caller into memory from [`midman_alloc`] and freed with
//!   [`midman_free`]. Results are left in library-owned buffers read through
//!   `midman_text_*`, `midman_floats_*` and `midman_error_*`; each stays valid
//!   until the next call to any function in this module.
//! - **Status codes, never unwinding.** Functions return `0` for success, `1` for
//!   "completed partially" (see [`midman_lab_step`]) and a negative code for
//!   failure, with the message in the error buffer. A panic is caught and
//!   reported as [`ERR_PANIC`] instead of crossing the boundary. (A build with
//!   `panic = "abort"`, as WebAssembly has, traps instead.)
//! - **One thread.** State is thread-local, so a handle is only valid on the thread
//!   that created it.
//!
//! Arrays of numbers are passed as `f64`: a model config is
//! `[vocab_size, d_model, n_layers, n_heads, n_kv_heads, d_ff, max_seq_len,
//! rope_theta, norm_eps, tie_embeddings]` and hyperparameters are
//! `[max_lr, min_lr, warmup_steps, total_steps, weight_decay, clip_norm, beta1,
//! beta2]`. Counts must be whole numbers; `tie_embeddings` is 0 or non-zero. The
//! same orders are listed by [`midman_describe`].

use crate::json::JsonWriter;
use crate::{config_report, rope_table, Hyper, Lab, LabConfig, MAX_PARAMS, MAX_TOKENS};
use midman_foundation::{Error, Result};
use midman_model_config::{ModelConfig, Preset};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The version of this ABI. It changes when a function is removed or its meaning
/// changes, not when one is added.
pub const ABI_VERSION: u32 = 1;

/// The call succeeded.
pub const STATUS_OK: i32 = 0;
/// [`midman_lab_step`] stopped before every requested step; see its docs.
pub const STATUS_STOPPED: i32 = 1;
/// A bad argument: a null pointer, an invalid handle, a number out of range.
pub const ERR_ARGUMENT: i32 = -1;
/// A model configuration problem.
pub const ERR_CONFIG: i32 = -2;
/// Tensor shapes did not fit.
pub const ERR_SHAPE: i32 = -3;
/// An index (such as a token id) was out of range.
pub const ERR_INDEX: i32 = -4;
/// Any other library error.
pub const ERR_OTHER: i32 = -5;
/// A panic was caught inside the library.
pub const ERR_PANIC: i32 = -99;

const CONFIG_LEN: usize = 10;
const HYPER_LEN: usize = 8;
/// Alignment of buffers from [`midman_alloc`], enough for `f64`.
const ALLOC_ALIGN: usize = 8;

const CONFIG_FIELDS: [&str; CONFIG_LEN] = [
    "vocab_size",
    "d_model",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "max_seq_len",
    "rope_theta",
    "norm_eps",
    "tie_embeddings",
];
const HYPER_FIELDS: [&str; HYPER_LEN] = [
    "max_lr",
    "min_lr",
    "warmup_steps",
    "total_steps",
    "weight_decay",
    "clip_norm",
    "beta1",
    "beta2",
];

#[derive(Default)]
struct State {
    labs: Vec<Option<Lab>>,
    text: Vec<u8>,
    floats: Vec<f32>,
    error: Vec<u8>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::default());
}

fn status_of(error: &Error) -> i32 {
    match error {
        Error::InvalidArgument(_) => ERR_ARGUMENT,
        Error::Config { .. } => ERR_CONFIG,
        Error::Shape { .. } => ERR_SHAPE,
        Error::IndexOutOfRange { .. } => ERR_INDEX,
        _ => ERR_OTHER,
    }
}

fn set_error(message: &str) {
    STATE.with(|cell| cell.borrow_mut().error = message.as_bytes().to_vec());
}

/// Runs `body` against the state with the result buffers cleared, turning an
/// error or a panic into a status code and an error message.
fn run(body: impl FnOnce(&mut State) -> Result<i32>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        STATE.with(|cell| {
            let mut state = cell.borrow_mut();
            state.text.clear();
            state.floats.clear();
            state.error.clear();
            body(&mut state)
        })
    }));
    match outcome {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            set_error(&error.to_string());
            status_of(&error)
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with no message".to_string());
            set_error(&format!("internal error: {message}"));
            ERR_PANIC
        }
    }
}

fn argument(message: impl Into<String>) -> Error {
    Error::invalid(message)
}

/// Reads `len` values the caller promises are readable at `ptr`.
///
/// # Safety
///
/// `ptr` must be null or point to `len` initialized `f64` values that stay valid
/// and unmodified for the returned lifetime.
unsafe fn read_f64s<'a>(ptr: *const f64, len: usize, what: &str) -> Result<&'a [f64]> {
    if ptr.is_null() {
        return Err(argument(format!("{what} is a null pointer")));
    }
    if (ptr as usize) & (std::mem::align_of::<f64>() - 1) != 0 {
        return Err(argument(format!("{what} is not aligned for f64")));
    }
    // SAFETY: the pointer is non-null and aligned, and the caller guarantees `len`
    // readable values (see this function's contract).
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// A whole number from 0 to `u32::MAX`, as a `usize`.
fn as_count(value: f64, name: &str) -> Result<usize> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= f64::from(u32::MAX) {
        Ok(value as usize)
    } else {
        Err(argument(format!("{name} must be a whole number from 0 to 4294967295, got {value}")))
    }
}

fn as_finite_f32(value: f64, name: &str) -> Result<f32> {
    if value.is_finite() && value.abs() <= f64::from(f32::MAX) {
        Ok(value as f32)
    } else {
        Err(argument(format!("{name} must be a finite number that fits in f32, got {value}")))
    }
}

fn config_from(values: &[f64]) -> Result<ModelConfig> {
    let count = |i: usize| as_count(values[i], CONFIG_FIELDS[i]);
    Ok(ModelConfig {
        vocab_size: count(0)?,
        d_model: count(1)?,
        n_layers: count(2)?,
        n_heads: count(3)?,
        n_kv_heads: count(4)?,
        d_ff: count(5)?,
        max_seq_len: count(6)?,
        rope_theta: as_finite_f32(values[7], CONFIG_FIELDS[7])?,
        norm_eps: as_finite_f32(values[8], CONFIG_FIELDS[8])?,
        tie_embeddings: values[9] != 0.0,
    })
}

fn config_values(config: &ModelConfig) -> [f64; CONFIG_LEN] {
    [
        config.vocab_size as f64,
        config.d_model as f64,
        config.n_layers as f64,
        config.n_heads as f64,
        config.n_kv_heads as f64,
        config.d_ff as f64,
        config.max_seq_len as f64,
        f64::from(config.rope_theta),
        f64::from(config.norm_eps),
        if config.tie_embeddings { 1.0 } else { 0.0 },
    ]
}

fn hyper_from(values: &[f64]) -> Result<Hyper> {
    Ok(Hyper {
        max_lr: as_finite_f32(values[0], HYPER_FIELDS[0])?,
        min_lr: as_finite_f32(values[1], HYPER_FIELDS[1])?,
        warmup_steps: as_count(values[2], HYPER_FIELDS[2])? as u64,
        total_steps: as_count(values[3], HYPER_FIELDS[3])? as u64,
        weight_decay: as_finite_f32(values[4], HYPER_FIELDS[4])?,
        clip_norm: as_finite_f32(values[5], HYPER_FIELDS[5])?,
        beta1: as_finite_f32(values[6], HYPER_FIELDS[6])?,
        beta2: as_finite_f32(values[7], HYPER_FIELDS[7])?,
    })
}

fn put_text(state: &mut State, json: String) {
    state.text = json.into_bytes();
}

fn lab_ref(state: &State, handle: u32) -> Result<&Lab> {
    (handle as usize)
        .checked_sub(1)
        .and_then(|i| state.labs.get(i))
        .and_then(Option::as_ref)
        .ok_or_else(|| argument(format!("{handle} is not a live lab handle")))
}

fn lab_mut(state: &mut State, handle: u32) -> Result<&mut Lab> {
    (handle as usize)
        .checked_sub(1)
        .and_then(|i| state.labs.get_mut(i))
        .and_then(Option::as_mut)
        .ok_or_else(|| argument(format!("{handle} is not a live lab handle")))
}

// ---------------------------------------------------------------- memory

/// The version of this ABI, [`ABI_VERSION`].
#[no_mangle]
pub extern "C" fn midman_abi_version() -> u32 {
    ABI_VERSION
}

/// Allocates `len` bytes (at least 1), aligned to 8, for the caller to fill with
/// input data. Returns null if the allocation fails. Free it with
/// [`midman_free`] using the same `len`.
#[no_mangle]
pub extern "C" fn midman_alloc(len: u32) -> *mut u8 {
    match Layout::from_size_align((len as usize).max(1), ALLOC_ALIGN) {
        // SAFETY: the layout has a non-zero size.
        Ok(layout) => unsafe { alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a buffer from [`midman_alloc`]. A null `ptr` is ignored.
///
/// # Safety
///
/// `ptr` must be null or a pointer returned by [`midman_alloc`] with exactly this
/// `len`, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn midman_free(ptr: *mut u8, len: u32) {
    if ptr.is_null() {
        return;
    }
    if let Ok(layout) = Layout::from_size_align((len as usize).max(1), ALLOC_ALIGN) {
        // SAFETY: by this function's contract the pointer came from `midman_alloc`
        // with this layout and has not been freed.
        unsafe { dealloc(ptr, layout) }
    }
}

// --------------------------------------------------------------- results

/// Pointer to the text result (UTF-8 JSON) of the last call, valid until the next call.
#[no_mangle]
pub extern "C" fn midman_text_ptr() -> *const u8 {
    STATE.with(|cell| cell.borrow().text.as_ptr())
}

/// Length in bytes of the text result.
#[no_mangle]
pub extern "C" fn midman_text_len() -> u32 {
    STATE.with(|cell| cell.borrow().text.len() as u32)
}

/// Pointer to the `f32` result of the last call, valid until the next call. It is
/// 4-byte aligned.
#[no_mangle]
pub extern "C" fn midman_floats_ptr() -> *const f32 {
    STATE.with(|cell| cell.borrow().floats.as_ptr())
}

/// Number of `f32` values in the numeric result.
#[no_mangle]
pub extern "C" fn midman_floats_len() -> u32 {
    STATE.with(|cell| cell.borrow().floats.len() as u32)
}

/// Pointer to the error message (UTF-8) of the last call, empty if it succeeded.
#[no_mangle]
pub extern "C" fn midman_error_ptr() -> *const u8 {
    STATE.with(|cell| cell.borrow().error.as_ptr())
}

/// Length in bytes of the error message.
#[no_mangle]
pub extern "C" fn midman_error_len() -> u32 {
    STATE.with(|cell| cell.borrow().error.len() as u32)
}

// ----------------------------------------------------------- stateless calls

/// Describes the ABI as JSON in the text result: the ABI version, the field order of
/// the config and hyperparameter arrays, the lab's size limits, and every preset
/// with its config values and exact parameter count.
#[no_mangle]
pub extern "C" fn midman_describe() -> i32 {
    run(|state| {
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("abi").integer(u64::from(ABI_VERSION));
        w.key("version").string(env!("CARGO_PKG_VERSION"));
        w.key("config_fields").begin_array();
        for name in CONFIG_FIELDS {
            w.string(name);
        }
        w.end_array();
        w.key("hyper_fields").begin_array();
        for name in HYPER_FIELDS {
            w.string(name);
        }
        w.end_array();
        w.key("limits").begin_object();
        w.key("max_params").integer(MAX_PARAMS);
        w.key("max_tokens").integer(MAX_TOKENS as u64);
        w.end_object();
        w.key("presets").begin_array();
        for preset in Preset::ALL {
            let config = preset.default_config();
            w.begin_object();
            w.key("name").string(preset.name());
            w.key("config").begin_array();
            for v in config_values(&config) {
                w.number(v);
            }
            w.end_array();
            w.key("params").integer(config.param_count()?);
            w.end_object();
        }
        w.end_array();
        w.end_object();
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// Writes the 10 config values of preset `index` (in [`Preset::ALL`] order) to `out`.
///
/// # Safety
///
/// `out` must point to at least 10 writable, 8-byte-aligned `f64` values.
#[no_mangle]
pub unsafe extern "C" fn midman_preset_config(index: u32, out: *mut f64) -> i32 {
    run(|_| {
        let preset = Preset::ALL
            .get(index as usize)
            .ok_or_else(|| argument(format!("there is no preset {index}")))?;
        if out.is_null() || (out as usize) & (std::mem::align_of::<f64>() - 1) != 0 {
            return Err(argument("out is null or not aligned for f64"));
        }
        let values = config_values(&preset.default_config());
        // SAFETY: `out` is non-null and aligned, and the caller guarantees room for
        // CONFIG_LEN values.
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), out, CONFIG_LEN) };
        Ok(STATUS_OK)
    })
}

/// Validates a model config and counts its parameters, without building anything.
///
/// Writes JSON to the text result. A valid config gives `{"valid":true,"head_dim":..,
/// "kv_dim":..,"params":{"embedding":..,"attention":..,"mlp":..,"norms":..,
/// "lm_head":..,"total":..}}`; an invalid one gives `{"valid":false,"error":".."}`.
/// Both return [`STATUS_OK`]: an invalid config is an answer, not a failure of the
/// call. A malformed argument (null pointer, a non-whole count) is an error status.
///
/// # Safety
///
/// `config` must point to 10 readable, 8-byte-aligned `f64` values.
#[no_mangle]
pub unsafe extern "C" fn midman_config_report(config: *const f64) -> i32 {
    run(|state| {
        // SAFETY: the caller guarantees CONFIG_LEN readable values.
        let values = unsafe { read_f64s(config, CONFIG_LEN, "config")? };
        let parsed = config_from(values)?;
        let mut w = JsonWriter::new();
        w.begin_object();
        match config_report(&parsed) {
            Ok(report) => {
                w.key("valid").boolean(true);
                w.key("head_dim").integer(report.head_dim as u64);
                w.key("kv_dim").integer(report.kv_dim as u64);
                w.key("params").begin_object();
                w.key("embedding").integer(report.params.embedding);
                w.key("attention").integer(report.params.attention);
                w.key("mlp").integer(report.params.mlp);
                w.key("norms").integer(report.params.norms);
                w.key("lm_head").integer(report.params.lm_head);
                w.key("total").integer(report.params.total());
                w.end_object();
            }
            Err(error) => {
                w.key("valid").boolean(false);
                w.key("error").string(&error.to_string());
            }
        }
        w.end_object();
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// The rotary rotation tables for `head_dim`, `max_seq_len` and `theta`. The
/// numeric result is the cosines then the sines, each `max_seq_len * head_dim / 2`
/// values with one row per position; the text result is
/// `{"positions":..,"half":..}`.
#[no_mangle]
pub extern "C" fn midman_rope_table(head_dim: u32, max_seq_len: u32, theta: f64) -> i32 {
    run(|state| {
        let theta = as_finite_f32(theta, "theta")?;
        let (positions, half, cos, sin) =
            rope_table(head_dim as usize, max_seq_len as usize, theta)?;
        state.floats = cos;
        state.floats.extend_from_slice(&sin);
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("positions").integer(positions as u64);
        w.key("half").integer(half as u64);
        w.end_object();
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

// ------------------------------------------------------------------- labs

/// Creates a lab and returns its handle, or 0 on failure (see the error buffer).
///
/// `config` is 10 `f64` values and `hyper` is 8 (orders in the module docs). The
/// seed is the 64-bit value `seed_hi << 32 | seed_lo`. The task is `batch`
/// random sequences of `seq` tokens; see [`Lab`] for the limits.
///
/// # Safety
///
/// `config` must point to 10 and `hyper` to 8 readable, 8-byte-aligned `f64` values.
#[no_mangle]
pub unsafe extern "C" fn midman_lab_create(
    config: *const f64,
    hyper: *const f64,
    seed_lo: u32,
    seed_hi: u32,
    batch: u32,
    seq: u32,
) -> u32 {
    let status = run(|state| {
        // SAFETY: the caller guarantees the two arrays (see this function's contract).
        let (config_values, hyper_values) = unsafe {
            (read_f64s(config, CONFIG_LEN, "config")?, read_f64s(hyper, HYPER_LEN, "hyper")?)
        };
        let lab = Lab::new(LabConfig {
            model: config_from(config_values)?,
            seed: (u64::from(seed_hi) << 32) | u64::from(seed_lo),
            batch: batch as usize,
            seq: seq as usize,
            hyper: hyper_from(hyper_values)?,
        })?;
        let slot = match state.labs.iter().position(Option::is_none) {
            Some(free) => free,
            None => {
                state.labs.push(None);
                state.labs.len() - 1
            }
        };
        state.labs[slot] = Some(lab);
        Ok(slot as i32 + 1)
    });
    if status > 0 {
        status as u32
    } else {
        0
    }
}

/// Runs `steps` training steps. The numeric result holds three values per completed
/// step: the loss before the update, the global gradient norm before clipping, and
/// the learning rate.
///
/// Returns [`STATUS_OK`] if every step ran. If training had to stop early because a
/// loss or gradient was not a finite number, it returns [`STATUS_STOPPED`], the
/// numeric result holds the steps that did complete, and the error buffer says why.
#[no_mangle]
pub extern "C" fn midman_lab_step(handle: u32, steps: u32) -> i32 {
    run(|state| {
        let report = lab_mut(state, handle)?.step(steps as usize);
        for r in &report.records {
            state.floats.extend_from_slice(&[r.loss, r.grad_norm, r.lr]);
        }
        match report.stopped {
            Some(reason) => {
                state.error = reason.into_bytes();
                Ok(STATUS_STOPPED)
            }
            None => Ok(STATUS_OK),
        }
    })
}

/// Measures the lab's loss and predictions without training. The text result is
/// `{"step":..,"loss":..,"accuracy":..,"batch":..,"seq":..,"inputs":[[..]],
/// "targets":[[..]],"predictions":[[..]]}`, one inner array per sequence.
#[no_mangle]
pub extern "C" fn midman_lab_eval(handle: u32) -> i32 {
    run(|state| {
        let lab = lab_ref(state, handle)?;
        let eval = lab.eval()?;
        let (batch, seq) = (lab.config().batch, lab.config().seq);
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("step").integer(lab.step_count());
        w.key("loss").number(f64::from(eval.loss));
        w.key("accuracy").number(f64::from(eval.accuracy));
        w.key("batch").integer(batch as u64);
        w.key("seq").integer(seq as u64);
        for (name, ids) in [
            ("inputs", lab.inputs()),
            ("targets", lab.targets()),
            ("predictions", &eval.predictions),
        ] {
            w.key(name).begin_array();
            for sequence in ids.chunks_exact(seq) {
                w.begin_array();
                for &id in sequence {
                    w.integer(id as u64);
                }
                w.end_array();
            }
            w.end_array();
        }
        w.end_object();
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// Summarizes every parameter. The text result is `{"step":..,"last_loss":..|null,
/// "num_parameters":..,"params":[{"name":..,"shape":[..],"rms":..,
/// "grad_norm":..|null}]}` in the model's fixed parameter order.
#[no_mangle]
pub extern "C" fn midman_lab_stats(handle: u32) -> i32 {
    run(|state| {
        let lab = lab_ref(state, handle)?;
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("step").integer(lab.step_count());
        w.key("last_loss").optional_number(lab.last_loss().map(f64::from));
        w.key("num_parameters").integer(lab.num_parameters() as u64);
        w.key("params").begin_array();
        for p in lab.param_stats() {
            w.begin_object();
            w.key("name").string(&p.name);
            w.key("shape").begin_array();
            for &d in &p.shape {
                w.integer(d as u64);
            }
            w.end_array();
            w.key("rms").number(f64::from(p.rms));
            w.key("grad_norm").optional_number(p.grad_norm.map(f64::from));
            w.end_object();
        }
        w.end_array();
        w.end_object();
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// The attention probabilities of one task sequence. The numeric result is
/// `layers * heads * seq * seq` values, layer-major then head, row-major; the text
/// result is `{"layers":..,"heads":..,"seq":..}`.
#[no_mangle]
pub extern "C" fn midman_lab_attention(handle: u32, sequence: u32) -> i32 {
    run(|state| {
        let maps = lab_ref(state, handle)?.attention(sequence as usize)?;
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("layers").integer(maps.layers as u64);
        w.key("heads").integer(maps.heads as u64);
        w.key("seq").integer(maps.seq as u64);
        w.end_object();
        state.floats = maps.data;
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// The token embedding table. The numeric result is `rows * cols` values, one row
/// per token; the text result is `{"rows":..,"cols":..}`.
#[no_mangle]
pub extern "C" fn midman_lab_embedding(handle: u32) -> i32 {
    run(|state| {
        let (rows, cols, values) = lab_ref(state, handle)?.embedding();
        let mut w = JsonWriter::new();
        w.begin_object();
        w.key("rows").integer(rows as u64);
        w.key("cols").integer(cols as u64);
        w.end_object();
        state.floats = values;
        put_text(state, w.finish());
        Ok(STATUS_OK)
    })
}

/// Destroys a lab and frees its memory. The handle becomes invalid.
#[no_mangle]
pub extern "C" fn midman_lab_destroy(handle: u32) -> i32 {
    run(|state| {
        lab_ref(state, handle)?;
        state.labs[handle as usize - 1] = None;
        Ok(STATUS_OK)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_text() -> String {
        STATE.with(|cell| String::from_utf8(cell.borrow().error.clone()).unwrap())
    }

    #[test]
    fn a_panic_becomes_a_status_and_a_message_instead_of_unwinding_out() {
        let status = run(|_| -> Result<i32> { panic!("boom") });
        assert_eq!(status, ERR_PANIC);
        assert!(error_text().contains("boom"), "{}", error_text());
        // The library still works afterwards.
        assert_eq!(midman_describe(), STATUS_OK);
        assert!(error_text().is_empty());
    }

    #[test]
    fn a_panic_with_a_formatted_message_is_reported_too() {
        let status = run(|_| -> Result<i32> { panic!("value {}", 42) });
        assert_eq!(status, ERR_PANIC);
        assert!(error_text().contains("value 42"));
    }

    #[test]
    fn library_errors_map_to_their_own_codes() {
        assert_eq!(status_of(&Error::invalid("x")), ERR_ARGUMENT);
        assert_eq!(status_of(&Error::config("f", "r")), ERR_CONFIG);
        assert_eq!(status_of(&Error::shape("op", "d")), ERR_SHAPE);
        assert_eq!(status_of(&Error::index("w", 3, 2)), ERR_INDEX);
        assert_eq!(status_of(&Error::parse("w", 1, "m")), ERR_OTHER);
    }

    #[test]
    fn counts_must_be_whole_numbers_in_range() {
        assert_eq!(as_count(12.0, "x").unwrap(), 12);
        assert_eq!(as_count(0.0, "x").unwrap(), 0);
        assert_eq!(as_count(f64::from(u32::MAX), "x").unwrap(), u32::MAX as usize);
        for bad in [-1.0, 1.5, f64::NAN, f64::INFINITY, 4_294_967_296.0] {
            assert!(as_count(bad, "x").is_err(), "{bad}");
        }
    }

    #[test]
    fn floats_must_be_finite_and_fit() {
        assert!(as_finite_f32(1e-5, "x").is_ok());
        assert!(as_finite_f32(f64::NAN, "x").is_err());
        assert!(as_finite_f32(1e300, "x").is_err());
    }
}
