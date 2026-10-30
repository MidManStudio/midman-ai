// ============================================================================
// NOTICE: Full documentation, design decisions, and fix history for this file
// live in docs/midman-training.md, section "state.rs"
// ============================================================================
//! The state of a run in one file: weights, optimizer moments and where the run is.

use crate::TrainConfig;
use midman_checkpoint::{NamedTensor, TensorFile};
use midman_foundation::{Error, Result};
use midman_optim::AdamW;
use midman_tensor::Parameter;
use std::collections::{BTreeMap, HashMap, HashSet};

/// The value of the `format` entry of a state file.
pub const STATE_FORMAT: &str = "midman-training-state";

/// The version of the state file format this build writes and reads.
pub const STATE_VERSION: u32 = 1;

const FORMAT_KEY: &str = "format";
const VERSION_KEY: &str = "state_version";
const STEP_KEY: &str = "step";
const EPOCH_KEY: &str = "epoch";
const POSITION_KEY: &str = "position";

const PARAM_PREFIX: &str = "param/";
const M_PREFIX: &str = "adamw.m/";
const V_PREFIX: &str = "adamw.v/";

/// How many names an error message lists before it says "and N more".
const NAMES_SHOWN: usize = 5;

/// Where a run is: steps taken, and the place in the data the next step starts at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Optimizer steps taken.
    pub step: u64,
    /// The pass over the data the batcher is in.
    pub epoch: u64,
    /// Batches of that pass already taken.
    pub position: usize,
}

/// Builds the state file of a run.
///
/// The file holds every parameter as `param/<name>`, its two moments as `adamw.m/<name>`
/// and `adamw.v/<name>` (none before the first step), the settings, the progress, and
/// `extra` (which must not use a key the format uses).
///
/// # Errors
///
/// Returns an error if two parameters share a name, if the optimizer holds moments for a
/// parameter that is not in `params` or of another size, if its step count is not
/// `progress.step`, or if `extra` uses a reserved key.
pub fn state_to_file(
    params: &[&Parameter],
    optimizer: &AdamW,
    progress: Progress,
    config: &TrainConfig,
    extra: &BTreeMap<String, String>,
) -> Result<TensorFile> {
    if optimizer.step_count() != progress.step {
        return Err(Error::invalid(format!(
            "the optimizer has taken {} steps but the run says {}",
            optimizer.step_count(),
            progress.step
        )));
    }
    let mut names = HashSet::new();
    for param in params {
        if !names.insert(param.name()) {
            return Err(Error::invalid(format!("two parameters are named `{}`", param.name())));
        }
    }
    let reserved = [FORMAT_KEY, VERSION_KEY, STEP_KEY, EPOCH_KEY, POSITION_KEY];
    let from_config = config.to_metadata();
    for key in extra.keys() {
        if reserved.contains(&key.as_str()) || from_config.contains_key(key) {
            return Err(Error::invalid(format!("metadata key `{key}` is set by the state writer")));
        }
    }

    let mut file = TensorFile::new();
    for (key, value) in extra.iter().chain(from_config.iter()) {
        file.set_metadata(key.as_str(), value.as_str());
    }
    file.set_metadata(FORMAT_KEY, STATE_FORMAT);
    file.set_metadata(VERSION_KEY, STATE_VERSION.to_string());
    file.set_metadata(STEP_KEY, progress.step.to_string());
    file.set_metadata(EPOCH_KEY, progress.epoch.to_string());
    file.set_metadata(POSITION_KEY, progress.position.to_string());

    for param in params {
        file.push(format!("{PARAM_PREFIX}{}", param.name()), param.shape(), param.data())?;
    }
    let shapes: HashMap<&str, &[usize]> = params.iter().map(|p| (p.name(), p.shape())).collect();
    for (name, m, v) in optimizer.moments() {
        let Some(shape) = shapes.get(name) else {
            return Err(Error::invalid(format!(
                "the optimizer holds moments for `{name}`, which is not one of the parameters"
            )));
        };
        let numel: usize = shape.iter().product();
        if m.len() != numel || v.len() != numel {
            return Err(Error::invalid(format!(
                "the moments of `{name}` hold {} values and the parameter {numel}",
                m.len()
            )));
        }
        file.push(format!("{M_PREFIX}{name}"), shape, m)?;
        file.push(format!("{V_PREFIX}{name}"), shape, v)?;
    }
    Ok(file)
}

/// Reads the progress and checks the header of a state file, without touching anything.
///
/// # Errors
///
/// Returns an error if the format or version entry is missing, wrong or from another
/// version, or if the step, epoch or position is missing or not a whole number.
pub fn progress_of(file: &TensorFile) -> Result<Progress> {
    let metadata = file.metadata();
    match metadata.get(FORMAT_KEY) {
        Some(name) if name == STATE_FORMAT => {}
        Some(other) => {
            return Err(Error::parse(
                "training state",
                0,
                format!("`{FORMAT_KEY}` is `{other}`, not `{STATE_FORMAT}`"),
            ));
        }
        None => {
            return Err(Error::parse(
                "training state",
                0,
                format!("the file has no `{FORMAT_KEY}` entry, so it is not a {STATE_FORMAT} file"),
            ));
        }
    }
    let number = |key: &str| -> Result<u64> {
        let Some(text) = metadata.get(key) else {
            return Err(Error::parse(
                "training state",
                0,
                format!("the file has no `{key}` entry"),
            ));
        };
        text.parse().map_err(|_| {
            Error::parse("training state", 0, format!("`{key}` is `{text}`, not a whole number"))
        })
    };
    let version = number(VERSION_KEY)?;
    if version != u64::from(STATE_VERSION) {
        return Err(Error::UnsupportedVersion {
            what: "training state".to_string(),
            found: u32::try_from(version).unwrap_or(u32::MAX),
            supported: STATE_VERSION,
        });
    }
    let position = number(POSITION_KEY)?;
    Ok(Progress {
        step: number(STEP_KEY)?,
        epoch: number(EPOCH_KEY)?,
        position: usize::try_from(position).map_err(|_| {
            Error::parse("training state", 0, format!("`{POSITION_KEY}` is too large: {position}"))
        })?,
    })
}

/// Puts the weights and the optimizer state of `file` into `params` and `optimizer`, and
/// returns the progress. Nothing is changed unless the whole load can succeed.
///
/// # Errors
///
/// Returns an error if the header is not acceptable (see [`progress_of`]), if the saved run
/// had other settings than `config`, if the optimizer's step count is not the run's step, if
/// a parameter or one of its moments is missing or has another shape, if the file holds a
/// tensor nobody wants, if there are moments in a run that has taken no step, or if the
/// moments are refused by [`AdamW::restore`].
pub fn load_state(
    file: &TensorFile,
    params: &mut [&mut Parameter],
    optimizer: &mut AdamW,
    config: &TrainConfig,
) -> Result<Progress> {
    let progress = progress_of(file)?;
    config.check_matches(file.metadata())?;

    let mut wanted = HashSet::new();
    for param in params.iter() {
        if !wanted.insert(param.name().to_string()) {
            return Err(Error::invalid(format!("two parameters are named `{}`", param.name())));
        }
    }
    let by_name: HashMap<&str, &NamedTensor> =
        file.tensors().iter().map(|tensor| (tensor.name(), tensor)).collect();

    // Every tensor in the file has to be one of the three kinds, for a parameter we have.
    let mut unexpected = Vec::new();
    for tensor in file.tensors() {
        let name = tensor.name();
        let known = [PARAM_PREFIX, M_PREFIX, V_PREFIX]
            .iter()
            .find_map(|prefix| name.strip_prefix(prefix))
            .is_some_and(|rest| wanted.contains(rest));
        if !known {
            unexpected.push(name);
        }
    }
    if !unexpected.is_empty() {
        return Err(Error::invalid(format!(
            "the saved run holds {} tensor(s) this model does not have: {}",
            unexpected.len(),
            list_names(&unexpected)
        )));
    }

    let with_moments = progress.step > 0;
    let mut missing = Vec::new();
    for param in params.iter() {
        let mut kinds = vec![PARAM_PREFIX];
        if with_moments {
            kinds.extend([M_PREFIX, V_PREFIX]);
        }
        for prefix in kinds {
            let key = format!("{prefix}{}", param.name());
            match by_name.get(key.as_str()) {
                None => missing.push(key),
                Some(tensor) if tensor.shape() != param.shape() => {
                    return Err(Error::shape(
                        "load_state",
                        format!(
                            "`{key}` has shape {:?} in the saved run but the parameter has {:?}",
                            tensor.shape(),
                            param.shape()
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
    }
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().map(String::as_str).collect();
        return Err(Error::invalid(format!(
            "the saved run is missing {} tensor(s): {}",
            missing.len(),
            list_names(&shown)
        )));
    }
    if !with_moments && file.tensors().iter().any(|t| !t.name().starts_with(PARAM_PREFIX)) {
        return Err(Error::invalid(
            "the saved run has taken no step and yet holds optimizer moments".to_string(),
        ));
    }

    let mut moments = Vec::new();
    if with_moments {
        for param in params.iter() {
            let get = |prefix: &str| by_name[format!("{prefix}{}", param.name()).as_str()].data();
            moments.push((
                param.name().to_string(),
                get(M_PREFIX).to_vec(),
                get(V_PREFIX).to_vec(),
            ));
        }
    }
    optimizer.restore(progress.step, moments)?;
    for param in params.iter_mut() {
        let tensor = by_name[format!("{PARAM_PREFIX}{}", param.name()).as_str()];
        param.set_data(tensor.data())?;
    }
    Ok(progress)
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

    fn params() -> (Parameter, Parameter) {
        (
            Parameter::new("w", Tensor::from_vec(vec![0.5, -0.5, 1.0, -1.0], &[2, 2]).unwrap()),
            Parameter::new("b", Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap()),
        )
    }

    fn config() -> TrainConfig {
        TrainConfig::new(2, 4, 100)
    }

    fn after_steps(n: usize) -> (Parameter, Parameter, AdamW) {
        let (mut w, mut b) = params();
        let mut opt = config().optimizer().unwrap();
        for i in 0..n {
            let x = i as f32 + 1.0;
            let grads = [vec![0.1 * x, -0.2, 0.3 / x, 0.4], vec![0.5, -0.5 * x, 0.1]];
            opt.step(&mut [&mut w, &mut b], &grads, 0.01).unwrap();
        }
        (w, b, opt)
    }

    fn progress(step: u64) -> Progress {
        Progress { step, epoch: 2, position: 7 }
    }

    fn file_of(w: &Parameter, b: &Parameter, opt: &AdamW, step: u64) -> TensorFile {
        state_to_file(&[w, b], opt, progress(step), &config(), &BTreeMap::new()).unwrap()
    }

    fn bits(p: &Parameter) -> Vec<u32> {
        p.data().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn a_state_survives_a_round_trip_through_bytes() {
        let (w, b, opt) = after_steps(3);
        let file = file_of(&w, &b, &opt, 3);
        let names: Vec<&str> = file.tensors().iter().map(NamedTensor::name).collect();
        assert_eq!(
            names,
            ["param/w", "param/b", "adamw.m/b", "adamw.v/b", "adamw.m/w", "adamw.v/w"]
        );
        assert_eq!(file.metadata()["format"], STATE_FORMAT);
        assert_eq!(file.metadata()["step"], "3");
        assert_eq!(file.metadata()["epoch"], "2");
        assert_eq!(file.metadata()["position"], "7");

        let file = TensorFile::from_bytes(&file.to_bytes().unwrap()).unwrap();
        let (mut w2, mut b2) = params();
        let mut opt2 = config().optimizer().unwrap();
        let got = load_state(&file, &mut [&mut w2, &mut b2], &mut opt2, &config()).unwrap();
        assert_eq!(got, progress(3));
        assert_eq!(bits(&w2), bits(&w));
        assert_eq!(bits(&b2), bits(&b));
        assert_eq!(opt2.step_count(), 3);
        let held = |o: &AdamW| -> Vec<(String, Vec<u32>, Vec<u32>)> {
            o.moments()
                .map(|(n, m, v)| {
                    (
                        n.to_string(),
                        m.iter().map(|x| x.to_bits()).collect(),
                        v.iter().map(|x| x.to_bits()).collect(),
                    )
                })
                .collect()
        };
        assert_eq!(held(&opt2), held(&opt));
    }

    #[test]
    fn a_run_before_its_first_step_has_no_moments_and_loads() {
        let (w, b) = params();
        let opt = config().optimizer().unwrap();
        let file = file_of(&w, &b, &opt, 0);
        assert_eq!(file.tensors().len(), 2);
        let (mut w2, mut b2) = (
            Parameter::new("w", Tensor::zeros(&[2, 2]).unwrap()),
            Parameter::new("b", Tensor::zeros(&[3]).unwrap()),
        );
        let mut opt2 = config().optimizer().unwrap();
        let got = load_state(&file, &mut [&mut w2, &mut b2], &mut opt2, &config()).unwrap();
        assert_eq!(got.step, 0);
        assert_eq!(bits(&w2), bits(&w));
        assert_eq!(opt2.moments().count(), 0);
    }

    #[test]
    fn the_writer_refuses_what_it_should() {
        let (w, b, opt) = after_steps(2);
        let cfg = config();
        let none = BTreeMap::new();
        // The step count must be the optimizer's.
        assert!(state_to_file(&[&w, &b], &opt, progress(3), &cfg, &none).is_err());
        // Two parameters with one name.
        let twin = Parameter::new("w", Tensor::zeros(&[2, 2]).unwrap());
        let err = state_to_file(&[&w, &twin, &b], &opt, progress(2), &cfg, &none)
            .unwrap_err()
            .to_string();
        assert!(err.contains("two parameters are named `w`"), "{err}");
        // Moments for a parameter that is not given.
        let err = state_to_file(&[&w], &opt, progress(2), &cfg, &none).unwrap_err().to_string();
        assert!(err.contains("`b`"), "{err}");
        // A parameter of another size than its moments.
        let wide = Parameter::new("b", Tensor::zeros(&[4]).unwrap());
        assert!(state_to_file(&[&w, &wide], &opt, progress(2), &cfg, &none).is_err());
        // Reserved keys.
        for key in ["format", "state_version", "step", "epoch", "position", "train.seed"] {
            let mut extra = BTreeMap::new();
            extra.insert(key.to_string(), "x".to_string());
            assert!(state_to_file(&[&w, &b], &opt, progress(2), &cfg, &extra).is_err(), "{key}");
        }
        let mut extra = BTreeMap::new();
        extra.insert("data.sequences".to_string(), "12".to_string());
        let file = state_to_file(&[&w, &b], &opt, progress(2), &cfg, &extra).unwrap();
        assert_eq!(file.metadata()["data.sequences"], "12");
    }

    fn load_into_fresh(
        file: &TensorFile,
        cfg: &TrainConfig,
    ) -> (Result<Progress>, Parameter, Parameter, AdamW) {
        let (mut w, mut b) = (
            Parameter::new("w", Tensor::full(&[2, 2], 9.0).unwrap()),
            Parameter::new("b", Tensor::full(&[3], 9.0).unwrap()),
        );
        let mut opt = cfg.optimizer().unwrap();
        let result = load_state(file, &mut [&mut w, &mut b], &mut opt, cfg);
        (result, w, b, opt)
    }

    fn assert_untouched(w: &Parameter, b: &Parameter, opt: &AdamW) {
        assert!(w.data().iter().chain(b.data()).all(|x| *x == 9.0));
        assert_eq!(opt.step_count(), 0);
        assert_eq!(opt.moments().count(), 0);
    }

    #[test]
    fn a_load_that_is_refused_changes_nothing() {
        let (w, b, opt) = after_steps(2);
        let good = file_of(&w, &b, &opt, 2);

        let rebuilt = |keep: &dyn Fn(&NamedTensor) -> bool, edit: &dyn Fn(&mut TensorFile)| {
            let mut f = TensorFile::new();
            for (k, v) in good.metadata() {
                f.set_metadata(k.as_str(), v.as_str());
            }
            for t in good.tensors() {
                if keep(t) {
                    f.push(t.name(), t.shape(), t.data()).unwrap();
                }
            }
            edit(&mut f);
            f
        };
        let all = |_: &NamedTensor| true;
        let nothing = |_: &mut TensorFile| {};

        let cases: Vec<(&str, TensorFile, &str)> = vec![
            ("no parameter", rebuilt(&|t| t.name() != "param/w", &nothing), "param/w"),
            ("no first moment", rebuilt(&|t| t.name() != "adamw.m/b", &nothing), "adamw.m/b"),
            ("no second moment", rebuilt(&|t| t.name() != "adamw.v/w", &nothing), "adamw.v/w"),
            (
                "a tensor nobody wants",
                rebuilt(&all, &|f| f.push("param/extra", &[1], &[0.0]).unwrap()),
                "param/extra",
            ),
            (
                "a tensor of another kind",
                rebuilt(&all, &|f| f.push("other/w", &[1], &[0.0]).unwrap()),
                "other/w",
            ),
            (
                "another format",
                rebuilt(&all, &|f| f.set_metadata("format", "something-else")),
                "something-else",
            ),
            ("another version", rebuilt(&all, &|f| f.set_metadata("state_version", "2")), ""),
            ("no step", rebuilt(&all, &|f| f.set_metadata("step", "many")), "many"),
            (
                "other settings",
                rebuilt(&all, &|f| f.set_metadata("train.max_lr", "0.5")),
                "train.max_lr",
            ),
        ];
        for (name, file, mention) in cases {
            let (result, w, b, opt) = load_into_fresh(&file, &config());
            let err = result.expect_err(name);
            assert!(err.to_string().contains(mention), "{name}: {err}");
            assert_untouched(&w, &b, &opt);
        }

        // A wrong shape, and moments in a run with no step, and a step count the moments
        // disagree with.
        let mut wrong_shape = TensorFile::new();
        for (k, v) in good.metadata() {
            wrong_shape.set_metadata(k.as_str(), v.as_str());
        }
        for t in good.tensors() {
            if t.name() == "param/b" {
                wrong_shape.push(t.name(), &[1, 3], t.data()).unwrap();
            } else {
                wrong_shape.push(t.name(), t.shape(), t.data()).unwrap();
            }
        }
        let (result, w2, b2, opt2) = load_into_fresh(&wrong_shape, &config());
        assert!(result.unwrap_err().to_string().contains("param/b"));
        assert_untouched(&w2, &b2, &opt2);

        let step_zero = good_with_step(&good, "0");
        let (result, w3, b3, opt3) = load_into_fresh(&step_zero, &config());
        assert!(result.unwrap_err().to_string().contains("no step"));
        assert_untouched(&w3, &b3, &opt3);
    }

    fn good_with_step(good: &TensorFile, step: &str) -> TensorFile {
        let mut f = TensorFile::new();
        for (k, v) in good.metadata() {
            f.set_metadata(k.as_str(), v.as_str());
        }
        f.set_metadata("step", step);
        for t in good.tensors() {
            f.push(t.name(), t.shape(), t.data()).unwrap();
        }
        f
    }

    #[test]
    fn moments_that_the_optimizer_refuses_leave_the_weights_alone() {
        let (w, b, opt) = after_steps(2);
        let good = file_of(&w, &b, &opt, 2);
        let mut f = TensorFile::new();
        for (k, v) in good.metadata() {
            f.set_metadata(k.as_str(), v.as_str());
        }
        for t in good.tensors() {
            if t.name() == "adamw.v/w" {
                f.push(t.name(), t.shape(), &[-1.0, 0.0, 0.0, 0.0]).unwrap();
            } else {
                f.push(t.name(), t.shape(), t.data()).unwrap();
            }
        }
        let (result, w2, b2, opt2) = load_into_fresh(&f, &config());
        assert!(result.unwrap_err().to_string().contains("second moment"));
        assert_untouched(&w2, &b2, &opt2);
    }

    #[test]
    fn two_parameters_with_one_name_are_refused_on_load() {
        let (w, b, opt) = after_steps(1);
        let file = file_of(&w, &b, &opt, 1);
        let (mut w1, mut w2) = (
            Parameter::new("w", Tensor::full(&[2, 2], 9.0).unwrap()),
            Parameter::new("w", Tensor::full(&[2, 2], 9.0).unwrap()),
        );
        let mut opt2 = config().optimizer().unwrap();
        let err = load_state(&file, &mut [&mut w1, &mut w2], &mut opt2, &config())
            .unwrap_err()
            .to_string();
        assert!(err.contains("two parameters are named `w`"), "{err}");
        assert!(w1.data().iter().all(|x| *x == 9.0));
        assert_eq!(opt2.step_count(), 0);
    }

    #[test]
    fn the_list_of_missing_names_is_cut_short() {
        let names: Vec<String> = (0..8).map(|i| format!("n{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let text = list_names(&refs);
        assert!(text.starts_with("`n0`, `n1`"));
        assert!(text.ends_with("and 3 more"), "{text}");
        assert_eq!(list_names(&refs[..5]), "`n0`, `n1`, `n2`, `n3`, `n4`");
    }

    #[test]
    fn progress_is_read_without_touching_anything() {
        let (w, b, opt) = after_steps(1);
        let file = file_of(&w, &b, &opt, 1);
        assert_eq!(progress_of(&file).unwrap(), progress(1));
        let mut broken = good_with_step(&file, "1");
        broken.set_metadata("position", "-3");
        assert!(progress_of(&broken).unwrap_err().to_string().contains("position"));
        let mut old = good_with_step(&file, "1");
        old.set_metadata("state_version", "9");
        match progress_of(&old).unwrap_err() {
            Error::UnsupportedVersion { found, supported, .. } => {
                assert_eq!((found, supported), (9, STATE_VERSION));
            }
            other => panic!("{other}"),
        }
        assert!(progress_of(&TensorFile::new()).is_err());
    }
}
