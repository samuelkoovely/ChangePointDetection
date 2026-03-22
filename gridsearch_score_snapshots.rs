//! Rust port of `gridsearch_score_snapshots.py`.
//!
//! This file ports the reusable grid-search logic into Rust while keeping the
//! two Python-specific boundaries explicit:
//! 1. entropy-signal generation still depends on project-specific temporal
//!    network objects (`TemporalNetwork`, `signal_generation.py`);
//! 2. change-point detection currently depends on `ruptures.KernelCPD`.
//!
//! The Rust code below therefore focuses on the parts that are stable and worth
//! porting now:
//! - sample and signal data structures
//! - signal bundling by lambda/window
//! - breakpoint post-processing
//! - Hausdorff/F1 evaluation
//! - best-parameter selection
//! - grid-search orchestration over pluggable signal generators / detectors
//!
//! Once the dense signal path is routed through `entropy_rs`, the `SignalGenerator`
//! implementation is the only part that should need project-specific wiring.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct CPSample<D> {
    pub data: D,
    pub true_change_points: Vec<f64>,
    pub n_bkps: usize,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SignalResult {
    pub lamda: f64,
    pub window: f64,
    pub k_samples: Vec<usize>,
    pub t_samples: Vec<f64>,
    pub signal: Vec<f64>,
}

impl SignalResult {
    pub fn validate(&self) -> Result<(), GridSearchError> {
        if self.k_samples.len() != self.t_samples.len() || self.k_samples.len() != self.signal.len()
        {
            return Err(GridSearchError::InvalidSignal {
                lamda: self.lamda,
                window: self.window,
                details: format!(
                    "length mismatch: k_samples={}, t_samples={}, signal={}",
                    self.k_samples.len(),
                    self.t_samples.len(),
                    self.signal.len()
                ),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LambdaSignalBundle {
    pub lamda: f64,
    pub windows: Vec<f64>,
    pub signals_by_window: Vec<Vec<SignalResult>>,
    pub sample_names: Vec<Option<String>>,
}

#[derive(Clone, Debug)]
pub struct WindowEvaluation {
    pub window: f64,
    pub f1_scores: Vec<f64>,
    pub hausdorff_scores: Vec<f64>,
    pub predicted_change_points: Vec<Vec<usize>>,
    pub mean_f1: f64,
    pub mean_hausdorff: f64,
}

#[derive(Clone, Debug)]
pub struct LambdaEvaluation {
    pub lamda: f64,
    pub windows: Vec<f64>,
    pub score_array: Vec<f64>,
    pub hausdorff_array: Vec<f64>,
    pub per_window: Vec<WindowEvaluation>,
    pub sample_names: Vec<Option<String>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMetric {
    F1,
    Hausdorff,
}

#[derive(Clone, Debug)]
pub struct GridSearchSummary {
    pub lambdas: Vec<f64>,
    pub windows: Vec<f64>,
    pub margin: f64,
    pub sample_fraction: f64,
    pub num_samples: usize,
    pub num_lambda_jobs: usize,
    pub num_parameter_pairs: usize,
    pub save_signals: bool,
    pub signals_outdir: Option<PathBuf>,
    pub selection_metric: SelectionMetric,
    pub score_array: Vec<Vec<f64>>,
    pub hausdorff_array: Vec<Vec<f64>>,
    pub lambda_results: Vec<LambdaEvaluation>,
    pub best_index: Option<(usize, usize)>,
    pub best_lamda: Option<f64>,
    pub best_window: Option<f64>,
    pub best_score: f64,
    pub best_f1: f64,
    pub best_hausdorff: f64,
    pub elapsed_seconds: f64,
    pub signal_generation_phase_seconds: f64,
    pub detection_metrics_phase_seconds: f64,
}

impl GridSearchSummary {
    pub fn selection_array(&self) -> &[Vec<f64>] {
        match self.selection_metric {
            SelectionMetric::F1 => &self.score_array,
            SelectionMetric::Hausdorff => &self.hausdorff_array,
        }
    }

    pub fn result_for_lambda(&self, lamda: f64) -> Option<&LambdaEvaluation> {
        self.lambda_results
            .iter()
            .find(|result| result.lamda.to_bits() == lamda.to_bits())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BestSelection {
    pub best_index: Option<(usize, usize)>,
    pub best_lamda: Option<f64>,
    pub best_window: Option<f64>,
    pub best_score: f64,
    pub best_f1: f64,
    pub best_hausdorff: f64,
}

pub trait SignalGenerator<D> {
    fn compute_signals_for_lambda(
        &self,
        data: &D,
        lamda: f64,
        windows: &[f64],
        sample_fraction: f64,
        p0: Option<&[f64]>,
    ) -> Result<Vec<SignalResult>, String>;
}

pub trait ChangePointDetector {
    fn detect(&self, signal: &[f64], n_bkps: usize) -> Result<Vec<usize>, String>;
}

pub trait SignalSaver {
    fn save_signal_result(&self, result: &SignalResult, sample_dir: &Path) -> Result<(), String>;
}

#[derive(Clone, Debug, PartialEq)]
pub enum GridSearchError {
    InvalidSignal {
        lamda: f64,
        window: f64,
        details: String,
    },
    MissingWindowResult {
        lamda: f64,
        expected_window: f64,
        sample_name: Option<String>,
    },
    DuplicateWindowResult {
        lamda: f64,
        window: f64,
        sample_name: Option<String>,
    },
    SignalCountMismatch {
        lamda: f64,
        window: f64,
        expected: usize,
        got: usize,
    },
    Generator {
        lamda: f64,
        sample_name: Option<String>,
        message: String,
    },
    Detector {
        lamda: f64,
        window: f64,
        sample_name: Option<String>,
        message: String,
    },
    Saver {
        lamda: f64,
        window: f64,
        sample_name: String,
        message: String,
    },
}

impl fmt::Display for GridSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridSearchError::InvalidSignal {
                lamda,
                window,
                details,
            } => write!(
                f,
                "invalid signal for lamda={lamda}, window={window}: {details}"
            ),
            GridSearchError::MissingWindowResult {
                lamda,
                expected_window,
                sample_name,
            } => write!(
                f,
                "missing window result for lamda={lamda}, window={expected_window}, sample={}",
                sample_name.as_deref().unwrap_or("<unnamed>")
            ),
            GridSearchError::DuplicateWindowResult {
                lamda,
                window,
                sample_name,
            } => write!(
                f,
                "duplicate window result for lamda={lamda}, window={window}, sample={}",
                sample_name.as_deref().unwrap_or("<unnamed>")
            ),
            GridSearchError::SignalCountMismatch {
                lamda,
                window,
                expected,
                got,
            } => write!(
                f,
                "expected {expected} signals for lamda={lamda}, window={window}, got {got}"
            ),
            GridSearchError::Generator {
                lamda,
                sample_name,
                message,
            } => write!(
                f,
                "signal generation failed for lamda={lamda}, sample={}: {message}",
                sample_name.as_deref().unwrap_or("<unnamed>")
            ),
            GridSearchError::Detector {
                lamda,
                window,
                sample_name,
                message,
            } => write!(
                f,
                "change-point detection failed for lamda={lamda}, window={window}, sample={}: {message}",
                sample_name.as_deref().unwrap_or("<unnamed>")
            ),
            GridSearchError::Saver {
                lamda,
                window,
                sample_name,
                message,
            } => write!(
                f,
                "saving signal failed for lamda={lamda}, window={window}, sample={sample_name}: {message}"
            ),
        }
    }
}

impl std::error::Error for GridSearchError {}

pub fn detect_change_points_from_signal<D: ChangePointDetector>(
    detector: &D,
    signal: &[f64],
    n_bkps: usize,
) -> Result<Vec<usize>, GridSearchError> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let mut breakpoint_indices = detector
        .detect(signal, n_bkps)
        .map_err(|message| GridSearchError::Detector {
            lamda: f64::NAN,
            window: f64::NAN,
            sample_name: None,
            message,
        })?;

    breakpoint_indices.retain(|&idx| idx < signal.len());
    breakpoint_indices.sort_unstable();
    breakpoint_indices.dedup();

    Ok(breakpoint_indices)
}

pub fn compute_and_store_signals_for_lambda<D, G>(
    samples: &[CPSample<D>],
    lamda: f64,
    windows: &[f64],
    sample_fraction: f64,
    p0: Option<&[f64]>,
    generator: &G,
    saver: Option<&dyn SignalSaver>,
    signals_outdir: Option<&Path>,
) -> Result<LambdaSignalBundle, GridSearchError>
where
    G: SignalGenerator<D>,
{
    let mut signals_by_window = vec![Vec::with_capacity(samples.len()); windows.len()];
    let sample_names = samples.iter().map(|sample| sample.name.clone()).collect();

    for (sample_idx, sample) in samples.iter().enumerate() {
        let generated = generator
            .compute_signals_for_lambda(&sample.data, lamda, windows, sample_fraction, p0)
            .map_err(|message| GridSearchError::Generator {
                lamda,
                sample_name: sample.name.clone(),
                message,
            })?;

        let mut seen = vec![false; windows.len()];

        for result in generated {
            result.validate()?;

            let Some(window_idx) = find_window_index(windows, result.window) else {
                continue;
            };

            if seen[window_idx] {
                return Err(GridSearchError::DuplicateWindowResult {
                    lamda,
                    window: result.window,
                    sample_name: sample.name.clone(),
                });
            }
            seen[window_idx] = true;

            if let (Some(saver), Some(base_dir)) = (saver, signals_outdir) {
                let sample_dir = base_dir.join(sample_dir_name(sample_idx, sample.name.as_deref()));
                saver
                    .save_signal_result(&result, &sample_dir)
                    .map_err(|message| GridSearchError::Saver {
                        lamda,
                        window: result.window,
                        sample_name: sample_dir
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("sample")
                            .to_string(),
                        message,
                    })?;
            }

            signals_by_window[window_idx].push(result);
        }

        for (window_idx, matched) in seen.iter().enumerate() {
            if !matched {
                return Err(GridSearchError::MissingWindowResult {
                    lamda,
                    expected_window: windows[window_idx],
                    sample_name: sample.name.clone(),
                });
            }
        }
    }

    Ok(LambdaSignalBundle {
        lamda,
        windows: windows.to_vec(),
        signals_by_window,
        sample_names,
    })
}

pub fn evaluate_precomputed_lambda_signals<D, Det>(
    samples: &[CPSample<D>],
    lamda: f64,
    windows: &[f64],
    margin: f64,
    signals_by_window: &[Vec<SignalResult>],
    detector: &Det,
) -> Result<LambdaEvaluation, GridSearchError>
where
    Det: ChangePointDetector,
{
    let mut score_array = Vec::with_capacity(windows.len());
    let mut hausdorff_array = Vec::with_capacity(windows.len());
    let mut per_window = Vec::with_capacity(windows.len());
    let sample_names = samples.iter().map(|sample| sample.name.clone()).collect();

    for (window_idx, window) in windows.iter().copied().enumerate() {
        let window_signals = signals_by_window.get(window_idx).ok_or_else(|| {
            GridSearchError::SignalCountMismatch {
                lamda,
                window,
                expected: samples.len(),
                got: 0,
            }
        })?;

        if window_signals.len() != samples.len() {
            return Err(GridSearchError::SignalCountMismatch {
                lamda,
                window,
                expected: samples.len(),
                got: window_signals.len(),
            });
        }

        let mut window_f1_scores = Vec::with_capacity(samples.len());
        let mut window_hausdorff_scores = Vec::with_capacity(samples.len());
        let mut window_predicted_change_points = Vec::with_capacity(samples.len());

        for (sample, signal_result) in samples.iter().zip(window_signals.iter()) {
            signal_result.validate()?;

            let mut predicted = detector
                .detect(&signal_result.signal, sample.n_bkps)
                .map_err(|message| GridSearchError::Detector {
                    lamda,
                    window,
                    sample_name: sample.name.clone(),
                    message,
                })?;

            predicted.retain(|&idx| idx < signal_result.signal.len());
            predicted.sort_unstable();
            predicted.dedup();

            let predicted_f64: Vec<f64> = predicted.iter().map(|&idx| idx as f64).collect();
            let f1 = f1_score(&sample.true_change_points, &predicted_f64, margin);
            let hausdorff = hausdorff_distance(&sample.true_change_points, &predicted_f64);

            window_predicted_change_points.push(predicted);
            window_f1_scores.push(f1);
            window_hausdorff_scores.push(hausdorff);
        }

        let mean_f1 = mean_or_nan(&window_f1_scores);
        let mean_hausdorff = mean_or_nan(&window_hausdorff_scores);

        score_array.push(mean_f1);
        hausdorff_array.push(mean_hausdorff);
        per_window.push(WindowEvaluation {
            window,
            f1_scores: window_f1_scores,
            hausdorff_scores: window_hausdorff_scores,
            predicted_change_points: window_predicted_change_points,
            mean_f1,
            mean_hausdorff,
        });
    }

    Ok(LambdaEvaluation {
        lamda,
        windows: windows.to_vec(),
        score_array,
        hausdorff_array,
        per_window,
        sample_names,
    })
}

pub fn grid_search<D, G, Det>(
    samples: &[CPSample<D>],
    lambdas: &[f64],
    windows: &[f64],
    margin: f64,
    sample_fraction: f64,
    p0: Option<&[f64]>,
    generator: &G,
    detector: &Det,
    saver: Option<&dyn SignalSaver>,
    signals_outdir: Option<&Path>,
    selection_metric: SelectionMetric,
) -> Result<GridSearchSummary, GridSearchError>
where
    G: SignalGenerator<D>,
    Det: ChangePointDetector,
{
    let t0 = Instant::now();

    let t_signal_phase_start = Instant::now();
    let mut lambda_signal_results = Vec::with_capacity(lambdas.len());
    for &lamda in lambdas {
        let signal_bundle = compute_and_store_signals_for_lambda(
            samples,
            lamda,
            windows,
            sample_fraction,
            p0,
            generator,
            saver,
            signals_outdir,
        )?;
        lambda_signal_results.push(signal_bundle);
    }
    let signal_generation_phase_seconds = t_signal_phase_start.elapsed().as_secs_f64();

    let t_detection_phase_start = Instant::now();
    let mut lambda_results = Vec::with_capacity(lambda_signal_results.len());
    for signal_bundle in &lambda_signal_results {
        let evaluation = evaluate_precomputed_lambda_signals(
            samples,
            signal_bundle.lamda,
            windows,
            margin,
            &signal_bundle.signals_by_window,
            detector,
        )?;
        lambda_results.push(evaluation);
    }
    let detection_metrics_phase_seconds = t_detection_phase_start.elapsed().as_secs_f64();
    let elapsed_seconds = t0.elapsed().as_secs_f64();

    let score_array: Vec<Vec<f64>> = lambda_results
        .iter()
        .map(|result| result.score_array.clone())
        .collect();
    let hausdorff_array: Vec<Vec<f64>> = lambda_results
        .iter()
        .map(|result| result.hausdorff_array.clone())
        .collect();

    let best = select_best_pair(
        lambdas,
        windows,
        &score_array,
        &hausdorff_array,
        selection_metric,
    );

    Ok(GridSearchSummary {
        lambdas: lambdas.to_vec(),
        windows: windows.to_vec(),
        margin,
        sample_fraction,
        num_samples: samples.len(),
        num_lambda_jobs: lambdas.len(),
        num_parameter_pairs: lambdas.len() * windows.len(),
        save_signals: saver.is_some() && signals_outdir.is_some(),
        signals_outdir: signals_outdir.map(Path::to_path_buf),
        selection_metric,
        score_array,
        hausdorff_array,
        lambda_results,
        best_index: best.best_index,
        best_lamda: best.best_lamda,
        best_window: best.best_window,
        best_score: best.best_score,
        best_f1: best.best_f1,
        best_hausdorff: best.best_hausdorff,
        elapsed_seconds,
        signal_generation_phase_seconds,
        detection_metrics_phase_seconds,
    })
}

pub fn select_best_pair(
    lambdas: &[f64],
    windows: &[f64],
    score_array: &[Vec<f64>],
    hausdorff_array: &[Vec<f64>],
    selection_metric: SelectionMetric,
) -> BestSelection {
    match selection_metric {
        SelectionMetric::F1 => {
            if let Some((i, j)) = argmax_ignore_nan(score_array) {
                BestSelection {
                    best_index: Some((i, j)),
                    best_lamda: lambdas.get(i).copied(),
                    best_window: windows.get(j).copied(),
                    best_score: score_array[i][j],
                    best_f1: score_array[i][j],
                    best_hausdorff: hausdorff_array
                        .get(i)
                        .and_then(|row| row.get(j))
                        .copied()
                        .unwrap_or(f64::NAN),
                }
            } else {
                BestSelection::none()
            }
        }
        SelectionMetric::Hausdorff => {
            if let Some((i, j)) = argmin_ignore_nan_as_inf(hausdorff_array) {
                BestSelection {
                    best_index: Some((i, j)),
                    best_lamda: lambdas.get(i).copied(),
                    best_window: windows.get(j).copied(),
                    best_score: hausdorff_array[i][j],
                    best_f1: score_array
                        .get(i)
                        .and_then(|row| row.get(j))
                        .copied()
                        .unwrap_or(f64::NAN),
                    best_hausdorff: hausdorff_array[i][j],
                }
            } else {
                BestSelection::none()
            }
        }
    }
}

impl BestSelection {
    fn none() -> Self {
        Self {
            best_index: None,
            best_lamda: None,
            best_window: None,
            best_score: f64::NAN,
            best_f1: f64::NAN,
            best_hausdorff: f64::NAN,
        }
    }
}

pub fn hausdorff_distance(true_cps: &[f64], pred_cps: &[f64]) -> f64 {
    let mut true_list = true_cps.to_vec();
    let mut pred_list = pred_cps.to_vec();
    true_list.sort_by(f64_total_cmp);
    pred_list.sort_by(f64_total_cmp);

    if true_list.is_empty() && pred_list.is_empty() {
        return 0.0;
    }
    if true_list.is_empty() || pred_list.is_empty() {
        return f64::INFINITY;
    }

    let term1 = pred_list
        .iter()
        .map(|&pred| min_distance(pred, &true_list))
        .fold(f64::NEG_INFINITY, f64::max);
    let term2 = true_list
        .iter()
        .map(|&truth| min_distance(truth, &pred_list))
        .fold(f64::NEG_INFINITY, f64::max);

    term1.max(term2)
}

pub fn true_positives(true_cps: &[f64], pred_cps: &[f64], margin: f64) -> Vec<f64> {
    assert!(margin >= 0.0, "margin must be non-negative");

    let mut true_list = true_cps.to_vec();
    let mut pred_list = pred_cps.to_vec();
    true_list.sort_by(f64_total_cmp);
    pred_list.sort_by(f64_total_cmp);

    if true_list.is_empty() || pred_list.is_empty() {
        return Vec::new();
    }

    true_list
        .into_iter()
        .filter(|&truth| pred_list.iter().any(|&pred| (pred - truth).abs() < margin))
        .collect()
}

pub fn precision(true_cps: &[f64], pred_cps: &[f64], margin: f64) -> f64 {
    if pred_cps.is_empty() {
        return 0.0;
    }
    true_positives(true_cps, pred_cps, margin).len() as f64 / pred_cps.len() as f64
}

pub fn recall(true_cps: &[f64], pred_cps: &[f64], margin: f64) -> f64 {
    if true_cps.is_empty() {
        return 0.0;
    }
    true_positives(true_cps, pred_cps, margin).len() as f64 / true_cps.len() as f64
}

pub fn f1_score(true_cps: &[f64], pred_cps: &[f64], margin: f64) -> f64 {
    let prec = precision(true_cps, pred_cps, margin);
    let rec = recall(true_cps, pred_cps, margin);

    if (prec + rec) == 0.0 {
        return 0.0;
    }
    2.0 * prec * rec / (prec + rec)
}

fn find_window_index(windows: &[f64], window: f64) -> Option<usize> {
    windows
        .iter()
        .position(|candidate| candidate.to_bits() == window.to_bits())
}

fn sample_dir_name(sample_idx: usize, sample_name: Option<&str>) -> String {
    sample_name
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| format!("sample_{sample_idx}"))
}

fn mean_or_nan(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn argmax_ignore_nan(matrix: &[Vec<f64>]) -> Option<(usize, usize)> {
    let mut best_index = None;
    let mut best_value = f64::NEG_INFINITY;

    for (i, row) in matrix.iter().enumerate() {
        for (j, value) in row.iter().copied().enumerate() {
            if value.is_nan() {
                continue;
            }
            if best_index.is_none() || value > best_value {
                best_index = Some((i, j));
                best_value = value;
            }
        }
    }

    best_index
}

fn argmin_ignore_nan_as_inf(matrix: &[Vec<f64>]) -> Option<(usize, usize)> {
    let mut best_index = None;
    let mut best_value = f64::INFINITY;

    for (i, row) in matrix.iter().enumerate() {
        for (j, raw_value) in row.iter().copied().enumerate() {
            let value = if raw_value.is_nan() {
                f64::INFINITY
            } else {
                raw_value
            };
            if !value.is_finite() {
                continue;
            }
            if best_index.is_none() || value < best_value {
                best_index = Some((i, j));
                best_value = value;
            }
        }
    }

    best_index
}

fn min_distance(x: f64, points: &[f64]) -> f64 {
    points
        .iter()
        .map(|&point| (x - point).abs())
        .fold(f64::INFINITY, f64::min)
}

fn f64_total_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.total_cmp(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hausdorff_handles_empty_inputs() {
        assert_eq!(hausdorff_distance(&[], &[]), 0.0);
        assert!(hausdorff_distance(&[1.0], &[]).is_infinite());
    }

    #[test]
    fn f1_matches_python_logic() {
        let truth = [50.0, 100.0, 150.0];
        let pred = [48.0, 103.0, 170.0];
        let score = f1_score(&truth, &pred, 5.0);
        assert!((score - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn select_best_pair_uses_nan_rules() {
        let lambdas = [0.1, 0.2];
        let windows = [1.0, 5.0];
        let score = vec![vec![f64::NAN, 0.4], vec![0.9, 0.2]];
        let haus = vec![vec![f64::NAN, 3.0], vec![2.0, 1.5]];

        let best_f1 = select_best_pair(&lambdas, &windows, &score, &haus, SelectionMetric::F1);
        assert_eq!(best_f1.best_index, Some((1, 0)));
        assert_eq!(best_f1.best_lamda, Some(0.2));
        assert_eq!(best_f1.best_window, Some(1.0));

        let best_haus =
            select_best_pair(&lambdas, &windows, &score, &haus, SelectionMetric::Hausdorff);
        assert_eq!(best_haus.best_index, Some((1, 1)));
        assert_eq!(best_haus.best_lamda, Some(0.2));
        assert_eq!(best_haus.best_window, Some(5.0));
    }
}
