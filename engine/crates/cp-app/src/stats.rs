//! Summaries of measured values: mean and nearest-rank percentiles.

/// Mean, percentiles and extremes of a series of measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    /// Number of values.
    pub count: usize,
    /// Arithmetic mean; 0 without values.
    pub mean: f64,
    /// Median (nearest rank).
    pub p50: f64,
    /// 95th percentile (nearest rank).
    pub p95: f64,
    /// 99th percentile (nearest rank).
    pub p99: f64,
    /// Smallest value.
    pub min: f64,
    /// Largest value.
    pub max: f64,
}

impl Summary {
    /// The summary of `values`; NaN values sort last (`f64::total_cmp`).
    pub fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = |percent: f64| {
            // Nearest rank: the smallest value with at least `percent` % of the values at or
            // below it.
            let index = ((percent / 100.0) * sorted.len() as f64).ceil() as usize;
            sorted[index.clamp(1, sorted.len()) - 1]
        };
        Self {
            count: values.len(),
            mean: values.iter().sum::<f64>() / values.len() as f64,
            p50: rank(50.0),
            p95: rank(95.0),
            p99: rank(99.0),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_use_the_nearest_rank() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        let summary = Summary::of(&values);
        assert_eq!(summary.count, 100);
        assert_eq!(summary.mean, 50.5);
        assert_eq!(summary.p50, 50.0);
        assert_eq!(summary.p95, 95.0);
        assert_eq!(summary.p99, 99.0);
        assert_eq!((summary.min, summary.max), (1.0, 100.0));
        let one = Summary::of(&[7.0]);
        assert_eq!((one.p50, one.p95, one.p99, one.max), (7.0, 7.0, 7.0, 7.0));
        assert_eq!(Summary::of(&[]), Summary::default());
    }
}
