use arrsac::Arrsac;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use sample_consensus::{Consensus, Estimator, Model};

pub struct Unsolvable(f64);

impl Model<i32> for Unsolvable {
    fn residual(&self, _data: &i32) -> f64 {
        return 0.01;
    }
}

pub struct UnsolvableEstimator {}

impl Estimator<i32> for UnsolvableEstimator {
    type Model = Unsolvable;
    type ModelIter = Option<Unsolvable>;
    const MIN_SAMPLES: usize = 4;

    fn estimate<I>(&self, _data: I) -> Self::ModelIter
    where
        I: Iterator<Item = i32> + Clone,
    {
        // Simulate all estimations failing.
        None
    }
}

/// It handles the case when the estimator fails to produce any solution from the given data
#[test]
pub fn no_valid_hypothesys() {
    let rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let mut arrsac = Arrsac::new(3.0, rng.clone());
    let estimator = UnsolvableEstimator {};
    let result = arrsac.model(&estimator, 1..999);
    assert!(result.is_none());
}
