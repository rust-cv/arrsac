#![feature(type_alias_impl_trait)]

use sample_consensus::{Consensus, Estimator, EstimatorData, Model};

struct Arrsac {
    max_candidate_hypothesis: usize,
    block_size: usize,
    probability_of_rejecting_good_model: f32,
}

impl<E> Consensus<E> for Arrsac
where
    E: Estimator,
{
    type Inliers = Inliers;
    type ModelInliers = impl Iterator<Item = (E::Model, Self::Inliers)>;

    fn model(&mut self, estimator: &E, data: &[EstimatorData<E>]) -> Option<E::Model> {
        unimplemented!()
    }

    fn models(&mut self, estimator: &E, data: &[EstimatorData<E>]) -> Self::ModelInliers {
        vec![].into_iter()
    }
}

pub struct Inliers;

impl Iterator for Inliers {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        unimplemented!()
    }
}

/// Algorithm 1 in "Randomized RANSAC with Sequential Probability Ratio Test".
///
/// Items with a residual error less than `threshold` are considered inliers.
///
/// `positive_likelyhood` - `δ / ε`
/// `negative_likelyhood` - `(1 - δ) / (1 - ε)`
pub fn likelyhood_ratio<'a, M: Model>(
    data: impl Iterator<Item = &'a M::Data>,
    model: &M,
    threshold: f32,
    positive_likelyhood: f32,
    negative_likelyhood: f32,
) -> f32
where
    M::Data: 'a,
{
    let (inliers, outliers) = data.fold((0, 0), |(inliers, outliers), data| {
        if model.residual(data) < threshold {
            (inliers + 1, outliers)
        } else {
            (inliers, outliers + 1)
        }
    });

    positive_likelyhood.powi(inliers) * negative_likelyhood.powi(outliers)
}
