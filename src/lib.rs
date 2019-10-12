#![feature(type_alias_impl_trait)]

use sample_consensus::{Consensus, Estimator, EstimatorData, Model};

struct Arrsac {
    max_candidate_hypothesis: usize,
    block_size: usize,
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
fn likelyhood_ratio<'a, M: Model>(
    data: impl Iterator<Item = &'a M::Data>,
    model: &M,
    threshold: f32,
    positive_likelyhood: f32,
    negative_likelyhood: f32,
) -> f32
where
    M::Data: 'a,
{
    data.map(|data| {
        if model.residual(data) < threshold {
            positive_likelyhood
        } else {
            negative_likelyhood
        }
    })
    .product()
}
