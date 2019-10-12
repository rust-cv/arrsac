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
/// This tests if a model is accepted. Returns `true` on accepted and `false` on rejected.
///
/// `inlier_threshold` - The model residual error threshold between inliers and outliers
/// `positive_likelyhood_ratio` - `δ / ε`
/// `negative_likelyhood_ratio` - `(1 - δ) / (1 - ε)`
/// `likelyhood_ratio_threshold` - The parameter `A` in the paper.
pub fn adapted_sprt<'a, M: Model>(
    data: impl Iterator<Item = &'a M::Data>,
    model: &M,
    inlier_threshold: f32,
    positive_likelyhood_ratio: f32,
    negative_likelyhood_ratio: f32,
    likelyhood_ratio_threshold: f32,
) -> bool
where
    M::Data: 'a,
{
    let mut likelyhood_ratio = 1.0;
    for data in data {
        likelyhood_ratio *= if model.residual(data) < inlier_threshold {
            positive_likelyhood_ratio
        } else {
            negative_likelyhood_ratio
        };

        if likelyhood_ratio > likelyhood_ratio_threshold {
            return false;
        }
    }

    true
}
