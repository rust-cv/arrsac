#![feature(type_alias_impl_trait)]

use sample_consensus::{Consensus, Estimator, Model};

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

    fn model<I>(&mut self, estimator: &E, data: I) -> Option<E::Model>
    where
        I: Iterator<Item = <<E as Estimator>::Model as Model>::Data> + Clone,
    {
        unimplemented!()
    }

    fn models<I>(&mut self, estimator: &E, data: I) -> Self::ModelInliers
    where
        I: Iterator<Item = <<E as Estimator>::Model as Model>::Data> + Clone,
    {
        unimplemented!();
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
