#![no_std]

extern crate alloc;
use core::cmp::Reverse;

use alloc::{vec, vec::Vec};
use rand_core::RngCore;
use sample_consensus::{Consensus, Estimator, Model};

/// The ARRSAC algorithm for sample consensus.
///
/// Don't forget to shuffle your input data points to avoid bias before
/// using this consensus process. It will not shuffle your data for you.
/// If you do not shuffle, the output will be biased towards data at the beginning
/// of the inputs.
pub struct Arrsac<R> {
    initialization_hypotheses: usize,
    initialization_blocks: usize,
    max_candidate_hypotheses: usize,
    estimations_per_block: usize,
    block_size: usize,
    likelihood_ratio_threshold: f32,
    inlier_threshold: f64,
    rng: R,
    random_samples: Vec<u32>,
}

impl<R> Arrsac<R>
where
    R: RngCore,
{
    /// `rng` should have the same properties you would want for a Monte Carlo simulation.
    /// It should generate random numbers quickly without having any discernable patterns.
    ///
    /// The `inlier_threshold` is the one parameter that is always specific to your dataset.
    /// This must be set to the threshold in which a data point's residual is considered an inlier.
    /// Some of the other parameters may need to be configured based on the amount of data,
    /// such as `block_size`, `likelihood_ratio_threshold`, and `block_size`. However,
    /// `inlier_threshold` has to be set based on the residual function used with the model.
    ///
    /// `initial_epsilon` must be higher than `initial_delta`. If you modify these values,
    /// you need to make sure that within one `block_size` the `likelihood_ratio_threshold`
    /// can be reached and a model can be rejected. Basically, make sure that
    /// `((1.0 - delta) / (1.0 - epsilon))^block_size >>> likelihood_ratio_threshold`.
    /// This must be done to ensure outlier models are rejected during the initial generation
    /// phase, which only processes `block_size` datapoints.
    ///
    /// `initial_epsilon` should also be as large as you can set it where it is still relatively
    /// pessimistic. This is so that we can more easily reject a model early in the process
    /// to compute an updated value for delta during the adaptive process. This may not be possible
    /// and will depend on your data.
    pub fn new(inlier_threshold: f64, rng: R) -> Self {
        Self {
            initialization_hypotheses: 256,
            initialization_blocks: 4,
            max_candidate_hypotheses: 64,
            estimations_per_block: 64,
            block_size: 64,
            likelihood_ratio_threshold: 1e3,
            inlier_threshold,
            rng,
            random_samples: vec![],
        }
    }

    /// Number of models generated in the initial step when epsilon and delta are being estimated.
    ///
    /// Default: `256`
    pub fn initialization_hypotheses(self, initialization_hypotheses: usize) -> Self {
        Self {
            initialization_hypotheses,
            ..self
        }
    }

    /// Number of data blocks used to compute the initial estimate of delta and epsilon
    /// before proceeding with regular block processing. This is used instead of
    /// an initial epsilon and delta, which were suggested by the paper.
    ///
    /// Default: `4`
    pub fn initialization_blocks(self, initialization_blocks: usize) -> Self {
        Self {
            initialization_blocks,
            ..self
        }
    }

    /// Maximum number of best hypotheses to retain during block processing
    ///
    /// This number is halved on each block such that on block `n` the number of
    /// hypotheses retained is `max_candidate_hypotheses >> n`.
    ///
    /// Default: `64`
    pub fn max_candidate_hypotheses(self, max_candidate_hypotheses: usize) -> Self {
        Self {
            max_candidate_hypotheses,
            ..self
        }
    }

    /// Number of estmations (may generate multiple hypotheses) that will be ran
    /// for each block of data evaluated
    ///
    /// Default: `64`
    pub fn estimations_per_block(self, estimations_per_block: usize) -> Self {
        Self {
            estimations_per_block,
            ..self
        }
    }

    /// Number of data points evaluated before more hypotheses are generated
    ///
    /// Default: `64`
    pub fn block_size(self, block_size: usize) -> Self {
        Self { block_size, ..self }
    }

    /// Once a model reaches this level of unlikelihood, it is rejected. Set this
    /// higher to make it less restrictive, usually at the cost of more execution time.
    ///
    /// Increasing this will make it more likely to find a good result.
    ///
    /// Decreasing this will speed up execution.
    ///
    /// This ratio is not exposed as a parameter in the original paper, but is instead computed
    /// recursively for a few iterations. It is roughly equivalent to the **reciprocal** of the
    /// **probability of rejecting a good model**. You can use that to control the probability
    /// that a good model is rejected.
    ///
    /// Default: `1e3`
    pub fn likelihood_ratio_threshold(self, likelihood_ratio_threshold: f32) -> Self {
        Self {
            likelihood_ratio_threshold,
            ..self
        }
    }

    /// Residual threshold for determining if a data point is an inlier or an outlier of a model
    pub fn inlier_threshold(self, inlier_threshold: f64) -> Self {
        Self {
            inlier_threshold,
            ..self
        }
    }

    /// Adapted from algorithm 3 from "A Comparative Analysis of RANSAC Techniques Leading to Adaptive
    /// Real-Time Random Sample Consensus", but it was effectively rewritten to avoid the need for
    /// initial epsilon and delta.
    ///
    /// Returns the initial models (and their num inliers) sorted by decreasing inliers
    /// and `delta` in that order.
    fn initial_hypotheses<E, Data>(
        &mut self,
        estimator: &E,
        data: impl Iterator<Item = Data> + Clone,
    ) -> (Vec<(E::Model, usize)>, f32)
    where
        E: Estimator<Data>,
    {
        assert!(
            self.initialization_blocks > 0,
            "ARRSAC must have at least 1 initialization block"
        );
        // NOTE: This whole function is different than that specified in the ARRSAC paper.
        // The reason is that you needed to provide a good initial guess for epsilon and delta
        // otherwise it could lead to delta exceeding epsilon or situations where models could no
        // longer be rejected or were almost always rejected. This solution is an imperfect compomise
        // that assumes that delta will be roughly equal to the inlier ratio of the worst model generated,
        // which both assumes that the worst model is an outlier and that it is actually representative of the
        // population. The assumption of epsilon also assumes that the best model is an inlier, but is an
        // otherwise good initial guess. The other caveat with this approach is that a sufficiently large
        // set of initial datapoints is required to be able to accurately determine epsilon and delta.
        // Therefore a new paremeter is added to separate the normal blocks from the initial generation set.
        let mut hypotheses = vec![];
        // We don't want more than `block_size` data points to be used to evaluate models initially.
        let initial_datapoints = core::cmp::min(
            self.initialization_blocks * self.block_size,
            data.clone().count(),
        );
        // Generate the initial batch of random hypotheses and count their inliers and outliers.
        for _ in 0..self.initialization_hypotheses {
            for model in self.generate_random_hypotheses(estimator, data.clone()) {
                let inliers = self.count_inliers(data.clone().take(initial_datapoints), &model);
                hypotheses.push((model, inliers));
            }
        }

        // Bail early when no hypothesis was found.
        // This will cause execution to terminate.
        if hypotheses.is_empty() {
            return (hypotheses, 0.0);
        }

        // Sort the hypotheses by their inliers.
        hypotheses.sort_unstable_by_key(|&(_, inliers)| Reverse(inliers));

        // Compute epsilon and delta using the best and worst model generated.
        let epsilon = hypotheses
            .first()
            .map(|&(_, inliers)| inliers as f32 / initial_datapoints as f32)
            .unwrap_or_default();
        let delta = hypotheses
            .last()
            .map(|&(_, inliers)| if inliers < E::MIN_SAMPLES {E::MIN_SAMPLES} else {inliers} as f32 / initial_datapoints as f32)
            .unwrap_or_default();

        if epsilon < delta {
            // If epsilon is less than delta, then better hypotheses will get rejected and worse accepted,
            // which is counter to what we want. In this case, we had a bad initialization, so clear the hypotheses.
            // This will cause execution to terminate.
            hypotheses.clear();
            return (hypotheses, delta);
        }

        // Populate hypotheses with hypotheses generated from the inliers of the best hypothesis.
        // This will use the initialization datapoints and filter with SPRT.
        self.populate_hypotheses_sprt(
            estimator,
            &mut hypotheses,
            delta,
            data,
            initial_datapoints,
            self.initialization_hypotheses,
        );

        // Sort the hypotheses by their inliers.
        hypotheses.sort_unstable_by_key(|&(_, inliers)| Reverse(inliers));

        // Filter down the hypotheses to just the best ones.
        hypotheses.truncate(self.max_candidate_hypotheses >> (self.initialization_blocks - 1));

        (hypotheses, delta)
    }

    /// Populates `self.random_samples` using a len.
    fn populate_samples(&mut self, num: usize, len: usize) {
        // We can generate no hypotheses if the amout of data is too low.
        if len < num {
            panic!("cannot use arrsac without having enough samples");
        }
        let len = len as u32;
        // Threshold generation below adapted from randomize::RandRangeU32.
        let threshold = len.wrapping_neg() % len;
        self.random_samples.clear();
        for _ in 0..num {
            loop {
                let mul = u64::from(self.rng.next_u32()).wrapping_mul(u64::from(len));
                if mul as u32 >= threshold {
                    let s = (mul >> 32) as u32;
                    if !self.random_samples.contains(&s) {
                        self.random_samples.push(s);
                        break;
                    }
                }
            }
        }
    }

    fn populate_hypotheses_sprt<E, Data>(
        &mut self,
        estimator: &E,
        hypotheses: &mut Vec<(E::Model, usize)>,
        delta: f32,
        data: impl Iterator<Item = Data> + Clone,
        num_checked: usize,
        num_hypotheses: usize,
    ) where
        E: Estimator<Data>,
    {
        // Update epsilon using the best model.
        // Since epsilon can only increase and delta is fixed, we can be sure that these ratios
        // will still be valid (epsilon > delta).
        let epsilon = hypotheses[0].1 as f32 / num_checked as f32;
        // Create the likelihood ratios for inliers and outliers.
        let positive_likelihood_ratio = delta / epsilon;
        let negative_likelihood_ratio = (1.0 - delta) / (1.0 - epsilon);
        // Generate the list of inliers for the best model.
        let mut inliers = self.inliers(data.clone().take(num_checked), &hypotheses[0].0);
        if inliers.len() <= E::MIN_SAMPLES {
            // If we don't have enough samples to generate more models, then we should expand the inliers to
            // the entire dataset.
            inliers = self.inliers(data.clone().take(num_checked), &hypotheses[0].0);
        }
        // We generate hypotheses until we reach the initial num hypotheses.
        // We can't count the number generated because it could generate 0 hypotheses
        // and then the loop would continue indefinitely.
        let mut random_hypotheses = Vec::new();
        for _ in 0..num_hypotheses {
            random_hypotheses.extend(self.generate_random_hypotheses_subset(
                estimator,
                data.clone(),
                &inliers,
            ));
            for model in random_hypotheses.drain(..) {
                if let Some(inliers) = self.asprt(
                    data.clone().take(num_checked),
                    &model,
                    positive_likelihood_ratio,
                    negative_likelihood_ratio,
                    E::MIN_SAMPLES,
                ) {
                    hypotheses.push((model, inliers));
                }
            }
        }
    }

    /// Generates as many hypotheses as one call to `Estimator::estimate()` returns from all data.
    fn generate_random_hypotheses<E, Data>(
        &mut self,
        estimator: &E,
        data: impl Iterator<Item = Data> + Clone,
    ) -> E::ModelIter
    where
        E: Estimator<Data>,
    {
        self.populate_samples(E::MIN_SAMPLES, data.clone().count());
        estimator.estimate(
            self.random_samples
                .iter()
                .map(|&ix| data.clone().nth(ix as usize).unwrap()),
        )
    }

    /// Generates as many hypotheses as one call to `Estimator::estimate()` returns from a subset of the data.
    fn generate_random_hypotheses_subset<E, Data>(
        &mut self,
        estimator: &E,
        data: impl Iterator<Item = Data> + Clone,
        subset: &[usize],
    ) -> E::ModelIter
    where
        E: Estimator<Data>,
    {
        self.populate_samples(E::MIN_SAMPLES, subset.len());
        estimator.estimate(
            core::mem::take(&mut self.random_samples)
                .iter()
                .map(|&ix| data.clone().nth(subset[ix as usize]).unwrap()),
        )
    }

    /// Algorithm 1 in "Randomized RANSAC with Sequential Probability Ratio Test".
    ///
    /// This tests if a model is accepted. Returns `Some(inliers)` if accepted or `None` if rejected.
    ///
    /// `inlier_threshold` - The model residual error threshold between inliers and outliers
    /// `positive_likelihood_ratio` - `δ / ε`
    /// `negative_likelihood_ratio` - `(1 - δ) / (1 - ε)`
    fn asprt<Data, M: Model<Data>>(
        &self,
        data: impl Iterator<Item = Data>,
        model: &M,
        positive_likelihood_ratio: f32,
        negative_likelihood_ratio: f32,
        minimum_samples: usize,
    ) -> Option<usize> {
        let mut likelihood_ratio = 1.0;
        let mut inliers = 0;
        for data in data {
            likelihood_ratio *= if model.residual(&data) < self.inlier_threshold {
                inliers += 1;
                positive_likelihood_ratio
            } else {
                negative_likelihood_ratio
            };

            if likelihood_ratio > self.likelihood_ratio_threshold || likelihood_ratio.is_nan() {
                return None;
            }
        }

        (inliers >= minimum_samples).then(|| inliers)
    }

    /// Determines the number of inliers a model has.
    fn count_inliers<Data, M: Model<Data>>(
        &self,
        data: impl Iterator<Item = Data>,
        model: &M,
    ) -> usize {
        data.filter(|data| model.residual(data) < self.inlier_threshold)
            .count()
    }

    /// Gets indices of inliers for a model.
    fn inliers<Data, M: Model<Data>>(
        &self,
        data: impl Iterator<Item = Data>,
        model: &M,
    ) -> Vec<usize> {
        data.enumerate()
            .filter(|(_, data)| model.residual(data) < self.inlier_threshold)
            .map(|(ix, _)| ix)
            .collect()
    }
}

impl<E, R, Data> Consensus<E, Data> for Arrsac<R>
where
    E: Estimator<Data>,
    R: RngCore,
{
    type Inliers = Vec<usize>;

    fn model<I>(&mut self, estimator: &E, data: I) -> Option<E::Model>
    where
        I: Iterator<Item = Data> + Clone,
    {
        self.model_inliers(estimator, data).map(|(model, _)| model)
    }

    fn model_inliers<I>(&mut self, estimator: &E, data: I) -> Option<(E::Model, Self::Inliers)>
    where
        I: Iterator<Item = Data> + Clone,
    {
        // Don't do anything if we don't have enough data.
        if data.clone().count() < E::MIN_SAMPLES {
            return None;
        }
        // Generate the initial set of hypotheses. This also gets us an estimate of delta.
        let (mut hypotheses, delta) = self.initial_hypotheses(estimator, data.clone());

        // If there are no initial hypotheses then initialization failed, so exit early.
        if hypotheses.is_empty() {
            return None;
        }

        // Gradually increase how many datapoints we are evaluating until we evaluate them all.
        // This starts at the first block that was not evaluated in initial_hypotheses.
        'outer: for block in self.initialization_blocks.. {
            let samples_up_to_beginning_of_block = block * self.block_size;
            let samples_up_to_end_of_block = samples_up_to_beginning_of_block + self.block_size;
            // Score hypotheses with samples.
            for sample in samples_up_to_beginning_of_block..samples_up_to_end_of_block {
                // Score the hypotheses with the new datapoint.
                let new_datapoint = if let Some(datapoint) = data.clone().nth(sample) {
                    datapoint
                } else {
                    // We reached the last datapoint, so break out of the outer loop.
                    break 'outer;
                };
                for (hypothesis, inlier_count) in hypotheses.iter_mut() {
                    if hypothesis.residual(&new_datapoint) < self.inlier_threshold {
                        *inlier_count += 1;
                    }
                }
            }
            // Sort the hypotheses by their inliers to find the best.
            hypotheses.sort_unstable_by_key(|&(_, inliers)| Reverse(inliers));
            // Populate hypotheses with hypotheses that pass SPRT.
            self.populate_hypotheses_sprt(
                estimator,
                &mut hypotheses,
                delta,
                data.clone(),
                samples_up_to_end_of_block,
                self.estimations_per_block,
            );
            // This will retain at least half of the hypotheses each time
            // and gradually decrease as the number of samples we are evaluating increases.
            // NOTE:
            // The paper says to use a peculiar formula that just results in doing
            // this basic right shift below, but as written it contained some apparent errors in
            // where it was ran. This seems to be the correct location to do this.
            hypotheses.sort_unstable_by_key(|&(_, inliers)| Reverse(inliers));
            hypotheses.truncate(self.max_candidate_hypotheses >> block);
            if hypotheses.len() <= 1 {
                break 'outer;
            }
        }
        hypotheses
            .into_iter()
            .max_by_key(|&(_, inliers)| inliers)
            .map(|(model, _)| {
                let inliers = self.inliers(data.clone(), &model);
                (model, inliers)
            })
    }
}
