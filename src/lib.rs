#![feature(type_alias_impl_trait)]

use rand::Rng;
use sample_consensus::{Consensus, Estimator, EstimatorData, Model};
use std::vec;

pub struct Arrsac<R> {
    max_candidate_hypothesis: usize,
    block_size: usize,
    max_delta_estimations: usize,
    likelyhood_ratio_threshold: f32,
    epsilon_update_threshold: f32,
    initial_epsilon: f32,
    initial_delta: f32,
    inlier_threshold: f32,
    rng: R,
}

impl<R> Arrsac<R>
where
    R: Rng,
{
    /// `probability_of_rejecting_good_model` is poorly named. This is a temporary way to test the algorithm
    /// without estimating. Increasing this will also make it less likely to accept a bad model.
    /// Set it to something like `0.01` for testing.
    pub fn new(
        max_candidate_hypothesis: usize,
        block_size: usize,
        max_delta_estimations: usize,
        probability_of_rejecting_good_model: f32,
        epsilon_update_threshold: f32,
        initial_epsilon: f32,
        initial_delta: f32,
        inlier_threshold: f32,
        rng: R,
    ) -> Self {
        Self {
            max_candidate_hypothesis,
            block_size,
            max_delta_estimations,
            likelyhood_ratio_threshold: probability_of_rejecting_good_model.recip(),
            epsilon_update_threshold,
            initial_epsilon,
            initial_delta,
            inlier_threshold,
            rng,
        }
    }

    /// Algorithm 3 from "A Comparative Analysis of RANSAC Techniques Leading to Adaptive Real-Time Random Sample Consensus"
    ///
    /// At least at present, this does not use the PROSAC method and instead does completely random sampling.
    ///
    /// Returns the initial models (and their num inliers), `epsilon`, and `delta` in that order.
    fn initial_hypotheses<E>(
        &mut self,
        estimator: &E,
        data: &[EstimatorData<E>],
    ) -> (Vec<(E::Model, usize)>, f32, f32)
    where
        E: Estimator,
    {
        let mut hypotheses = vec![];
        // We don't want more than `block_size` data points to be used to evaluate models initially.
        let initial_datapoints = std::cmp::min(self.block_size, data.len());
        // Set the best inliers to be the floor of what the number of inliers would need to be to be the initial epsilon.
        let mut best_inliers = (self.initial_epsilon * initial_datapoints as f32).floor() as usize;
        // Set the initial epsilon (inlier ratio in good model).
        let mut epsilon = self.initial_epsilon;
        // Set the initial delta (outlier ratio in good model).
        let mut delta = self.initial_delta;
        let mut positive_likelyhood_ratio = delta / epsilon;
        let mut negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);
        let mut current_delta_estimations = 0;
        let mut total_delta_inliers = 0;
        let mut inlier_indices = vec![];
        // Generate the random hypotheses using all the data, not just the evaluation data.
        let random_hypotheses: Vec<_> = (0..self.max_candidate_hypothesis)
            .flat_map(|_| self.generate_random_hypotheses(estimator, data))
            .collect();
        // Iterate through all the randomly generated hypotheses to update epsilon and delta while finding good models.
        for model in random_hypotheses {
            // Check if the model satisfies the ASPRT test on only `inital_datapoints` evaluation data.
            let (pass, inliers) = self.asprt(
                data[..initial_datapoints].iter(),
                &model,
                positive_likelyhood_ratio,
                negative_likelyhood_ratio,
            );
            if pass {
                // If this has the largest support (most inliers) then we update the
                // approximation of epsilon.
                if inliers > best_inliers {
                    best_inliers = inliers;
                    // Update epsilon (this can only increase, since there are more inliers).
                    epsilon = inliers as f32 / data.len() as f32;
                    // Will decrease positive likelyhood ratio.
                    positive_likelyhood_ratio = delta / epsilon;
                    // Will increase negative likelyhood ratio.
                    negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);

                    // Update the inlier indices.
                    inlier_indices = self.inliers(data.iter(), &model);
                }
                hypotheses.push((model, inliers));
            } else if current_delta_estimations < self.max_delta_estimations {
                // We want to add the information about inliers to our estimation of delta.
                // We only do this up to `max_delta_estimations` times to avoid wasting too much time.
                total_delta_inliers += self.count_inliers(data.iter(), &model);
                current_delta_estimations += 1;
                // Update delta.
                delta =
                    total_delta_inliers as f32 / (current_delta_estimations * data.len()) as f32;
                // May change positive likelyhood ratio.
                positive_likelyhood_ratio = delta / epsilon;
                // May change negative likelyhood ratio.
                negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);
            }
        }

        (hypotheses, epsilon, delta)
    }

    /// Generate a random hypothesis
    ///
    /// TODO: This generates totally random hypotheses without accounting for strength of data points.
    /// ARRSAC uses the PROSAC algorithm for sampling in a way that favors stronger data.
    /// This method still works great, but eventually we should support random sampling that accounts
    /// for strength of data points.
    fn generate_random_hypotheses<E>(
        &mut self,
        estimator: &E,
        data: &[EstimatorData<E>],
    ) -> E::ModelIter
    where
        E: Estimator,
    {
        // We can generate no hypotheses if the amout of data is too low.
        if data.len() < E::MIN_SAMPLES {
            panic!("cannot call generate_random_hypotheses without having enough samples");
        }
        let mut random_samples = vec![0; E::MIN_SAMPLES];
        for n in 0..E::MIN_SAMPLES {
            loop {
                let s = self.rng.gen_range(0, data.len());
                if !random_samples[0..n].contains(&s) {
                    random_samples[n] = s;
                    break;
                }
            }
        }
        estimator.estimate(random_samples.iter().map(|&ix| &data[ix]))
    }

    /// Algorithm 1 in "Randomized RANSAC with Sequential Probability Ratio Test".
    ///
    /// This tests if a model is accepted. Returns `true` on accepted and `false` on rejected.
    ///
    /// `inlier_threshold` - The model residual error threshold between inliers and outliers
    /// `positive_likelyhood_ratio` - `δ / ε`
    /// `negative_likelyhood_ratio` - `(1 - δ) / (1 - ε)`
    fn asprt<'a, M: Model>(
        &self,
        data: impl Iterator<Item = &'a M::Data>,
        model: &M,
        positive_likelyhood_ratio: f32,
        negative_likelyhood_ratio: f32,
    ) -> (bool, usize)
    where
        M::Data: 'a,
    {
        let mut likelyhood_ratio = 1.0;
        let mut inliers = 0;
        for data in data {
            likelyhood_ratio *= if model.residual(data) < self.inlier_threshold {
                inliers += 1;
                positive_likelyhood_ratio
            } else {
                negative_likelyhood_ratio
            };

            if likelyhood_ratio > self.likelyhood_ratio_threshold {
                return (false, 0);
            }
        }

        (true, inliers)
    }

    /// This function tells you how many models should be retained to include data item `i`.
    fn num_to_retain(&self, item: usize, remaining: usize) -> usize {
        // TODO: See if there is some way to re-write this to include no floating-point math.
        std::cmp::min(
            remaining / 2,
            (self.max_candidate_hypothesis as f32
                * 2.0f32.powf(-(item as f32) / self.block_size as f32)) as usize,
        )
    }

    /// Determines the number of inliers a model has.
    fn count_inliers<'a, M: Model>(
        &self,
        data: impl Iterator<Item = &'a M::Data>,
        model: &M,
    ) -> usize
    where
        M::Data: 'a,
    {
        data.filter(|data| model.residual(data) < self.inlier_threshold)
            .count()
    }

    /// Gets indices of inliers for a model.
    fn inliers<'a, M: Model>(
        &self,
        data: impl Iterator<Item = &'a M::Data>,
        model: &M,
    ) -> Vec<usize>
    where
        M::Data: 'a,
    {
        data.enumerate()
            .filter(|(ix, data)| model.residual(data) < self.inlier_threshold)
            .map(|(ix, data)| ix)
            .collect()
    }
}

impl<E, R> Consensus<E> for Arrsac<R>
where
    E: Estimator,
    R: Rng,
{
    type Inliers = Vec<usize>;

    fn model(&mut self, estimator: &E, data: &[EstimatorData<E>]) -> Option<E::Model> {
        unimplemented!()
    }

    fn model_inliers(
        &mut self,
        estimator: &E,
        data: &[EstimatorData<E>],
    ) -> Option<(E::Model, Self::Inliers)> {
        // Generate the initial set of hypotheses. This also gets us an estimate of epsilon and delta.
        // We only want to give it one block size of data for the initial generation.
        let (mut hypotheses, mut epsilon, mut delta) = self.initial_hypotheses(estimator, data);
        let inital_num_hypotheses = hypotheses.len();

        // Gradually increase how many datapoints we are evaluating until we evaluate them all.
        for num_data in self.block_size + 1..data.len() {
            // This will retain no more than half of the hypotheses each time
            // and gradually decrease as the number of samples we are evaluating increases.
            let retain = self.num_to_retain(num_data, hypotheses.len());
            // We need to sort the hypotheses based on how good they are (number inliers).
            // The best hypotheses go to the beginning.
            hypotheses.sort_unstable_by_key(|&(_, inliers)| -(inliers as isize));
            hypotheses.resize_with(retain, || {
                panic!("Arrsac::models should never resize the hypotheses to be higher");
            });
            if hypotheses.len() <= 1 {
                break;
            }
            // Score the hypotheses with the new datapoint.
            let new_datapoint = &data[num_data - 1];
            for (hypothesis, inlier_count) in hypotheses.iter_mut() {
                if hypothesis.residual(new_datapoint) < self.inlier_threshold {
                    *inlier_count += 1;
                }
            }
            // Every block size we do this.
            if num_data % self.block_size == 0 {
                // First, update epsilon using the best model.
                // Technically model 0 might no longer be the best model after evaluating the last data-point,
                // but that is not that important.
                epsilon = hypotheses[0].1 as f32 / num_data as f32;
                // Create the likelyhood ratios for inliers and outliers.
                let positive_likelyhood_ratio = delta / epsilon;
                let negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);
                // We generate hypotheses until we reach the max candidate hypotheses.
                if hypotheses.len() < self.max_candidate_hypothesis {}
            }
        }
        hypotheses.into_iter().next().map(|(model, _)| {
            let inliers = self.inliers(data.iter(), &model);
            (model, inliers)
        })
    }
}

pub struct Inliers;

impl Iterator for Inliers {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        unimplemented!()
    }
}
