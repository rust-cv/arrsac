#![feature(type_alias_impl_trait)]

use rand::Rng;
use sample_consensus::{Consensus, Estimator, EstimatorData, Model};

pub struct Arrsac<R> {
    max_iterations: usize,
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
        max_iterations: usize,
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
            max_iterations,
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
    /// Returns the initial models, `epsilon`, and `delta` in that order.
    fn initial_hypotheses<E>(
        &mut self,
        data: &[EstimatorData<E>],
        estimator: &E,
    ) -> (Vec<E::Model>, f32, f32)
    where
        E: Estimator,
    {
        let mut hypotheses = vec![];
        let mut best_inliers = 0;
        let mut epsilon = self.initial_epsilon;
        let mut delta = self.initial_delta;
        let mut positive_likelyhood_ratio = delta / epsilon;
        let mut negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);
        let mut current_delta_estimations = 0;
        let mut total_delta_inliers = 0;
        let mut inlier_indices = vec![];
        let random_hypotheses: Vec<_> = (0..self.max_candidate_hypothesis)
            .flat_map(|_| self.generate_random_hypothesis(data, estimator))
            .collect();
        for model in random_hypotheses {
            // Check if the model satisfies the ASPRT test.
            let (pass, inliers) = self.asprt(
                data.iter(),
                &model,
                positive_likelyhood_ratio,
                negative_likelyhood_ratio,
            );
            if pass {
                // If this has the largest support (most inliers) then we update the
                // approximation of epsilon.
                if inliers > best_inliers {
                    best_inliers = inliers;
                    // Update epsilon.
                    epsilon = inliers as f32 / data.len() as f32;
                    // May decrease positive likelyhood ratio.
                    positive_likelyhood_ratio = delta / epsilon;
                    // May increase negative likelyhood ratio.
                    negative_likelyhood_ratio = (1.0 - delta) / (1.0 - epsilon);

                    // Update the inlier indices.
                    inlier_indices = self.inliers(data.iter(), &model);
                }
                hypotheses.push(model);
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
    fn generate_random_hypothesis<E>(
        &mut self,
        data: &[EstimatorData<E>],
        estimator: &E,
    ) -> Option<E::Model>
    where
        E: Estimator,
    {
        // We can generate no hypotheses if the amout of data is too low.
        if data.len() < E::MIN_SAMPLES {
            return None;
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
