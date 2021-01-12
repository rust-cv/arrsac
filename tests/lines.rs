use arrsac::Arrsac;
use nalgebra::Vector2;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng, SeedableRng};
use rand_pcg::Pcg64;
use sample_consensus::{Consensus, Estimator, Model};

#[derive(Debug)]
struct Line {
    norm: Vector2<f64>,
    c: f64,
}

impl Model<Vector2<f64>> for Line {
    fn residual(&self, point: &Vector2<f64>) -> f64 {
        (self.norm.dot(point) + self.c).abs()
    }
}

struct LineEstimator;

impl Estimator<Vector2<f64>> for LineEstimator {
    type Model = Line;
    type ModelIter = std::iter::Once<Line>;
    const MIN_SAMPLES: usize = 2;

    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = Vector2<f64>> + Clone,
    {
        let a = data.next().unwrap();
        let b = data.next().unwrap();
        let norm = Vector2::new(a.y - b.y, b.x - a.x).normalize();
        let c = -norm.dot(&b);
        std::iter::once(Line { norm, c })
    }
}

#[test]
fn lines() {
    let mut rng = Pcg64::from_seed([7; 32]);
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(3.0, Pcg64::from_seed([7; 32]));

    for _ in 0..2000 {
        // Generate <a, b> and normalize.
        let norm = Vector2::new(rng.gen_range(-10.0..10.0), rng.gen_range(-10.0..10.0)).normalize();
        // Get parallel ray.
        let ray = Vector2::new(norm.y, -norm.x);
        // Generate random c.
        let c = rng.gen_range(-10.0..10.0);

        // Generate random number of points between 50 and 1000.
        let num = rng.gen_range(100..1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0);
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0);
        // Generate the points.
        let points: Vec<Vector2<f64>> = (0..num)
            .map(|_| {
                let residual: f64 = residuals.sample(&mut rng);
                let distance: f64 = distances.sample(&mut rng);
                let along = ray * distance;
                let against = (residual - c) * norm;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, points.iter().copied())
            .expect("unable to estimate a model");
        // Check the slope using the cosine distance.
        assert!(
            model.norm.dot(&norm).abs() > 0.9,
            "slope out of expected range"
        );
    }
}
