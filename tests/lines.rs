use arrsac::{Arrsac, Config};
use nalgebra::Vector2;
use pcg_rand::Pcg64;
use rand::distributions::Uniform;
use rand::{distributions::Distribution, Rng};
use sample_consensus::{Consensus, Estimator, Model};

#[derive(Debug)]
struct Line {
    norm: Vector2<f32>,
    c: f32,
}

impl Model for Line {
    type Data = Vector2<f32>;

    fn residual(&self, point: &Self::Data) -> f32 {
        (self.norm.dot(point) + self.c).abs()
    }
}

struct LineEstimator;

impl Estimator for LineEstimator {
    type Model = Line;
    type ModelIter = std::iter::Once<Line>;
    const MIN_SAMPLES: usize = 2;

    fn estimate<'a, I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = &'a Vector2<f32>> + Clone,
    {
        let a = data.next().unwrap();
        let b = data.next().unwrap();
        let norm = Vector2::new(a.y - b.y, b.x - a.x).normalize();
        let c = -norm.dot(b);
        std::iter::once(Line { norm, c })
    }
}

#[test]
fn lines() {
    let mut rng = Pcg64::new_unseeded();
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(Config::new(3.0), Pcg64::new_unseeded());

    for _ in 0..20_000 {
        // Generate <a, b> and normalize.
        let norm = Vector2::new(rng.gen_range(-10.0, 10.0), rng.gen_range(-10.0, 10.0)).normalize();
        // Get parallel ray.
        let ray = Vector2::new(norm.y, -norm.x);
        // Generate random c.
        let c = rng.gen_range(-10.0, 10.0);

        // Generate random number of points between 50 and 1000.
        let num = rng.gen_range(100, 1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0);
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0);
        // Generate the points.
        let points: Vec<Vector2<f32>> = (0..num)
            .map(|_| {
                let residual: f32 = residuals.sample(&mut rng);
                let distance: f32 = distances.sample(&mut rng);
                let along = ray * distance;
                let against = (residual - c) * norm;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, &points)
            .expect("unable to estimate a model");
        // Check the slope using the cosine distance.
        assert!(
            model.norm.dot(&norm).abs() > 0.9,
            "slope out of expected range"
        );
    }
}
