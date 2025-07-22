use arrsac::Arrsac;
use rand::{
    distr::{Distribution, Uniform},
    Rng, SeedableRng,
};
use rand_xoshiro::Xoshiro256PlusPlus;
use sample_consensus::{Consensus, Estimator, Model};

#[derive(Debug, Clone, Copy)]
struct Vector2<T> {
    x: T,
    y: T,
}

impl Vector2<f64> {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
    fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    fn normalize(&self) -> Self {
        let v_norm = self.norm();
        Self {
            x: self.x / v_norm,
            y: self.y / v_norm,
        }
    }
}

impl core::ops::Mul<Vector2<f64>> for f64 {
    type Output = Vector2<f64>;
    fn mul(self, rhs: Vector2<f64>) -> Self::Output {
        Vector2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl core::ops::Add for Vector2<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    // The max candidate hypotheses had to be increased dramatically to ensure all 1000 cases find a
    // good-fitting line.
    let mut arrsac = Arrsac::new(3.0, rng.clone());

    for _ in 0..2000 {
        // Generate <a, b> and normalize.
        let norm =
            Vector2::new(rng.random_range(-10.0..10.0), rng.random_range(-10.0..10.0)).normalize();
        // Get parallel ray.
        let ray = Vector2::new(norm.y, -norm.x);
        // Generate random c.
        let c = rng.random_range(-10.0..10.0);

        // Generate random number of points between 50 and 1000.
        let num = rng.random_range(50..1000);
        // The points should be no more than 5.0 away from the line and be evenly distributed away from the line.
        let residuals = Uniform::new(-5.0, 5.0).unwrap();
        // The points must be generated along the line, but the distance should be bounded to make it more difficult.
        let distances = Uniform::new(-50.0, 50.0).unwrap();
        // Generate the points.
        let points: Vec<Vector2<f64>> = (0..num)
            .map(|_| {
                let residual: f64 = residuals.sample(&mut rng);
                let distance: f64 = distances.sample(&mut rng);
                let along = distance * ray;
                let against = (residual - c) * norm;
                along + against
            })
            .collect();

        let model = arrsac
            .model(&LineEstimator, points.iter().copied())
            .expect("unable to estimate a model");
        // Check the slope using the cosine distance.
        assert!(
            model.norm.dot(&norm).abs() > 0.99,
            "slope out of expected range"
        );
    }
}
