# arrsac

Implements the ARRSAC algorithm from the paper "A Comparative Analysis of RANSAC Techniques Leading to Adaptive Real-Time Random Sample Consensus";
the paper "Randomized RANSAC with Sequential Probability Ratio Test" is also used to implement the SPRT for RANSAC.

This can be used as a `Consensus` algorithm with the [`sample-consensus`](https://crates.io/crates/sample-consensus) crate.
ARRSAC can replace RANSAC and is almost always a faster solution, given that you are willing to tune the parameters.
