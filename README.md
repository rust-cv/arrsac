# arrsac

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] [![docs.rs][di]][dl] ![LoC][lo] ![ci][bci]

[ci]: https://img.shields.io/crates/v/arrsac.svg
[cl]: https://crates.io/crates/arrsac/

[di]: https://docs.rs/arrsac/badge.svg
[dl]: https://docs.rs/arrsac/

[lo]: https://tokei.rs/b1/github/rust-cv/arrsac?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[bci]: https://github.com/rust-cv/arrsac/workflows/ci/badge.svg

Implements the ARRSAC algorithm from the paper "A Comparative Analysis of RANSAC Techniques Leading to Adaptive Real-Time Random Sample Consensus";
the paper "Randomized RANSAC with Sequential Probability Ratio Test" is also used to implement the SPRT for RANSAC.

Some things were modified from the original papers which improve corner cases or convenience in regular usage.

This can be used as a `Consensus` algorithm with the [`sample-consensus`](https://crates.io/crates/sample-consensus) crate.
ARRSAC can replace RANSAC and is almost always a faster solution, given that you are willing to tune the parameters.
