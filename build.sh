pushd gpu
krnlc
popd
cargo test --release -- test_diamond_partitioning --nocapture

