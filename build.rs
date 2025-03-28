use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shader_crate = "tsdistances_gpu";
    let target = "spirv-unknown-spv1.5";

    SpirvBuilder::new(shader_crate, target)
        .shader_crate_default_features(false)
        .print_metadata(MetadataPrintout::Full)
        .build()?;
    Ok(())
}
