use FLIP::flip::{Flip, WorldProperties};
use criterion::{black_box, criterion_group, criterion_main, Criterion};


fn bench_fluid_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("FLIP_Fluid_Update");

    // Test with a couple of different grid sizes.
    // 32x32 and 64x64 are good baselines. (Ensure they are > 8 to spawn particles)
    let grid_sizes = [32, 64, 128, 256];

    for &size in grid_sizes.iter() {
        group.bench_with_input(
            criterion::BenchmarkId::from_parameter(format!("{}x{}", size, size)),
            &size,
            |b, &size| {
                let props = WorldProperties {
                    gravity: -9.81,
                    border_damping: 0.1,
                    density: 1.0,
                };
                
                // Initialize the fluid simulation
                let mut flip = Flip::new(size, size, 1.0, props);

                // WARMUP: 
                // A fluid sim at rest (frame 0) solves almost instantly because velocities 
                // and pressures are 0. We step the simulation forward a few frames so 
                // gravity accelerates the particles and the PCG solver actually has to work.
                for _ in 0..15 {
                    flip.update(0.016); // 60 FPS delta time
                }

                // Run the actual benchmark iteration
                b.iter(|| {
                    // black_box prevents the compiler from overly optimizing the loop
                    flip.update(black_box(0.016));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fluid_update);
criterion_main!(benches);