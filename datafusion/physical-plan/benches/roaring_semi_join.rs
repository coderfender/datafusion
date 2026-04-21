use criterion::{Criterion, black_box, criterion_group, criterion_main};
use roaring::RoaringBitmap;
use std::collections::HashSet;

fn bench_semi_join(c: &mut Criterion) {
    let build_size = 100_000u32;
    let probe_size = 1_000_000;

    let build_keys: Vec<u32> = (0..build_size).collect();
    let probe_keys: Vec<u32> = (0..probe_size)
        .map(|i| (i as u32 * 7) % (build_size * 2))
        .collect();

    let hashset: HashSet<u32> = build_keys.iter().copied().collect();
    let roaring: RoaringBitmap = build_keys.iter().copied().collect();

    c.bench_function("hashset_probe_100k", |b| {
        b.iter(|| {
            let mut hits = 0u64;
            for &key in &probe_keys {
                if hashset.contains(&key) {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });

    c.bench_function("roaring_probe_100k", |b| {
        b.iter(|| {
            let mut hits = 0u64;
            for &key in &probe_keys {
                if roaring.contains(key) {
                    hits += 1;
                }
            }
            black_box(hits)
        })
    });
}

criterion_group!(benches, bench_semi_join);
criterion_main!(benches);
