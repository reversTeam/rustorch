//! Benchmark for the `GET /runs` listing path. Plan P2.2 sets the
//! goal at `<100ms p99` for a 1000-run table; this bench measures
//! the DB-side cost (where the time actually lives) so we can track
//! regressions independent of the HTTP layer.

use criterion::{criterion_group, criterion_main, Criterion};
use rustorch_console_server::db;
use serde_json::json;
use tokio::runtime::Runtime;

fn bench_list_1000(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let pool = rt.block_on(async {
        let pool = db::connect(":memory:").await.unwrap();
        for i in 0..1000 {
            db::insert_run(
                &pool,
                db::NewRun {
                    title: Some(format!("r{i}")),
                    cfg_json: json!({"lr": 1e-3, "i": i}),
                    sweep_id: None,
                },
            )
            .await
            .unwrap();
        }
        pool
    });

    c.bench_function("list_runs_1000_no_filter", |b| {
        b.iter(|| {
            rt.block_on(async {
                let rows = db::list_runs(&pool, None, 1000, 0).await.unwrap();
                criterion::black_box(rows);
            });
        });
    });

    c.bench_function("list_runs_1000_status_filter", |b| {
        b.iter(|| {
            rt.block_on(async {
                let rows = db::list_runs(&pool, Some(db::RunStatus::Queued), 100, 0)
                    .await
                    .unwrap();
                criterion::black_box(rows);
            });
        });
    });
}

criterion_group!(benches, bench_list_1000);
criterion_main!(benches);
