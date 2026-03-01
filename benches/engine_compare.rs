use std::{rc::Rc, sync::Arc};

use arrow::{
    array::{ArrayRef, Int64Array, RecordBatch, StringArray},
    compute::{concat_batches, lexsort_to_indices, take, SortColumn},
    datatypes::{DataType, Field, Schema},
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use how_query_engines_work_rust::{
    data_frame::ExecutionContext,
    data_source::DataSource,
    jit::compile,
    logical_exprs::{aggregate::AggregateExpr, LogicalExpr},
    logical_plans::LogicalPlan,
    optimizer::optimize,
    planner::create_physical_plan,
};

#[derive(Clone, Copy)]
enum Engine {
    Volcano,
    Jit,
}

impl Engine {
    fn as_str(self) -> &'static str {
        match self {
            Engine::Volcano => "volcano",
            Engine::Jit => "jit",
        }
    }
}

struct Scenario {
    name: &'static str,
    build_plan: fn() -> LogicalPlan,
    estimated_rows: usize,
}

struct SyntheticBenchmarkDataSource {
    schema: Schema,
    batches: Vec<RecordBatch>,
}

impl SyntheticBenchmarkDataSource {
    fn new(total_rows: usize, batch_size: usize) -> Self {
        const FIRST: [&str; 8] = [
            "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi",
        ];
        const LAST: [&str; 8] = [
            "Smith", "Johnson", "Brown", "Williams", "Jones", "Miller", "Davis", "Taylor",
        ];

        let schema = Schema::new(vec![
            Field::new("first_name", DataType::Utf8, false),
            Field::new("last_name", DataType::Utf8, false),
            Field::new("age", DataType::Int64, false),
        ]);

        let mut batches = Vec::new();
        let mut row = 0usize;
        while row < total_rows {
            let len = (total_rows - row).min(batch_size);

            let first_name = (0..len)
                .map(|i| FIRST[(row + i) % FIRST.len()])
                .collect::<Vec<_>>();
            let last_name = (0..len)
                .map(|i| LAST[(row + i) % LAST.len()])
                .collect::<Vec<_>>();
            let age = (0..len)
                .map(|i| (((row + i) as i64 * 17 + 11) % 97) + 18)
                .collect::<Vec<_>>();

            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![
                    Arc::new(StringArray::from(first_name)),
                    Arc::new(StringArray::from(last_name)),
                    Arc::new(Int64Array::from(age)),
                ],
            )
            .unwrap();
            batches.push(batch);
            row += len;
        }

        Self { schema, batches }
    }
}

impl DataSource for SyntheticBenchmarkDataSource {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn scan(&self, projection: Vec<String>) -> Vec<RecordBatch> {
        if projection.is_empty() {
            return self.batches.clone();
        }

        let mut indices = Vec::with_capacity(projection.len());
        for column in projection {
            if let Ok(index) = self.schema.index_of(&column) {
                indices.push(index);
            }
        }

        self.batches
            .iter()
            .map(|batch| batch.project(&indices).unwrap())
            .collect()
    }
}

fn normalize_output(mut batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
    if batches.is_empty() {
        return vec![];
    }

    let schema = batches[0].schema();
    let merged = concat_batches(&schema, &batches).unwrap();
    if merged.num_rows() <= 1 {
        return vec![merged];
    }

    let sort_columns = merged
        .columns()
        .iter()
        .cloned()
        .map(|values| SortColumn {
            values,
            options: None,
        })
        .collect::<Vec<_>>();
    let indices = lexsort_to_indices(&sort_columns, None).unwrap();
    let sorted_columns = merged
        .columns()
        .iter()
        .map(|column| take(column.as_ref(), &indices, None).unwrap())
        .collect::<Vec<ArrayRef>>();
    batches.clear();
    vec![RecordBatch::try_new(schema, sorted_columns).unwrap()]
}

fn assert_parity(
    name: &str,
    volcano_out: Vec<RecordBatch>,
    other_out: Vec<RecordBatch>,
    engine: Engine,
) {
    let volcano_out = normalize_output(volcano_out);
    let other_out = normalize_output(other_out);

    if !volcano_out.is_empty() && !other_out.is_empty() {
        assert_eq!(
            volcano_out[0].schema(),
            other_out[0].schema(),
            "schema mismatch for scenario '{name}' against {}",
            engine.as_str()
        );
    } else {
        assert_eq!(
            volcano_out.is_empty(),
            other_out.is_empty(),
            "empty/non-empty mismatch for scenario '{name}' against {}",
            engine.as_str()
        );
    }

    assert_eq!(
        volcano_out,
        other_out,
        "output mismatch for scenario '{name}' against {}",
        engine.as_str()
    );
}

fn plan_numeric_eq() -> LogicalPlan {
    let age = LogicalExpr::col("age");
    ExecutionContext::csv("employee.csv")
        .filter(age.clone().eq(LogicalExpr::lit_long(42)))
        .project(vec![age])
        .logical_plan()
}

fn plan_mixed_string() -> LogicalPlan {
    let first_name = LogicalExpr::col("first_name");
    let last_name = LogicalExpr::col("last_name");
    let age = LogicalExpr::col("age");
    ExecutionContext::csv("employee.csv")
        .filter(first_name.clone().neq(last_name))
        .project(vec![first_name, age])
        .logical_plan()
}

fn plan_aggregate_grouped() -> LogicalPlan {
    let first_name = LogicalExpr::col("first_name");
    let age = LogicalExpr::col("age");
    ExecutionContext::csv("employee.csv")
        .aggregate(
            vec![first_name],
            vec![
                AggregateExpr::count(age.clone()),
                AggregateExpr::max(age.clone()),
                AggregateExpr::avg(age),
            ],
        )
        .logical_plan()
}

fn plan_full_pipeline_all_nodes() -> LogicalPlan {
    let first_name = LogicalExpr::col("first_name");
    let last_name = LogicalExpr::col("last_name");
    let age = LogicalExpr::col("age");
    ExecutionContext::csv("employee.csv")
        .filter(
            first_name
                .clone()
                .neq(last_name)
                .and(age.clone().gte(LogicalExpr::lit_long(20))),
        )
        .project(vec![first_name.clone(), age.clone()])
        .aggregate(
            vec![first_name],
            vec![AggregateExpr::count(age.clone()), AggregateExpr::avg(age)],
        )
        .logical_plan()
}

fn plan_deep_predicate_synthetic() -> LogicalPlan {
    let scan = LogicalPlan::scan(
        "synthetic://deep_predicate".to_string(),
        Rc::new(SyntheticBenchmarkDataSource::new(4_194_304, 131_072)),
        vec![],
    );

    let first = LogicalExpr::col("first_name");
    let last = LogicalExpr::col("last_name");
    let age = LogicalExpr::col("age");

    let predicate = age
        .clone()
        .gt(LogicalExpr::lit_long(25))
        .and(age.clone().lt(LogicalExpr::lit_long(85)))
        .and(age.clone().neq(LogicalExpr::lit_long(29)))
        .and(age.clone().neq(LogicalExpr::lit_long(31)))
        .and(age.clone().neq(LogicalExpr::lit_long(37)))
        .and(age.clone().neq(LogicalExpr::lit_long(41)))
        .and(age.clone().neq(LogicalExpr::lit_long(43)))
        .and(age.clone().neq(LogicalExpr::lit_long(47)))
        .and(age.clone().neq(LogicalExpr::lit_long(53)))
        .and(age.clone().neq(LogicalExpr::lit_long(59)))
        .and(age.clone().neq(LogicalExpr::lit_long(61)))
        .and(age.clone().neq(LogicalExpr::lit_long(67)))
        .and(age.clone().neq(LogicalExpr::lit_long(71)))
        .and(age.clone().neq(LogicalExpr::lit_long(73)))
        .and(
            first
                .clone()
                .eq(LogicalExpr::lit_str("Alice"))
                .or(first.clone().eq(LogicalExpr::lit_str("Bob")))
                .or(first.clone().eq(LogicalExpr::lit_str("Charlie"))),
        )
        .and(last.clone().neq(LogicalExpr::lit_str("Brown")))
        .and(last.clone().neq(LogicalExpr::lit_str("Davis")))
        .and(last.clone().neq(LogicalExpr::lit_str("Miller")))
        .and(first.clone().neq(last.clone()))
        .and(first.clone().gt(LogicalExpr::lit_str("A")))
        .and(last.clone().lt(LogicalExpr::lit_str("zzzz")));

    LogicalPlan::projection(
        LogicalPlan::filter(scan, predicate),
        vec![first.clone(), last, age],
    )
}

fn plan_large_inmem_m() -> LogicalPlan {
    let scan = LogicalPlan::scan(
        "synthetic://large_inmem_m".to_string(),
        Rc::new(SyntheticBenchmarkDataSource::new(2_097_152, 131_072)),
        vec![],
    );

    let first_name = LogicalExpr::col("first_name");
    let last_name = LogicalExpr::col("last_name");
    let age = LogicalExpr::col("age");

    let filtered = LogicalPlan::filter(
        scan,
        age.clone()
            .gt(LogicalExpr::lit_long(30))
            .and(age.clone().lt(LogicalExpr::lit_long(80)))
            .and(first_name.clone().neq(last_name.clone())),
    );
    let projected = LogicalPlan::projection(filtered, vec![first_name.clone(), age.clone()]);
    LogicalPlan::aggregate(
        projected,
        vec![first_name],
        vec![
            AggregateExpr::count(age.clone()),
            AggregateExpr::sum(age.clone()),
            AggregateExpr::avg(age),
        ],
    )
}

fn plan_large_inmem_l() -> LogicalPlan {
    let scan = LogicalPlan::scan(
        "synthetic://large_inmem_l".to_string(),
        Rc::new(SyntheticBenchmarkDataSource::new(4_194_304, 131_072)),
        vec![],
    );

    let first_name = LogicalExpr::col("first_name");
    let last_name = LogicalExpr::col("last_name");
    let age = LogicalExpr::col("age");

    let filtered = LogicalPlan::filter(
        scan,
        age.clone()
            .gt(LogicalExpr::lit_long(30))
            .and(age.clone().lt(LogicalExpr::lit_long(80)))
            .and(first_name.clone().neq(last_name.clone())),
    );
    let projected = LogicalPlan::projection(filtered, vec![first_name.clone(), age.clone()]);
    LogicalPlan::aggregate(
        projected,
        vec![first_name],
        vec![
            AggregateExpr::count(age.clone()),
            AggregateExpr::sum(age.clone()),
            AggregateExpr::avg(age),
        ],
    )
}

fn bench_scenario(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    scenario: &Scenario,
) {
    let logical_plan = optimize((scenario.build_plan)());

    let mut volcano = create_physical_plan(&logical_plan);
    let mut jit = compile(Arc::new(create_physical_plan(&logical_plan))).unwrap();

    let volcano_out = volcano.execute();
    assert_parity(scenario.name, volcano_out, jit.execute(), Engine::Jit);

    group.throughput(Throughput::Elements(scenario.estimated_rows as u64));

    group.bench_function(
        BenchmarkId::new(scenario.name, Engine::Volcano.as_str()),
        |b| {
            b.iter(|| {
                let _ = volcano.execute();
            })
        },
    );
    group.bench_function(BenchmarkId::new(scenario.name, Engine::Jit.as_str()), |b| {
        b.iter(|| {
            let _ = jit.execute();
        })
    });
}

fn bench_engine_compare(c: &mut Criterion) {
    let core_scenarios = [
        Scenario {
            name: "numeric_eq",
            build_plan: plan_numeric_eq,
            estimated_rows: 20_000,
        },
        Scenario {
            name: "mixed_string",
            build_plan: plan_mixed_string,
            estimated_rows: 20_000,
        },
        Scenario {
            name: "aggregate_grouped",
            build_plan: plan_aggregate_grouped,
            estimated_rows: 20_000,
        },
        Scenario {
            name: "full_pipeline_all_nodes",
            build_plan: plan_full_pipeline_all_nodes,
            estimated_rows: 20_000,
        },
        Scenario {
            name: "deep_predicate_synthetic",
            build_plan: plan_deep_predicate_synthetic,
            estimated_rows: 4_194_304,
        },
    ];
    let large_scenarios = [
        Scenario {
            name: "large_inmem_m",
            build_plan: plan_large_inmem_m,
            estimated_rows: 2_097_152,
        },
        Scenario {
            name: "large_inmem_l",
            build_plan: plan_large_inmem_l,
            estimated_rows: 4_194_304,
        },
    ];

    let mut core_group = c.benchmark_group("engine_compare_core");
    for scenario in &core_scenarios {
        bench_scenario(&mut core_group, scenario);
    }
    core_group.finish();

    let mut large_group = c.benchmark_group("engine_compare_large_inmem");
    for scenario in &large_scenarios {
        bench_scenario(&mut large_group, scenario);
    }
    large_group.finish();
}

criterion_group!(benches, bench_engine_compare);
criterion_main!(benches);
