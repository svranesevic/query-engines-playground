use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use how_query_engines_work_rust::data_frame::ExecutionContext;

use how_query_engines_work_rust::jit;
use how_query_engines_work_rust::logical_exprs::{aggregate::AggregateExpr, LogicalExpr};
use how_query_engines_work_rust::optimizer::optimize;
use how_query_engines_work_rust::planner::create_physical_plan;

fn main() {
    let ctx = ExecutionContext::csv("employee.csv");

    let first_name_col = LogicalExpr::col("first_name");
    let age_col = LogicalExpr::col("age");

    let count = AggregateExpr::count(LogicalExpr::lit_long(42));

    let age = LogicalExpr::lit_long(20);
    let filter = age_col.gte(age);
    let logical_plan = ctx
        .filter(filter)
        .project(vec![first_name_col, age_col.clone()])
        .aggregate(vec![age_col], vec![count])
        .logical_plan();
    println!("Logical Plan:\n{}", logical_plan.format());

    let logical_plan = optimize(logical_plan);
    println!();
    println!("Optimized Logical Plan:\n{}", logical_plan.format());

    let mut physical_plan = create_physical_plan(&logical_plan);
    println!();
    println!("Physical Plan:\n{}", physical_plan.format());

    let results = physical_plan.execute();
    let formatted = pretty_format_batches(&results).unwrap();
    println!();
    println!("Volcano results:\n{}", formatted);

    let mut plan = jit::compile(Arc::new(create_physical_plan(&logical_plan)))
        .expect("JIT compilation failed");
    let results = plan.execute();
    let formatted = pretty_format_batches(&results).unwrap();
    println!();
    println!("JIT Results:\n{}", formatted);
}
