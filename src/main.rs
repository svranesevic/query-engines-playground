#![allow(dead_code)]

mod data_frame;
mod data_source;
mod logical_exprs;
mod logical_plans;
mod optimizer;
mod physical_exprs;
mod physical_plans;
mod planner;

use data_frame::ExecutionContext;
use logical_exprs::{aggregate::AggregateExpr, LogicalExpr};
use optimizer::optimize;
use planner::create_physical_plan;

fn main() {
    let ctx = ExecutionContext::csv("employee.csv");

    let first_name_col = LogicalExpr::col("first_name");
    let age_col = LogicalExpr::col("age");

    let count = AggregateExpr::count(LogicalExpr::lit_long(42));

    let age = LogicalExpr::lit_long(42);
    let filter = age_col.gte(age);
    let logical_plan = ctx
        .filter(filter)
        .project(vec![first_name_col, age_col.clone()])
        .aggregate(vec![age_col], vec![count])
        .logical_plan();
    println!("Logical Plan:\n{}", logical_plan.format());

    println!();

    let logical_plan = optimize(logical_plan);
    println!("Optimized Logical Plan:\n{}", logical_plan.format());

    println!();

    let physical_plan = create_physical_plan(&logical_plan);
    println!("Physical Plan:\n{}", physical_plan.format());
}
