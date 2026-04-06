use datafusion::prelude::*;
use datafusion_common::Result;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();
    let sql = "SELECT * FROM (SELECT null AS id1) t1 INNER JOIN (SELECT null AS id2) t2 ON id1 = id2";
    let df = ctx.sql(sql).await?;
    df.show().await?;
    Ok(())
}
