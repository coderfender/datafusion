// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use datafusion_expr::logical_plan::dml::InsertOp;

/// FFI safe version of [`InsertOp`].
#[repr(u8)]
pub enum FFiInsertOp {
    Append,
    Overwrite,
    Replace,
}

impl From<FFiInsertOp> for InsertOp {
    fn from(value: FFiInsertOp) -> Self {
        match value {
            FFiInsertOp::Append => InsertOp::Append,
            FFiInsertOp::Overwrite => InsertOp::Overwrite,
            FFiInsertOp::Replace => InsertOp::Replace,
        }
    }
}

impl From<InsertOp> for FFiInsertOp {
    fn from(value: InsertOp) -> Self {
        match value {
            InsertOp::Append => FFiInsertOp::Append,
            InsertOp::Overwrite => FFiInsertOp::Overwrite,
            InsertOp::Replace => FFiInsertOp::Replace,
        }
    }
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::dml::InsertOp;

    use super::FFiInsertOp;

    fn test_round_trip_insert_op(insert_op: InsertOp) {
        let ffi_insert_op: FFiInsertOp = insert_op.into();
        let round_trip: InsertOp = ffi_insert_op.into();

        assert_eq!(insert_op, round_trip);
    }

    /// This test ensures we have not accidentally mapped the FFI
    /// enums to the wrong internal enums values.
    #[test]
    fn test_all_round_trip_insert_ops() {
        test_round_trip_insert_op(InsertOp::Append);
        test_round_trip_insert_op(InsertOp::Overwrite);
        test_round_trip_insert_op(InsertOp::Replace);
    }
}
