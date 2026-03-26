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

use std::ffi::c_void;
use std::sync::Arc;
use stabby::stabby;
use stabby::string::String as SString;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use crate::config::FFI_ConfigOptions;
use crate::execution_plan::FFI_ExecutionPlan;
use crate::util::FFIResult;

#[repr(C)]
pub struct FFI_PhysicalOptimizerRule {

    /// FFI equivalent to the `name` of a [`PhysicalOptimizerRule`]
    pub name: unsafe extern "C" fn(rule : &Self) -> SString,
    pub schema_check: unsafe extern "C" fn(rule : &Self) -> bool,
    pub optimize: unsafe extern "C" fn (
        rule: &Self,
        plan: FFI_ExecutionPlan,
        config: FFI_ConfigOptions,
    ) -> FFIResult<FFI_ExecutionPlan>,

    pub clone: unsafe extern "C" fn(rule: &Self) -> Self,
    pub release: unsafe extern "C" fn(rule: &mut Self),
    pub private_data: *mut c_void,

}

unsafe impl Send for FFI_PhysicalOptimizerRule {}
unsafe impl Sync for FFI_PhysicalOptimizerRule {}

struct PhysicalOptimizerRulePrivateData {
    rule: Arc<dyn PhysicalOptimizerRule + Send + Sync>,
}
