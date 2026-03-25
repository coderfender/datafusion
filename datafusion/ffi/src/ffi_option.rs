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

//! FFI-safe Option and Result types that do not require `IStable` bounds.
//!
//! stabby's `Option<T>` and `Result<T, E>` require `T: IStable` for niche
//! optimization. Many of our FFI structs contain self-referential function
//! pointers and cannot implement `IStable`. These simple `#[repr(C)]` types
//! provide the same FFI-safe semantics without that constraint.

/// An FFI-safe option type.
#[repr(C, u8)]
#[derive(Debug, Clone)]
pub enum FFI_Option<T> {
    Some(T),
    None,
}

impl<T> From<Option<T>> for FFI_Option<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => FFI_Option::Some(v),
            None => FFI_Option::None,
        }
    }
}

impl<T> From<FFI_Option<T>> for Option<T> {
    fn from(opt: FFI_Option<T>) -> Self {
        match opt {
            FFI_Option::Some(v) => Some(v),
            FFI_Option::None => None,
        }
    }
}

impl<T> FFI_Option<T> {
    pub fn as_ref(&self) -> Option<&T> {
        match self {
            FFI_Option::Some(v) => Some(v),
            FFI_Option::None => None,
        }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> FFI_Option<U> {
        match self {
            FFI_Option::Some(v) => FFI_Option::Some(f(v)),
            FFI_Option::None => FFI_Option::None,
        }
    }

    pub fn into_option(self) -> Option<T> {
        self.into()
    }
}

/// An FFI-safe result type.
#[repr(C, u8)]
#[derive(Debug, Clone)]
pub enum FFI_Result<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> From<Result<T, E>> for FFI_Result<T, E> {
    fn from(res: Result<T, E>) -> Self {
        match res {
            Ok(v) => FFI_Result::Ok(v),
            Err(e) => FFI_Result::Err(e),
        }
    }
}

impl<T, E> From<FFI_Result<T, E>> for Result<T, E> {
    fn from(res: FFI_Result<T, E>) -> Self {
        match res {
            FFI_Result::Ok(v) => Ok(v),
            FFI_Result::Err(e) => Err(e),
        }
    }
}

impl<T, E> FFI_Result<T, E> {
    pub fn is_ok(&self) -> bool {
        matches!(self, FFI_Result::Ok(_))
    }

    pub fn is_err(&self) -> bool {
        matches!(self, FFI_Result::Err(_))
    }

    pub fn unwrap_err(self) -> E {
        match self {
            FFI_Result::Err(e) => e,
            FFI_Result::Ok(_) => panic!("called unwrap_err on Ok"),
        }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> FFI_Result<U, E> {
        match self {
            FFI_Result::Ok(v) => FFI_Result::Ok(f(v)),
            FFI_Result::Err(e) => FFI_Result::Err(e),
        }
    }

    pub fn into_result(self) -> Result<T, E> {
        self.into()
    }
}

impl<T: PartialEq, E: PartialEq> PartialEq for FFI_Result<T, E> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (FFI_Result::Ok(a), FFI_Result::Ok(b)) => a == b,
            (FFI_Result::Err(a), FFI_Result::Err(b)) => a == b,
            _ => false,
        }
    }
}
