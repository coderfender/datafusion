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

use arrow::array::{
    ArrayRef, AsArray, BooleanArray, Int64Array, ListArray, PrimitiveArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{ArrowPrimitiveType, Field};
use datafusion_common::HashSet;
use datafusion_common::hash_utils::RandomState;
use datafusion_expr_common::groups_accumulator::{EmitTo, GroupsAccumulator};
use std::hash::Hash;
use std::mem::size_of;
use std::sync::Arc;

use crate::aggregate::groups_accumulator::accumulate::accumulate;

pub struct PrimitiveDistinctCountGroupsAccumulator<T: ArrowPrimitiveType>
where
    T::Native: Eq + Hash,
{
    seen: HashSet<(usize, T::Native), RandomState>,
    counts: Vec<i64>,
}

impl<T: ArrowPrimitiveType> PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            seen: HashSet::default(),
            counts: Vec::new(),
        }
    }
}

impl<T: ArrowPrimitiveType> Default for PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T: ArrowPrimitiveType + Send + std::fmt::Debug> GroupsAccumulator
    for PrimitiveDistinctCountGroupsAccumulator<T>
where
    T::Native: Eq + Hash,
{
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> datafusion_common::Result<()> {
        debug_assert_eq!(values.len(), 1);
        self.counts.resize(total_num_groups, 0);
        let arr = values[0].as_primitive::<T>();
        accumulate(group_indices, arr, opt_filter, |group_idx, value| {
            if self.seen.insert((group_idx, value)) {
                self.counts[group_idx] += 1;
            }
        });
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> datafusion_common::Result<ArrayRef> {
        let counts = emit_to.take_needed(&mut self.counts);

        match emit_to {
            EmitTo::All => {
                self.seen.clear();
            }
            EmitTo::First(n) => {
                let mut remaining = HashSet::default();
                for (group_idx, value) in self.seen.drain() {
                    if group_idx >= n {
                        remaining.insert((group_idx - n, value));
                    }
                }
                self.seen = remaining;
            }
        }

        Ok(Arc::new(Int64Array::from(counts)))
    }

    fn state(&mut self, emit_to: EmitTo) -> datafusion_common::Result<Vec<ArrayRef>> {
        let num_emitted = match emit_to {
            EmitTo::All => self.counts.len(),
            EmitTo::First(n) => n,
        };

        let mut group_values: Vec<Vec<T::Native>> = vec![Vec::new(); num_emitted];

        if matches!(emit_to, EmitTo::All) {
            for (group_idx, value) in self.seen.drain() {
                group_values[group_idx].push(value);
            }
            self.counts.clear();
        } else {
            let mut remaining = HashSet::default();
            for (group_idx, value) in self.seen.drain() {
                if group_idx < num_emitted {
                    group_values[group_idx].push(value);
                } else {
                    remaining.insert((group_idx - num_emitted, value));
                }
            }
            self.seen = remaining;
            let _ = emit_to.take_needed(&mut self.counts);
        }

        let mut offsets = vec![0i32];
        let mut all_values = Vec::new();
        for values in &group_values {
            all_values.extend(values.iter().copied());
            offsets.push(all_values.len() as i32);
        }

        let values_array = Arc::new(PrimitiveArray::<T>::from_iter_values(all_values));
        let list_array = ListArray::new(
            Arc::new(Field::new_list_field(T::DATA_TYPE, true)),
            OffsetBuffer::new(offsets.into()),
            values_array,
            None,
        );

        Ok(vec![Arc::new(list_array)])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> datafusion_common::Result<()> {
        debug_assert_eq!(values.len(), 1);
        self.counts.resize(total_num_groups, 0);
        let list_array = values[0].as_list::<i32>();

        for (row_idx, &group_idx) in group_indices.iter().enumerate() {
            let inner = list_array.value(row_idx);
            let inner_arr = inner.as_primitive::<T>();
            for &value in inner_arr.values().iter() {
                if self.seen.insert((group_idx, value)) {
                    self.counts[group_idx] += 1;
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        size_of::<Self>()
            + self.seen.capacity() * (size_of::<(usize, T::Native)>() + size_of::<u64>())
            + self.counts.capacity() * size_of::<i64>()
    }
}
