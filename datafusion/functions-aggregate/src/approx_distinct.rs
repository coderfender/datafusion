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

//! Defines physical expressions that can evaluated at runtime during query execution

use crate::hyperloglog::{HLL_HASH_STATE, HyperLogLog};
use arrow::array::{Array, AsArray, BinaryArray, BooleanArray, StringViewArray};
use arrow::array::{
    GenericBinaryArray, GenericStringArray, OffsetSizeTrait, PrimitiveArray,
};
use arrow::datatypes::{
    ArrowPrimitiveType, Date32Type, Date64Type, FieldRef, Int8Type, Int16Type, Int32Type,
    Int64Type, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow::{array::ArrayRef, datatypes::DataType, datatypes::Field};
use datafusion_common::ScalarValue;
use datafusion_common::{
    DataFusionError, Result, downcast_value, internal_datafusion_err, internal_err,
    not_impl_err,
};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Documentation, Signature, Volatility,
};
use datafusion_functions_aggregate_common::noop_accumulator::NoopAccumulator;
use datafusion_macros::user_doc;
use std::fmt::{Debug, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

make_udaf_expr_and_func!(
    ApproxDistinct,
    approx_distinct,
    expression,
    "approximate number of distinct input values",
    approx_distinct_udaf
);

impl<T: Hash + ?Sized> From<&HyperLogLog<T>> for ScalarValue {
    fn from(v: &HyperLogLog<T>) -> ScalarValue {
        let values = v.as_ref().to_vec();
        ScalarValue::Binary(Some(values))
    }
}

impl<T: Hash + ?Sized> TryFrom<&[u8]> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &[u8]) -> Result<HyperLogLog<T>> {
        let arr: [u8; 16384] = v.try_into().map_err(|_| {
            internal_datafusion_err!("Impossibly got invalid binary array from states")
        })?;
        Ok(HyperLogLog::<T>::new_with_registers(arr))
    }
}

impl<T: Hash + ?Sized> TryFrom<&ScalarValue> for HyperLogLog<T> {
    type Error = DataFusionError;
    fn try_from(v: &ScalarValue) -> Result<HyperLogLog<T>> {
        if let ScalarValue::Binary(Some(slice)) = v {
            slice.as_slice().try_into()
        } else {
            internal_err!(
                "Impossibly got invalid scalar value while converting to HyperLogLog"
            )
        }
    }
}

#[derive(Debug)]
struct NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    hll: HyperLogLog<T::Native>,
}

impl<T> NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType,
    T::Native: Hash,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<str>,
    phantom_data: PhantomData<T>,
}

impl<T> StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

#[derive(Debug)]
struct StringViewHLLAccumulator {
    hll: HyperLogLog<str>,
}

impl StringViewHLLAccumulator {
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
        }
    }
}

#[derive(Debug)]
struct BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    hll: HyperLogLog<[u8]>,
    phantom_data: PhantomData<T>,
}

impl<T> BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    pub fn new() -> Self {
        Self {
            hll: HyperLogLog::new(),
            phantom_data: PhantomData,
        }
    }
}

#[derive(Debug)]
struct BoolDistinctAccumulator {
    seen_true: bool,
    seen_false: bool,
}

impl BoolDistinctAccumulator {
    fn new() -> Self {
        Self {
            seen_true: false,
            seen_false: false,
        }
    }
}

impl Accumulator for BoolDistinctAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &BooleanArray = downcast_value!(values[0], BooleanArray);
        for value in array.iter().flatten() {
            if value {
                self.seen_true = true;
            } else {
                self.seen_false = true;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let count = (self.seen_true as u64) + (self.seen_false as u64);
        Ok(ScalarValue::UInt64(Some(count)))
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Pack into 1 byte: bit 0 = seen_false, bit 1 = seen_true
        let packed = (self.seen_false as u8) | ((self.seen_true as u8) << 1);
        Ok(vec![ScalarValue::Binary(Some(vec![packed]))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = downcast_value!(states[0], BinaryArray);
        for data in array.iter().flatten() {
            if !data.is_empty() {
                self.seen_false |= (data[0] & 1) != 0;
                self.seen_true |= (data[0] & 2) != 0;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Bitmap256Accumulator {
    /// 256 bits = 4 x u64, tracks values 0-255
    bitmap: [u64; 4],
}

impl Bitmap256Accumulator {
    fn new() -> Self {
        Self { bitmap: [0; 4] }
    }

    #[inline]
    fn set_bit(&mut self, value: u8) {
        let word = (value >> 6) as usize;
        let bit = value & 63;
        self.bitmap[word] |= 1u64 << bit;
    }

    #[inline]
    fn count(&self) -> u64 {
        self.bitmap.iter().map(|w| w.count_ones() as u64).sum()
    }

    fn merge(&mut self, other: &[u64; 4]) {
        for i in 0..4 {
            self.bitmap[i] |= other[i];
        }
    }
}

impl Accumulator for Bitmap256Accumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_primitive::<UInt8Type>();
        for value in array.iter().flatten() {
            self.set_bit(value);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = downcast_value!(states[0], BinaryArray);
        for data in array.iter().flatten() {
            if data.len() == 32 {
                // Convert &[u8] to [u64; 4]
                let mut other = [0u64; 4];
                for i in 0..4 {
                    let offset = i * 8;
                    other[i] =
                        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                }
                self.merge(&other);
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Serialize [u64; 4] as 32 bytes
        let mut bytes = Vec::with_capacity(32);
        for word in &self.bitmap {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(vec![ScalarValue::Binary(Some(bytes))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::UInt64(Some(self.count())))
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }
}

#[derive(Debug)]
struct Bitmap256AccumulatorI8 {
    bitmap: [u64; 4],
}

impl Bitmap256AccumulatorI8 {
    fn new() -> Self {
        Self { bitmap: [0; 4] }
    }

    #[inline]
    fn set_bit(&mut self, value: i8) {
        // Convert i8 to u8 by reinterpreting bits
        let idx = value as u8;
        let word = (idx >> 6) as usize;
        let bit = idx & 63;
        self.bitmap[word] |= 1u64 << bit;
    }

    #[inline]
    fn count(&self) -> u64 {
        self.bitmap.iter().map(|w| w.count_ones() as u64).sum()
    }

    fn merge(&mut self, other: &[u64; 4]) {
        for i in 0..4 {
            self.bitmap[i] |= other[i];
        }
    }
}

impl Accumulator for Bitmap256AccumulatorI8 {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_primitive::<Int8Type>();
        for value in array.iter().flatten() {
            self.set_bit(value);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = downcast_value!(states[0], BinaryArray);
        for data in array.iter().flatten() {
            if data.len() == 32 {
                let mut other = [0u64; 4];
                for i in 0..4 {
                    let offset = i * 8;
                    other[i] =
                        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                }
                self.merge(&other);
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut bytes = Vec::with_capacity(32);
        for word in &self.bitmap {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(vec![ScalarValue::Binary(Some(bytes))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::UInt64(Some(self.count())))
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }
}

/// Accumulator for u16 distinct counting using a 65536-bit bitmap
#[derive(Debug)]
struct Bitmap65536Accumulator {
    /// 65536 bits = 1024 x u64, tracks values 0-65535
    bitmap: Box<[u64; 1024]>,
}

impl Bitmap65536Accumulator {
    fn new() -> Self {
        Self {
            bitmap: Box::new([0; 1024]),
        }
    }

    #[inline]
    fn set_bit(&mut self, value: u16) {
        let word = (value / 64) as usize;
        let bit = value % 64;
        self.bitmap[word] |= 1u64 << bit;
    }

    #[inline]
    fn count(&self) -> u64 {
        self.bitmap.iter().map(|w| w.count_ones() as u64).sum()
    }

    fn merge(&mut self, other: &[u64; 1024]) {
        for i in 0..1024 {
            self.bitmap[i] |= other[i];
        }
    }
}

impl Accumulator for Bitmap65536Accumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_primitive::<UInt16Type>();
        for value in array.iter().flatten() {
            self.set_bit(value);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = downcast_value!(states[0], BinaryArray);
        for data in array.iter().flatten() {
            if data.len() == 8192 {
                let mut other = [0u64; 1024];
                for i in 0..1024 {
                    let offset = i * 8;
                    other[i] =
                        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                }
                self.merge(&other);
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut bytes = Vec::with_capacity(8192);
        for word in self.bitmap.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(vec![ScalarValue::Binary(Some(bytes))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::UInt64(Some(self.count())))
    }

    fn size(&self) -> usize {
        size_of::<Self>() + 8192
    }
}

/// Accumulator for i16 distinct counting using a 65536-bit bitmap
#[derive(Debug)]
struct Bitmap65536AccumulatorI16 {
    bitmap: Box<[u64; 1024]>,
}

impl Bitmap65536AccumulatorI16 {
    fn new() -> Self {
        Self {
            bitmap: Box::new([0; 1024]),
        }
    }

    #[inline]
    fn set_bit(&mut self, value: i16) {
        let idx = value as u16;
        let word = (idx / 64) as usize;
        let bit = idx % 64;
        self.bitmap[word] |= 1u64 << bit;
    }

    #[inline]
    fn count(&self) -> u64 {
        self.bitmap.iter().map(|w| w.count_ones() as u64).sum()
    }

    fn merge(&mut self, other: &[u64; 1024]) {
        for i in 0..1024 {
            self.bitmap[i] |= other[i];
        }
    }
}

impl Accumulator for Bitmap65536AccumulatorI16 {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array = values[0].as_primitive::<Int16Type>();
        for value in array.iter().flatten() {
            self.set_bit(value);
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let array = downcast_value!(states[0], BinaryArray);
        for data in array.iter().flatten() {
            if data.len() == 8192 {
                let mut other = [0u64; 1024];
                for i in 0..1024 {
                    let offset = i * 8;
                    other[i] =
                        u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
                }
                self.merge(&other);
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut bytes = Vec::with_capacity(8192);
        for word in self.bitmap.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        Ok(vec![ScalarValue::Binary(Some(bytes))])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::UInt64(Some(self.count())))
    }

    fn size(&self) -> usize {
        size_of::<Self>() + 8192
    }
}

macro_rules! default_accumulator_impl {
    () => {
        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            assert_eq!(1, states.len(), "expect only 1 element in the states");
            let binary_array = downcast_value!(states[0], BinaryArray);
            for v in binary_array.iter() {
                let v = v.ok_or_else(|| {
                    internal_datafusion_err!(
                        "Impossibly got empty binary array from states"
                    )
                })?;
                let other = v.try_into()?;
                self.hll.merge(&other);
            }
            Ok(())
        }

        fn state(&mut self) -> Result<Vec<ScalarValue>> {
            let value = ScalarValue::from(&self.hll);
            Ok(vec![value])
        }

        fn evaluate(&mut self) -> Result<ScalarValue> {
            Ok(ScalarValue::UInt64(Some(self.hll.count() as u64)))
        }

        fn size(&self) -> usize {
            // HLL has static size
            std::mem::size_of_val(self)
        }
    };
}

impl<T> Accumulator for BinaryHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericBinaryArray<T> =
            downcast_value!(values[0], GenericBinaryArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Accumulator for StringViewHLLAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &StringViewArray = downcast_value!(values[0], StringViewArray);

        // When all strings are stored inline in the StringView (≤ 12 bytes),
        // hash the raw u128 view directly instead of materializing a &str.
        if array.data_buffers().is_empty() {
            for (i, &view) in array.views().iter().enumerate() {
                if !array.is_null(i) {
                    self.hll.add_hashed(HLL_HASH_STATE.hash_one(view));
                }
            }
        } else {
            self.hll.extend(array.iter().flatten());
        }

        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for StringHLLAccumulator<T>
where
    T: OffsetSizeTrait,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &GenericStringArray<T> =
            downcast_value!(values[0], GenericStringArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl<T> Accumulator for NumericHLLAccumulator<T>
where
    T: ArrowPrimitiveType + Debug,
    T::Native: Hash,
{
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let array: &PrimitiveArray<T> = downcast_value!(values[0], PrimitiveArray, T);
        // flatten because we would skip nulls
        self.hll.extend(array.into_iter().flatten());
        Ok(())
    }

    default_accumulator_impl!();
}

impl Debug for ApproxDistinct {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApproxDistinct")
            .field("name", &self.name())
            .field("signature", &self.signature)
            .finish()
    }
}

impl Default for ApproxDistinct {
    fn default() -> Self {
        Self::new()
    }
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate number of distinct input values calculated using the HyperLogLog algorithm.",
    syntax_example = "approx_distinct(expression)",
    sql_example = r#"```sql
> SELECT approx_distinct(column_name) FROM table_name;
+-----------------------------------+
| approx_distinct(column_name)      |
+-----------------------------------+
| 42                                |
+-----------------------------------+
```"#,
    standard_argument(name = "expression",)
)]
#[derive(PartialEq, Eq, Hash)]
pub struct ApproxDistinct {
    signature: Signature,
}

impl ApproxDistinct {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for ApproxDistinct {
    fn name(&self) -> &str {
        "approx_distinct"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        if args.input_fields[0].data_type().is_null() {
            Ok(vec![
                Field::new(
                    format_state_name(args.name, self.name()),
                    DataType::Null,
                    true,
                )
                .into(),
            ])
        } else {
            Ok(vec![
                Field::new(
                    format_state_name(args.name, "hll_registers"),
                    DataType::Binary,
                    false,
                )
                .into(),
            ])
        }
    }

    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        let data_type = acc_args.expr_fields[0].data_type();

        let accumulator: Box<dyn Accumulator> = match data_type {
            // TODO u8, i8, u16, i16 shall really be done using bitmap, not HLL
            // TODO support for boolean (trivial case)
            // https://github.com/apache/datafusion/issues/1109
            DataType::UInt8 => Box::new(Bitmap256Accumulator::new()),
            DataType::UInt16 => Box::new(Bitmap65536Accumulator::new()),
            DataType::UInt32 => Box::new(NumericHLLAccumulator::<UInt32Type>::new()),
            DataType::UInt64 => Box::new(NumericHLLAccumulator::<UInt64Type>::new()),
            DataType::Int8 => Box::new(Bitmap256AccumulatorI8::new()),
            DataType::Int16 => Box::new(Bitmap65536AccumulatorI16::new()),
            DataType::Int32 => Box::new(NumericHLLAccumulator::<Int32Type>::new()),
            DataType::Int64 => Box::new(NumericHLLAccumulator::<Int64Type>::new()),
            DataType::Date32 => Box::new(NumericHLLAccumulator::<Date32Type>::new()),
            DataType::Date64 => Box::new(NumericHLLAccumulator::<Date64Type>::new()),
            DataType::Time32(TimeUnit::Second) => {
                Box::new(NumericHLLAccumulator::<Time32SecondType>::new())
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                Box::new(NumericHLLAccumulator::<Time32MillisecondType>::new())
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                Box::new(NumericHLLAccumulator::<Time64MicrosecondType>::new())
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                Box::new(NumericHLLAccumulator::<Time64NanosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                Box::new(NumericHLLAccumulator::<TimestampSecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMillisecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampMicrosecondType>::new())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Box::new(NumericHLLAccumulator::<TimestampNanosecondType>::new())
            }
            DataType::Utf8 => Box::new(StringHLLAccumulator::<i32>::new()),
            DataType::LargeUtf8 => Box::new(StringHLLAccumulator::<i64>::new()),
            DataType::Utf8View => Box::new(StringViewHLLAccumulator::new()),
            DataType::Binary => Box::new(BinaryHLLAccumulator::<i32>::new()),
            DataType::LargeBinary => Box::new(BinaryHLLAccumulator::<i64>::new()),
            DataType::Boolean => Box::new(BoolDistinctAccumulator::new()),
            DataType::Null => {
                Box::new(NoopAccumulator::new(ScalarValue::UInt64(Some(0))))
            }
            other => {
                return not_impl_err!(
                    "Support for 'approx_distinct' for data type {other} is not implemented"
                );
            }
        };
        Ok(accumulator)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}
