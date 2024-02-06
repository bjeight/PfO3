use std::{cmp::Ordering, collections::HashMap};

use numpy::{ndarray::{ArrayBase, Dim, ViewRepr}, PyReadonlyArray2};
use pyo3::pyfunction;

pub fn distinct_haplotypes_counts(aa: ArrayBase<ViewRepr<&i8>, Dim<[usize; 2]>>) -> HashMap::<Vec<i8>, u64> {
    let mut m_hap: HashMap::<Vec<i8>, u64> = HashMap::new();
    for row in aa.axis_iter(numpy::ndarray::Axis(1)) {
        let rs = row.to_vec();
        m_hap.entry(rs)
            .and_modify(|e| { *e += 1 })
            .or_insert(1);
    }
    m_hap
}

#[pyfunction]
pub fn distinct_counts(a: PyReadonlyArray2<i8>) -> Vec<u64> {
    let aa = a.as_array();
    let m_hap = distinct_haplotypes_counts(aa);
    let mut counts: Vec<u64> = vec![];
    for (_, v) in m_hap.iter() {
        counts.push(*v)
    }
    counts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    counts
}

pub fn freqs_from_counts(counts: Vec<u64>) -> Vec<f64> {
    let mut v: Vec<f64> = vec![];
    let s: u64 = counts.iter().sum();
    for c in counts {
        v.push(c as f64 / s as f64);
    }
    v
}