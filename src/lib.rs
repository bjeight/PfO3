use std::collections::HashSet;

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray3,
    ndarray::{Array, Dim, AssignElem, ArrayBase, OwnedRepr, ViewRepr},
    ToPyArray, PyArray};

mod haplotypes;

mod garud;
use crate::garud::*;

mod ehh;
use crate::ehh::*;

fn _index_site_polymorphic(aa: ArrayBase<ViewRepr<&i8>, Dim<[usize; 2]>>) -> Vec<i32> {
    let mut idx: Vec<i32> = vec![];
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(0)).enumerate() {
        let mut hs: HashSet<i8> = HashSet::new();
        for x in v {
            if *x > -1 {
                hs.insert(*x);
            }
        }
        if hs.len() > 1 {
            idx.push(i as i32);
        }
    }
    idx
}

#[pyfunction]
fn index_site_polymorphic<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>) -> Vec<i32> {
    let aa = a.as_array();
    let idx: Vec<i32> = _index_site_polymorphic(aa);
    idx
}

#[pyfunction]
fn filter_site_polymorphic<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let idx = _index_site_polymorphic(aa);
    let idx_hs: HashSet<i32> = idx.into_iter().collect();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((0, aa.shape()[1]));
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(0)).enumerate() {
        if idx_hs.contains(&(i as i32)) {
            b.push(numpy::ndarray::Axis(0), v).unwrap();
        }
    }
    b.to_pyarray(py)
}

fn _index_axis_missing(aa: ArrayBase<ViewRepr<&i8>, Dim<[usize; 2]>>, axis: usize, threshold: f64) -> Vec<u32> {
    let mut idx: Vec<u32> = vec![];
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(axis)).enumerate() {
        let mut m: u64 = 0;
        for x in v {
            if *x < 0 {
                m += 1;
            }
        }
        if m as f64 / v.len() as f64 <= threshold {
            idx.push(i as u32);
        }
    }
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn index_sample_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> Vec<u32> {
    let aa = a.as_array();
    let idx = _index_axis_missing(aa, 1, threshold);
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn filter_sample_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let idx = _index_axis_missing(aa, 1, threshold);
    let idx_hs: HashSet<u32> = idx.into_iter().collect();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((aa.shape()[0], 0));
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(1)).enumerate() {
        if idx_hs.contains(&(i as u32)) {
            b.push(numpy::ndarray::Axis(1), v).unwrap();
        }
    }
    b.to_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn index_site_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> Vec<u32> {
    let aa = a.as_array();
    let idx = _index_axis_missing(aa, 0, threshold);
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn filter_site_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let idx = _index_axis_missing(aa, 0, threshold);
    let idx_hs: HashSet<u32> = idx.into_iter().collect();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((0, aa.shape()[1]));
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(0)).enumerate() {
        if idx_hs.contains(&(i as u32)) {
            b.push(numpy::ndarray::Axis(0), v).unwrap();
        }
    }
    b.to_pyarray(py)
}


#[pyfunction]
fn haploidify_samples<'a>(py: Python<'a>, a: PyReadonlyArray3<'a, i8>) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let shape = aa.shape();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((shape[0], shape[1]));

    fn haploidify_genotype(a1: i8, a2: i8) -> i8 {
        if a1 == a2 {
            return a1
        }
        -1
    }

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let a1: i8 = aa[[i,j,0]];
            let a2: i8 = aa[[i,j,1]];
            let ht: i8 = haploidify_genotype(a1, a2);
            b[[i,j]].assign_elem(ht)
        }
    }

    b.to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn PfO3(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_site_polymorphic, m)?)?;
    m.add_function(wrap_pyfunction!(filter_site_polymorphic, m)?)?;
    m.add_function(wrap_pyfunction!(index_site_missing, m)?)?;
    m.add_function(wrap_pyfunction!(filter_site_missing, m)?)?;
    m.add_function(wrap_pyfunction!(index_sample_missing, m)?)?;
    m.add_function(wrap_pyfunction!(filter_sample_missing, m)?)?;
    m.add_function(wrap_pyfunction!(haploidify_samples, m)?)?;
    Ok(())
}