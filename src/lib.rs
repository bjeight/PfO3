use std::{collections::{HashMap, HashSet}, iter::zip, cmp::Ordering};

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    ndarray::{Array, Dim, AssignElem, ArrayBase, OwnedRepr, ViewRepr},
    ToPyArray, PyArray};


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

fn _index_vec_missing(aa: ArrayBase<ViewRepr<&i8>, Dim<[usize; 2]>>, axis: usize, threshold: f64) -> Vec<i32> {
    let mut idx: Vec<i32> = vec![];
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(axis)).enumerate() {
        let mut m: i64 = 0;
        for x in v {
            if *x < 0 {
                m += 1;
            }
        }
        if m as f64 / v.len() as f64 <= threshold {
            idx.push(i as i32);
        }
    }
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn index_sample_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> Vec<i32> {
    let aa = a.as_array();
    let idx = _index_vec_missing(aa, 1, threshold);
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn filter_sample_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let idx = _index_vec_missing(aa, 1, threshold);
    let idx_hs: HashSet<i32> = idx.into_iter().collect();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((aa.shape()[0], 0));
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(1)).enumerate() {
        if idx_hs.contains(&(i as i32)) {
            b.push(numpy::ndarray::Axis(1), v).unwrap();
        }
    }
    b.to_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn index_site_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> Vec<i32> {
    let aa = a.as_array();
    let idx = _index_vec_missing(aa, 0, threshold);
    idx
}

#[pyfunction]
#[pyo3(signature = (a, threshold = 0.1))]
fn filter_site_missing<'a>(py: Python<'a>, a: PyReadonlyArray2<'a, i8>, threshold: f64) -> &'a PyArray<i8, Dim<[usize; 2]>> {
    let aa = a.as_array();
    let idx = _index_vec_missing(aa, 0, threshold);
    let idx_hs: HashSet<i32> = idx.into_iter().collect();
    let mut b: ArrayBase<OwnedRepr<i8>, Dim<[usize; 2]>> = Array::zeros((0, aa.shape()[1]));
    for (i, v) in aa.axis_iter(numpy::ndarray::Axis(0)).enumerate() {
        if idx_hs.contains(&(i as i32)) {
            b.push(numpy::ndarray::Axis(0), v).unwrap();
        }
    }
    b.to_pyarray(py)
}


#[pyfunction]
fn distinct_counts(a: PyReadonlyArray2<i8>) -> Vec<f64> {

    fn any_missing(s: &Vec<i8>) -> bool {
        for i in s {
            if *i < 0 {
                return true
            }
        }
        false
    }

    fn total_missing(v: &Vec<i8>) -> i64 {
        let mut c: i64 = 0;
        for i in v {
            if *i < 0 {
                c += 1;
            }
        }
        c
    }

    fn get_scores(vs: &Vec<Vec<i8>>) -> Vec<i64> {
        let mut cs = vec![];
        for v in vs {
            cs.push(total_missing(v))
        }
        cs
    }

    fn are_same(s1: &Vec<i8>, s2: &Vec<i8>) -> bool {
        assert_eq!(s1.len(), s2.len()); // ?
        for i in 0..s1.len() {
            if s1[i] < 0 || s2[i] < 0 {
                continue
            }
            if s1[i] != s2[i] {
                return false
            }
        }
        true
    }

    let aa = a.as_array();
    let mut m_full_hap: HashMap::<Vec<i8>, f64> = HashMap::new();
    let mut m_miss_hap: HashMap::<Vec<i8>, f64> = HashMap::new();
    // let mut leftovers: Vec<Vec<i8>> = vec![vec![]];
    for row in aa.axis_iter(numpy::ndarray::Axis(1)) {
        let rs = row.to_vec();
        if !any_missing(&rs) { // if there are no missing values they can be added to our map
            m_full_hap.entry(rs)
                .and_modify(|e| { *e += 1.0 })
                .or_insert(1.0);
        } else { //otherwise, keep them aside for comparing against the more complete haplotypes later
            m_miss_hap.entry(rs)
                .and_modify(|e| { *e += 1.0 })
                .or_insert(1.0);
        }
    }

    // NB - handle the case where there is now nothing in m (e.g. every sequence had at least one missing value)?
    // NB - what about the case where partial haplotypes match with each other?

    let leftovers = m_miss_hap.keys().cloned().collect();
    let scores = get_scores(&leftovers);
    let mut zipped: Vec<(i64, Vec<i8>)> = zip(scores, leftovers).collect();
    zipped.sort_by(|a, b| a.0.cmp(&b.0));
    let leftover_sorted: Vec<Vec<i8>> = zipped.into_iter().map(|x| x.1).collect();

    let mut to_add: HashMap<Vec<i8>, f64> = HashMap::new();
    for row in leftover_sorted {
        let mut key_matches: Vec<Vec<i8>> = vec![];
        for k in m_full_hap.keys() {
            if are_same(&row, k) {
                key_matches.push(k.clone())
            }
        }
        let n_match = key_matches.len();
        if n_match == 0 {
            let c = m_miss_hap.remove(&row).unwrap();
            to_add.insert(row, c);
        } else {
            let c = m_miss_hap.remove(&row).unwrap() / n_match as f64;
            for m in key_matches {
                m_full_hap.entry(m)
                    .and_modify(|e| { *e += c });
            }
        }
    }

    for (k,v) in to_add.drain() {
        m_full_hap.insert(k, v);
    }

    let mut counts: Vec<f64> = vec![];
    for (_, v) in m_full_hap.drain() {
        counts.push(v)
    }

    counts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    counts
}

#[pyfunction]
#[pyo3(signature = (s))]
fn return_array(py: Python<'_>, s: usize) -> &PyArray<i8, Dim<[usize; 2]>> {
    let a = Array::zeros((s, s));
    a.to_pyarray(py)
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
    m.add_function(wrap_pyfunction!(distinct_counts, m)?)?;
    m.add_function(wrap_pyfunction!(haploidify_samples, m)?)?;
    Ok(())
}