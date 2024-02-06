use numpy::{ndarray::{s, ArrayBase, Dim, ViewRepr}, PyReadonlyArray2};

use crate::haplotypes::*;

fn factorial(n: u64) -> u64 {
    (1..=n).product()
}

fn combinations(n: u64, r: u64) -> u64 {
    (n - r + 1..=n).product::<u64>() / factorial(r)
}

fn ehh_i(aa: ArrayBase<ViewRepr<&i8>, Dim<[usize; 2]>>) -> f64 {
    let n = aa.shape()[0] as u64;
    let m_hap = distinct_haplotypes_counts(aa);
    let mut homozygosities: Vec<f64> = vec![];
    for (_, v) in m_hap.iter() {
        let h = combinations(*v, 2) as f64 / combinations(n, 2) as f64;
        homozygosities.push(h);
    }
    let s = homozygosities.iter().sum::<f64>();
    s
}

fn ehh(a: PyReadonlyArray2<i8>, i: usize, w: usize) -> f64 {
    let aa = a.as_array();
    let aw = aa.slice(s![i-w..i+w,..]);
    let ehh_i = ehh_i(aw);
    ehh_i
}
