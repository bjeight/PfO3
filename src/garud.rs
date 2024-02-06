fn h1(freqs: Vec<f64>) -> Option<f64> {
    match freqs.len() {
        0 => return None,
        _ => {
            let mut v: Vec<f64> = vec![];
            for f in freqs {
                v.push(f * f)
            }
            let h1: f64 = v.iter().sum();
            return Some(h1);
        }
    }
}

fn h12(freqs: Vec<f64>) -> Option<f64> {
    match freqs.len() {
        0 => return None,
        1 | 2 => return Some(1.0),
        _ => {
            let firsttwo = (freqs[0] + freqs[1]) * (freqs[0] + freqs[1]);
            let mut v: Vec<f64> = vec![];
            for f in freqs[2..].iter() {
                v.push(f * f)
            }
            let h12: f64 = firsttwo + v.iter().sum::<f64>();
            return Some(h12)
        }
    }   
}

fn h123(freqs: Vec<f64>) -> Option<f64> {
    match freqs.len() {
        0 => return None,
        1 | 2 | 3 => return Some(1.0),
        _ => {
            let firstthree = (freqs[0] + freqs[1] + freqs[2]) * (freqs[0] + freqs[1] + freqs[2]);
            let mut v: Vec<f64> = vec![];
            for f in freqs[3..].iter() {
                v.push(f * f)
            }
            let h123: f64 = firstthree + v.iter().sum::<f64>();
            return Some(h123)
        }
    }   
}

fn h2_h1(freqs: Vec<f64>) -> Option<f64> {
    match freqs.len() {
        0 | 1 => return None,
        _ => {
            let mut v: Vec<f64> = vec![];
            for f in freqs[1..].iter() {
                v.push(f * f)
            }
            let h2_h1: f64 = v.iter().sum::<f64>();
            return Some(h2_h1);
        }
    }
}

// #[pyfunction]
// fn distinct_counts(a: PyReadonlyArray2<i8>) -> Vec<f64> {

//     fn any_missing(s: &Vec<i8>) -> bool {
//         for i in s {
//             if *i < 0 {
//                 return true
//             }
//         }
//         false
//     }

//     fn total_missing(v: &Vec<i8>) -> i64 {
//         let mut c: i64 = 0;
//         for i in v {
//             if *i < 0 {
//                 c += 1;
//             }
//         }
//         c
//     }

//     fn get_scores(vs: &Vec<Vec<i8>>) -> Vec<i64> {
//         let mut cs = vec![];
//         for v in vs {
//             cs.push(total_missing(v))
//         }
//         cs
//     }

//     fn are_same(s1: &Vec<i8>, s2: &Vec<i8>) -> bool {
//         assert_eq!(s1.len(), s2.len()); // ?
//         for i in 0..s1.len() {
//             if s1[i] < 0 || s2[i] < 0 {
//                 continue
//             }
//             if s1[i] != s2[i] {
//                 return false
//             }
//         }
//         true
//     }

//     let aa = a.as_array();
//     let mut m_full_hap: HashMap::<Vec<i8>, f64> = HashMap::new();
//     let mut m_miss_hap: HashMap::<Vec<i8>, f64> = HashMap::new();
//     // let mut leftovers: Vec<Vec<i8>> = vec![vec![]];
//     for row in aa.axis_iter(numpy::ndarray::Axis(1)) {
//         let rs = row.to_vec();
//         if !any_missing(&rs) { // if there are no missing values they can be added to our map
//             m_full_hap.entry(rs)
//                 .and_modify(|e| { *e += 1.0 })
//                 .or_insert(1.0);
//         } else { //otherwise, keep them aside for comparing against the more complete haplotypes later
//             m_miss_hap.entry(rs)
//                 .and_modify(|e| { *e += 1.0 })
//                 .or_insert(1.0);
//         }
//     }

//     // NB - handle the case where there is now nothing in m (e.g. every sequence had at least one missing value)?
//     // NB - what about the case where partial haplotypes match with each other?

//     let leftovers = m_miss_hap.keys().cloned().collect();
//     let scores = get_scores(&leftovers);
//     let mut zipped: Vec<(i64, Vec<i8>)> = zip(scores, leftovers).collect();
//     zipped.sort_by(|a, b| a.0.cmp(&b.0));
//     let leftover_sorted: Vec<Vec<i8>> = zipped.into_iter().map(|x| x.1).collect();

//     let mut to_add: HashMap<Vec<i8>, f64> = HashMap::new();
//     for row in leftover_sorted {
//         let mut key_matches: Vec<Vec<i8>> = vec![];
//         for k in m_full_hap.keys() {
//             if are_same(&row, k) {
//                 key_matches.push(k.clone())
//             }
//         }
//         let n_match = key_matches.len();
//         if n_match == 0 {
//             let c = m_miss_hap.remove(&row).unwrap();
//             to_add.insert(row, c);
//         } else {
//             let c = m_miss_hap.remove(&row).unwrap() / n_match as f64;
//             for m in key_matches {
//                 m_full_hap.entry(m)
//                     .and_modify(|e| { *e += c });
//             }
//         }
//     }

//     for (k,v) in to_add.drain() {
//         m_full_hap.insert(k, v);
//     }

//     let mut counts: Vec<f64> = vec![];
//     for (_, v) in m_full_hap.drain() {
//         counts.push(v)
//     }

//     counts.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));

//     counts
// }

// m.add_function(wrap_pyfunction!(distinct_counts, m)?)?;
