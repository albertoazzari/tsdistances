use krnl::macros::module;

#[module]
#[allow(dead_code)]
#[allow(unexpected_cfgs)]
pub mod warp {

    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::buffer::UnsafeIndex;
    use krnl_core::buffer::UnsafeSlice;

    pub struct GpuMatrix<'a> {
        diagonal: UnsafeSlice<'a, f32>,
        diagonal_offset: usize,
        mask: usize,
    }

    impl GpuMatrix<'_> {
        #[inline(always)]
        fn get_diagonal_cell(&self, _diag_row: usize, diag_offset: isize) -> f32 {
            unsafe {
                *self
                    .diagonal
                    .unsafe_index(self.diagonal_offset + (diag_offset as usize & self.mask))
            }
        }

        #[inline(always)]
        fn set_diagonal_cell(&mut self, _diag_row: usize, diag_offset: isize, value: f32) {
            unsafe {
                *self
                    .diagonal
                    .unsafe_index_mut(self.diagonal_offset + (diag_offset as usize & self.mask)) =
                    value;
            }
        }
    }

    #[inline(always)]
    fn warp_kernel_inner(
        mut matrix: GpuMatrix,
        d_offset: u64,
        a_start: u64,
        b_start: u64,
        diag_mid: i64,
        diag_count: u64,
        warp: u64,
        max_subgroup_threads: u64,
        dist_lambda: impl Fn(u64, u64, f32, f32, f32) -> f32,
    ) {
        let mut i = a_start;
        let mut j = b_start;
        let mut s = diag_mid;
        let mut e = diag_mid;

        for d in 2..diag_count {
            let k = (warp * 2) as i64 + s;
            if k <= e {
                let i1 = i - warp;
                let j1 = j + warp;

                let dleft = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k - 1) as isize);
                let ddiag = matrix.get_diagonal_cell((d_offset + d - 2) as usize, k as isize);
                let dup = matrix.get_diagonal_cell((d_offset + d - 1) as usize, (k + 1) as isize);

                let value = dist_lambda(i1, j1, dleft, ddiag, dup);

                matrix.set_diagonal_cell((d_offset + d) as usize, k as isize, value);
            }
            // Warp synchronize
            #[cfg(target_arch = "spirv")]
            unsafe {
                crate::krnl_core::spirv_std::arch::workgroup_memory_barrier_with_group_sync()
            };

            if d <= max_subgroup_threads {
                i += 1;
                s -= 1;
                e += 1;
            } else {
                j += 1;
                s += 1;
                e -= 1;
            }
        }
    }

    #[inline(always)]
    fn warp_kernel(
        global_id: u64,
        first_coord: i64,
        row: u64,
        diamonds_count: u64,
        a_start: u64,
        b_start: u64,
        a_len: u64,
        b_len: u64,
        max_subgroup_threads: u64,
        diagonal: krnl_core::buffer::UnsafeSlice<f32>,
        diagonal_offset: u64,
        diagonal_len: u64,
        distance_lambda: impl Fn(u64, u64, f32, f32, f32) -> f32,
    ) {
        let warp_id = global_id % max_subgroup_threads;
        let diamond_id = global_id / max_subgroup_threads;

        if diamond_id >= diamonds_count {
            return;
        }

        let diag_start = first_coord + ((diamond_id * max_subgroup_threads) as i64) * 2;
        let d_a_start = a_start - diamond_id * max_subgroup_threads;
        let d_b_start = b_start + diamond_id * max_subgroup_threads;

        let alen = a_len - d_a_start;
        let blen = b_len - d_b_start;

        let matrix = GpuMatrix {
            diagonal,
            diagonal_offset: diagonal_offset as usize,
            mask: diagonal_len as usize - 1,
        };

        warp_kernel_inner(
            matrix,
            row * max_subgroup_threads,
            d_a_start,
            d_b_start,
            diag_start + (max_subgroup_threads as i64),
            (max_subgroup_threads * 2 + 1).min(alen + blen + 1),
            warp_id,
            max_subgroup_threads,
            distance_lambda,
        );
    }

    macro_rules! warp_kernel_spec {
        ($(
            fn $name:ident[$impl_struct:ident](
                $a:ident[$a_offset:ident],
                $b:ident[$b_offset:ident],
                $i:ident,
                $j:ident,
                $x:ident,
                $y:ident,
                $z:ident,
                [$($param1:ident: $ty1:ty)?],
                [$($param2:ident: $ty2:ty)?],
                [$($param3:ident: $ty3:ty)?],
                [$($param4:ident: $ty4:ty)?],
                [$($vec5:ident: $ty5:ty)?]
            ) $body:block
        )*) => {
            $(
                pub mod $name {
                    #[cfg(not(target_arch = "spirv"))]
                    use krnl::krnl_core;
                    use krnl_core::macros::kernel;

                    #[cfg(not(target_arch = "spirv"))]
                    use super::GpuKernelImpl;

                    #[cfg(not(target_arch = "spirv"))]
                    pub struct $impl_struct {
                        $(pub $param1: $ty1,)?
                        $(pub $param2: $ty2,)?
                        $(pub $param3: $ty3,)?
                        $(pub $param4: $ty4,)?
                        $(pub $vec5:  krnl::buffer::Buffer<$ty5>,)?
                    }

                    #[cfg(not(target_arch = "spirv"))]
                    impl GpuKernelImpl for $impl_struct {
                        fn dispatch(
                            &self,
                            device: krnl::device::Device,
                            first_coord: i64,
                            row: u64,
                            diamonds_count: u64,
                            a_start: u64,
                            b_start: u64,
                            a_len: u64,
                            b_len: u64,
                            max_subgroup_threads: u64,
                            a: krnl::buffer::Slice<f32>,
                            b: krnl::buffer::Slice<f32>,
                            diagonal: krnl::buffer::SliceMut<f32>,
                        ) {
                            single_call::builder()
                                .unwrap()
                                .build(device)
                                .unwrap()
                                .with_global_threads((diamonds_count * max_subgroup_threads) as u32)
                                .dispatch(
                                    first_coord,
                                    row,
                                    diamonds_count,
                                    a_start,
                                    b_start,
                                    a_len,
                                    b_len,
                                    max_subgroup_threads,
                                    $(self.$param1,)?
                                    $(self.$param2,)?
                                    $(self.$param3,)?
                                    $(self.$param4,)?
                                    $(self.$vec5.as_slice(),)?
                                    a,
                                    b,
                                    diagonal,
                                )
                                .unwrap();
                        }

                        fn dispatch_batch(
                            &self,
                            device: krnl::device::Device,
                            first_coord: i64,
                            row: u64,
                            diamonds_count: u64,
                            a_start: u64,
                            b_start: u64,
                            a_len: u64,
                            b_len: u64,
                            padded_a_len: u64,
                            padded_b_len: u64,
                            max_subgroup_threads: u64,
                            a: krnl::buffer::Slice<f32>,
                            b: krnl::buffer::Slice<f32>,
                            diagonal: krnl::buffer::SliceMut<f32>,
                        ) {
                            let a_count = a.len() as u64 / padded_a_len;
                            let b_count = b.len() as u64 / padded_b_len;
                            batch_call::builder()
                                .unwrap()
                                .build(device)
                                .unwrap()
                                .with_global_threads((a_count * b_count * diamonds_count * max_subgroup_threads) as u32)
                                .dispatch(
                                    first_coord,
                                    row,
                                    diamonds_count,
                                    a_start,
                                    b_start,
                                    a_len,
                                    b_len,
                                    padded_a_len,
                                    padded_b_len,
                                    max_subgroup_threads,
                                    $(self.$param1,)?
                                    $(self.$param2,)?
                                    $(self.$param3,)?
                                    $(self.$param4,)?
                                    $(self.$vec5.as_slice(),)?
                                    a,
                                    b,
                                    diagonal,
                                )
                                .unwrap();
                        }
                    }

                    #[kernel]
                    fn single_call(
                        first_coord: i64,
                        row: u64,
                        diamonds_count: u64,
                        a_start: u64,
                        b_start: u64,
                        a_len: u64,
                        b_len: u64,
                        max_subgroup_threads: u64,
                        $(param1: $ty1,)?
                        $(param2: $ty2,)?
                        $(param3: $ty3,)?
                        $(param4: $ty4,)?
                        $(#[global] vec5: Slice<$ty5>,)?
                        #[global] $a: Slice<f32>,
                        #[global] $b: Slice<f32>,
                        #[global] diagonal: UnsafeSlice<f32>,
                    ) {
                        use krnl_core::buffer::UnsafeIndex;
                        use krnl_core::num_traits::Float;

                        $(let $param1 = param1;)?
                        $(let $param2 = param2;)?
                        $(let $param3 = param3;)?
                        $(let $param4 = param4;)?
                        $(let $vec5 = vec5;)?

                        let $a_offset = 0;
                        let $b_offset = 0;

                        let global_id = kernel.global_id() as u64;
                        super::warp_kernel(
                            global_id,
                            first_coord,
                            row,
                            diamonds_count,
                            a_start,
                            b_start,
                            a_len,
                            b_len,
                            max_subgroup_threads,
                            diagonal,
                            0,
                            diagonal.len() as u64,
                            |$i, $j, $x, $y, $z| $body,
                        );
                    }

                    #[kernel]
                    fn batch_call(
                        first_coord: i64,
                        row: u64,
                        diamonds_count: u64,
                        a_start: u64,
                        b_start: u64,
                        a_len: u64,
                        b_len: u64,
                        padded_a_len: u64,
                        padded_b_len: u64,
                        max_subgroup_threads: u64,
                        $(param1: $ty1,)?
                        $(param2: $ty2,)?
                        $(param3: $ty3,)?
                        $(param4: $ty4,)?
                        $(#[global] vec5: Slice<$ty5>,)?
                        #[global] $a: Slice<f32>,
                        #[global] $b: Slice<f32>,
                        #[global] diagonal: UnsafeSlice<f32>,
                    ) {
                        use krnl_core::buffer::UnsafeIndex;
                        use krnl_core::num_traits::Float;

                        $(let $param1 = param1;)?
                        $(let $param2 = param2;)?
                        $(let $param3 = param3;)?
                        $(let $param4 = param4;)?
                        $(let $vec5 = vec5;)?


                        let global_id = kernel.global_id() as u64;
                        let threads_stride = diamonds_count * max_subgroup_threads;

                        let pair_index = global_id / threads_stride;
                        let instance_id = global_id % threads_stride;

                        let a_count = $a.len() / padded_a_len as usize;
                        let b_count = $b.len() / padded_b_len as usize;

                        let a_index = pair_index / b_count as u64;
                        let b_index = pair_index % b_count as u64;

                        let diagonal_stride = (diagonal.len() / (a_count * b_count)) as u64;
                        let diagonal_offset = pair_index * diagonal_stride;

                        let $a_offset = a_index as usize * padded_a_len as usize;
                        let $b_offset = b_index as usize * padded_b_len as usize;

                        super::warp_kernel(
                            instance_id,
                            first_coord,
                            row,
                            diamonds_count,
                            a_start,
                            b_start,
                            a_len,
                            b_len,
                            max_subgroup_threads,
                            diagonal,
                            diagonal_offset,
                            diagonal_stride,
                            |$i, $j, $x, $y, $z| $body,
                        );
                    }
                }
            )*
        };
    }

    #[cfg(not(target_arch = "spirv"))]
    pub trait GpuKernelImpl {
        fn dispatch(
            &self,
            device: krnl::device::Device,
            first_coord: i64,
            row: u64,
            diamonds_count: u64,
            a_start: u64,
            b_start: u64,
            a_len: u64,
            b_len: u64,
            max_subgroup_threads: u64,
            a: krnl::buffer::Slice<f32>,
            b: krnl::buffer::Slice<f32>,
            diagonal: krnl::buffer::SliceMut<f32>,
        );

        fn dispatch_batch(
            &self,
            device: krnl::device::Device,
            first_coord: i64,
            row: u64,
            diamonds_count: u64,
            a_start: u64,
            b_start: u64,
            a_len: u64,
            b_len: u64,
            padded_a_len: u64,
            padded_b_len: u64,
            max_subgroup_threads: u64,
            a: krnl::buffer::Slice<f32>,
            b: krnl::buffer::Slice<f32>,
            diagonal: krnl::buffer::SliceMut<f32>,
        );
    }

    const MSM_C: f32 = 1.0;
    #[inline(always)]
    pub fn msm_cost_function(x: f32, y: f32, z: f32) -> f32 {
        MSM_C + (y.min(z) - x).max(x - y.max(z)).max(0.0)
    }

    warp_kernel_spec! {
        fn erp_distance[ERPImpl](a[a_offset], b[b_offset], i, j, x, y, z, [gap_penalty: f32], [], [], [], []) {
            (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
            .min((z + (a[a_offset + i as usize] - gap_penalty).abs()).min(x + (b[b_offset + j as usize] - gap_penalty).abs()))
        }
        fn lcss_distance[LCSSImpl](a[a_offset], b[b_offset], i, j, x, y, z, [epsilon: f32], [], [], [], []) {
            let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
            (dist <= epsilon) as i32 as f32 * (y + 1.0) + (dist > epsilon) as i32 as f32 * x.max(z)
        }
        fn dtw_distance[DTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
            let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
            dist + z.min(x.min(y))
        }
        fn wdtw_distance[WDTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], [weights: f32]) {
            let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2) * weights[(i as i32 - j as i32).abs() as usize];
            dist + x.min(y.min(z))
        }
        fn msm_distance[MSMImpl](a[a_offset], b[b_offset], i, j, x, y, z, [], [], [], [], []) {
            (y + (a[a_offset + i as usize] - b[b_offset + j as usize]).abs())
            .min(
                z + super::msm_cost_function(a[a_offset + i as usize], if i == 0 {0.0} else {a[a_offset + i as usize - 1]}, b[b_offset + j as usize]),
            )
            .min(
                x + super::msm_cost_function(b[b_offset + j as usize], a[a_offset + i as usize], if j == 0 {0.0} else {b[b_offset + j as usize - 1]}),
            )
        }
        fn twe_distance[TWEImpl](a[a_offset], b[b_offset], i, j, x, y, z, [stiffness: f32], [penalty: f32], [], [], []) {
            let delete_addition = penalty + stiffness;
            // deletion in a
            let del_a =
            z + (if i == 0 {0.0} else {a[a_offset + i as usize - 1]} - a[a_offset + i as usize]).abs() + delete_addition;

            // deletion in b
            let del_b =
                x + (if j == 0 {0.0} else {b[b_offset + j as usize - 1]} - b[b_offset + j as usize]).abs() + delete_addition;

            // match
            let match_current = (a[a_offset + i as usize] - b[b_offset + j as usize]).abs();
            let match_previous = (if i == 0 {0.0} else {a[a_offset + i as usize - 1]}
                - if j == 0 {0.0} else {b[b_offset + j as usize - 1]})
            .abs();
            let match_a_b = y
                + match_current
                + match_previous
                + stiffness * (2.0 * (i as isize - j as isize).abs() as f32);

            del_a.min(del_b.min(match_a_b))
        }
        fn adtw_distance[ADTWImpl](a[a_offset], b[b_offset], i, j, x, y, z, [w: f32], [], [], [], []) {
            let dist = (a[a_offset + i as usize] - b[b_offset + j as usize]).powi(2);
                    dist + (z + w).min((x + w).min(y))
        }

    }
}
