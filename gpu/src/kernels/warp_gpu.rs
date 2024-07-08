use krnl::macros::module;


#[module]
pub mod kernels {

    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::buffer::UnsafeIndex;
    use krnl_core::buffer::UnsafeSlice;
    use krnl_core::macros::kernel;

    pub trait DiagonalMatrix: Sync + Send {
        fn new(a_len: usize, b_len: usize, init_val: f32) -> Self;
        fn set_diagonal_cell(&mut self, diag_row: usize, diag_offset: isize, value: f32);
        fn get_diagonal_cell(&self, diag_row: usize, diag_offset: isize) -> f32;

        fn index_mat_to_diag(i: usize, j: usize) -> (usize, isize) {
            (i + j, (j as isize) - (i as isize))
        }
        fn index_diag_to_mat(r: usize, c: isize) -> (usize, usize) {
            ((r as isize - c) as usize / 2, (r as isize + c) as usize / 2)
        }

        fn debug_print(&self);
    }

    pub struct GpuMatrix<'a> {
        diagonal: UnsafeSlice<'a, f32>,
        mask: usize,
    }

    impl DiagonalMatrix for GpuMatrix<'_> {
        fn new(_a_len: usize, _b_len: usize, _init_val: f32) -> Self {
            unimplemented!()
        }

        #[inline(always)]
        fn get_diagonal_cell(&self, _diag_row: usize, diag_offset: isize) -> f32 {
            unsafe { *self.diagonal.unsafe_index(diag_offset as usize & self.mask) }
        }

        #[inline(always)]
        fn set_diagonal_cell(&mut self, _diag_row: usize, diag_offset: isize, value: f32) {
            unsafe {
                *self
                    .diagonal
                    .unsafe_index_mut(diag_offset as usize & self.mask) = value;
            }
        }

        fn debug_print(&self) {
            unimplemented!()
        }
    }

    #[inline(always)]
    pub fn warp_kernel_impl<M: DiagonalMatrix>(
        mut matrix: M,
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

    #[kernel]
    pub fn warp_kernel(
        first_coord: i64,
        row: u64,
        diamonds_count: u64,
        a_start: u64,
        b_start: u64,
        a_len: u64,
        b_len: u64,
        max_subgroup_threads: u64,
        #[global] a: Slice<f32>,
        #[global] b: Slice<f32>,
        #[global] diagonal: UnsafeSlice<f32>,
    ) {
        use crate::krnl_core::num_traits::Float;
        use crate::krnl_core::num_traits::Pow;
        use krnl_core::buffer::UnsafeIndex;

        let global_id = kernel.global_id() as u64;

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
            mask: diagonal.len() - 1,
        };

        warp_kernel_impl(
            matrix,
            row * max_subgroup_threads,
            d_a_start,
            d_b_start,
            diag_start + (max_subgroup_threads as i64),
            (max_subgroup_threads * 2 + 1).min(alen + blen + 1),
            warp_id,
            max_subgroup_threads,
            |i, j, x, y, z| {
                let dist = (a[i as usize] - b[j as usize]).abs();
                dist + z.min(x.min(y))
            },
        );
    }
}
