use std::{cell, mem};
use rdst::{RadixKey, RadixSort};

use rayon::{iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator}, slice::ParallelSlice};
use ultraviolet::{Vec2, Vec2x8};
use wide::{CmpGt, CmpLe, CmpLt, f32x8, i32x8, u32x8};

const SAFTY: f32 = 1e-5;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    
    let mut sum = f32x8::ZERO;
    let (chunks_a, reminder_a) = a.as_chunks::<8>();
    let (chunks_b, reminder_b) = b.as_chunks::<8>();
    
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b.iter()) {
        let va = f32x8::from(*chunk_a);
        let vb = f32x8::from(*chunk_b);
        sum += va * vb;
    }
    
    let mut result = sum.to_array().iter().sum();
    for (remainder_a, remainder_b) in reminder_a.iter().zip(reminder_b.iter()) {
        result += remainder_a * remainder_b;
    }
    
    return result;
}

#[inline(always)]
pub fn temporal_kernel_w_t(tau: f32x8) -> f32x8 {
    let half = f32x8::splat(0.5);
    let zero = f32x8::ZERO;
    let one = f32x8::splat(1.0);
    
    let tau_shifted = tau - half;
    let t_sq = tau_shifted * tau_shifted;
    
    let inner = (one - t_sq).fast_max(zero);
    let w_poly6 = inner * inner * inner;
    
    let w_t = w_poly6 * f32x8::splat(2.1875);
    
    let mask = tau.cmp_le(half);
    mask.blend(w_t, zero)
}

#[inline(always)]
pub fn gather_f32x8(slice: &[f32], indices: u32x8) -> f32x8 {
    let indexes = indices.as_array_ref();
    unsafe {
        return f32x8::from([
            *slice.get_unchecked(*indexes.get_unchecked(0) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(1) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(2) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(3) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(4) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(5) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(6) as usize),
            *slice.get_unchecked(*indexes.get_unchecked(7) as usize),
        ]);
    }
}

#[inline(always)]
pub fn scatter_f32x8(slice: &mut [f32], values: f32x8, indices: u32x8) {
    let indexes_8= indices.as_array_ref();
    let values_8= values.as_array_ref();
    unsafe {
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(0) as usize) = *values_8.get_unchecked(0);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(1) as usize) = *values_8.get_unchecked(1);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(2) as usize) = *values_8.get_unchecked(2);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(3) as usize) = *values_8.get_unchecked(3);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(4) as usize) = *values_8.get_unchecked(4);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(5) as usize) = *values_8.get_unchecked(5);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(6) as usize) = *values_8.get_unchecked(6);
        *slice.get_unchecked_mut(*indexes_8.get_unchecked(7) as usize) = *values_8.get_unchecked(7);
    }
}

#[inline(always)]
pub fn gather_ptr_f32x8(error_ptr: SendPtr<f32>, indices: u32x8) -> f32x8 {
    let indexes = indices.as_array_ref();
    unsafe {
        f32x8::from([
            *error_ptr.add(*indexes.get_unchecked(0) as usize),
            *error_ptr.add(*indexes.get_unchecked(1) as usize),
            *error_ptr.add(*indexes.get_unchecked(2) as usize),
            *error_ptr.add(*indexes.get_unchecked(3) as usize),
            *error_ptr.add(*indexes.get_unchecked(4) as usize),
            *error_ptr.add(*indexes.get_unchecked(5) as usize),
            *error_ptr.add(*indexes.get_unchecked(6) as usize),
            *error_ptr.add(*indexes.get_unchecked(7) as usize),
        ])
    }
}

#[inline(always)]
pub fn scatter_ptr_f32x8(error_ptr: SendPtr<f32>, values: f32x8, indices: u32x8) {
    let indexes = indices.as_array_ref();
    let vals = values.as_array_ref();
    unsafe {
        *error_ptr.add(*indexes.get_unchecked(0) as usize) = *vals.get_unchecked(0);
        *error_ptr.add(*indexes.get_unchecked(1) as usize) = *vals.get_unchecked(1);
        *error_ptr.add(*indexes.get_unchecked(2) as usize) = *vals.get_unchecked(2);
        *error_ptr.add(*indexes.get_unchecked(3) as usize) = *vals.get_unchecked(3);
        *error_ptr.add(*indexes.get_unchecked(4) as usize) = *vals.get_unchecked(4);
        *error_ptr.add(*indexes.get_unchecked(5) as usize) = *vals.get_unchecked(5);
        *error_ptr.add(*indexes.get_unchecked(6) as usize) = *vals.get_unchecked(6);
        *error_ptr.add(*indexes.get_unchecked(7) as usize) = *vals.get_unchecked(7);
    }
}

#[inline(always)]
pub fn bilinear_interpolate(
    weight_bottom_left: f32x8,
    weight_bottom_right: f32x8,
    weight_top_left: f32x8,
    weight_top_right: f32x8,
    velocity_bottom_left: f32x8,
    velocity_bottom_right: f32x8,
    velocity_top_left: f32x8, 
    velocity_top_right: f32x8
) -> f32x8 {
    let mut velocity = f32x8::ZERO;
    
    velocity += velocity_bottom_left * weight_bottom_left;
    velocity += velocity_bottom_right * weight_bottom_right;
    velocity += velocity_top_left * weight_top_left;
    velocity += velocity_top_right * weight_top_right;

    return velocity;
}

#[inline(always)]
pub fn expand_bits(mut v: u32x8) -> u32x8 {
    v = v & u32x8::splat(0x0000FFFF);
    v = (v | (v << 8)) & u32x8::splat(0x00FF00FF);
    v = (v | (v << 4)) & u32x8::splat(0x0F0F0F0F);
    v = (v | (v << 2)) & u32x8::splat(0x33333333);
    v = (v | (v << 1)) & u32x8::splat(0x55555555);
    v
}

#[inline(always)]
pub fn morton_code(x: u32x8, y: u32x8) -> u32x8 {
    expand_bits(x) | (expand_bits(y) << 1)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SortableTuple {
    pub key: u32,
    pub val: u32,
}

impl SortableTuple {
    pub fn new(key: u32, val: u32) -> Self {
        Self {
            key: key,
            val: val,
        }
    }
}

impl RadixKey for SortableTuple {
    const LEVELS: usize = 4;

    #[inline]
    fn get_level(&self, level: usize) -> u8 {
        (self.key >> (level * 8)) as u8
    }
}

#[derive(Copy, Clone)]
pub struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline(always)]
    pub unsafe fn add(self, count: usize) -> *mut T {
        unsafe {
            self.0.add(count)
        }
    }
}

#[derive(Default)]
pub struct MultiGridLevel {
    pub width: usize,
    pub height: usize,
    pub cells: usize,
    
    pub type_fluid: Vec<u64>,
    pub fluid_indices: Vec<u32>,
    pub red_indices: Vec<u32>,
    pub black_indices: Vec<u32>,

    pub diag_laplacian: Vec<f32>,
    pub inv_diag_laplacian: Vec<f32>,
    pub plus_x_laplacian: Vec<f32>,
    pub plus_y_laplacian: Vec<f32>,

    pub residual: Vec<f32>,
    pub error: Vec<f32>,
}


impl MultiGridLevel {
    pub fn new(width: usize, height: usize) -> Self {
        let mut multigrid = MultiGridLevel::default(); 

        let cells = width * height;
        
        multigrid.width = width;
        multigrid.height = height;
        multigrid.cells = cells;
        multigrid.type_fluid.resize(cells.div_ceil(64), 0);
        multigrid.fluid_indices.reserve(cells.div_ceil(8) * 8);
        multigrid.red_indices.reserve(cells.div_ceil(2).div_ceil(8) * 8);
        multigrid.black_indices.reserve(cells.div_ceil(2).div_ceil(8) * 8);
        multigrid.diag_laplacian.resize(cells.div_ceil(8) * 8, 0.0);
        multigrid.inv_diag_laplacian.resize(cells.div_ceil(8) * 8, 0.0);
        multigrid.plus_x_laplacian.resize(cells.div_ceil(8) * 8, 0.0);
        multigrid.plus_y_laplacian.resize(cells.div_ceil(8) * 8, 0.0);

        multigrid.residual.resize(cells.div_ceil(8) * 8, 0.0);
        multigrid.error.resize(cells.div_ceil(8) * 8, 0.0);

        return multigrid;
    }

    #[inline(always)]
    pub fn is_fluid_index(&self, index: usize) -> u64 {
        unsafe {
            let fluid = *self.type_fluid.get_unchecked(index >> 6);
            return 1 & (fluid >> (index & 63));
        }
    }

    pub fn smooth_level(&mut self, iterations: usize, reverse: bool) {


        let width = self.width as usize;
        let error_ptr = SendPtr(self.error.as_mut_ptr());

        let inv_diag_lap = self.inv_diag_laplacian.as_slice();
        let plus_x = self.plus_x_laplacian.as_slice();
        let plus_y = self.plus_y_laplacian.as_slice();
        let residual = self.residual.as_slice();

        let process_chunk = |chunk: &[u32]| {
            for &index in chunk {
                let cell_index = index as usize;

                unsafe {
                    let inv_diag = *inv_diag_lap.get_unchecked(cell_index);
                    
                    if inv_diag != 0.0 {
                        let mut sum = *residual.get_unchecked(cell_index);

                        let px = *plus_x.get_unchecked(cell_index);
                        if px != 0.0 { sum = (-px).mul_add(*error_ptr.add(cell_index + 1), sum); }

                        let py = *plus_y.get_unchecked(cell_index);
                        if py != 0.0 { sum = (-py).mul_add(*error_ptr.add(cell_index + width), sum); }

                        if cell_index > 0 {
                            sum = (-plus_x.get_unchecked(cell_index - 1)).mul_add(*error_ptr.add(cell_index - 1), sum);
                        }

                        if cell_index >= width {
                            sum = (-plus_y.get_unchecked(cell_index - width)).mul_add(*error_ptr.add(cell_index - width), sum);
                        }
                        
                        *error_ptr.add(cell_index) = sum * inv_diag;
                    }
                }
            }
        };

        let chunk_size = 4096;
        let use_parallel = self.red_indices.len() > chunk_size;

        for _ in 0..iterations {
            if reverse {
                if use_parallel {
                    self.black_indices.par_chunks(chunk_size).for_each(process_chunk);
                    self.red_indices.par_chunks(chunk_size).for_each(process_chunk);
                } else {
                    self.black_indices.chunks(chunk_size).for_each(process_chunk);
                    self.red_indices.chunks(chunk_size).for_each(process_chunk);
                }
            } else {
                if use_parallel {
                    self.red_indices.par_chunks(chunk_size).for_each(process_chunk);
                    self.black_indices.par_chunks(chunk_size).for_each(process_chunk);
                } else {
                    self.red_indices.chunks(chunk_size).for_each(process_chunk);
                    self.black_indices.chunks(chunk_size).for_each(process_chunk);
                }
            }
        }
    }

    pub fn restrict(&self, coarse: &mut Self) {
        coarse.error.fill(0.0);

        let residual_ptr = SendPtr(coarse.residual.as_mut_ptr());
        
        let process = |&coarse_index| {
            let coarse_x = coarse_index % coarse.width as u32;
            let coarse_y = coarse_index / coarse.width as u32; 

            let mut sum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let fine_x = coarse_x * 2 + dx;
                    let fine_y = coarse_y * 2 + dy;

                    let fine_index = (fine_y * self.width as u32 + fine_x) as usize;
                    if self.is_fluid_index(fine_index) == 1 {
                        unsafe {
                            let b = *self.residual.get_unchecked(fine_index);

                            let mut ax = *self.diag_laplacian.get_unchecked(fine_index) * *self.error.get_unchecked(fine_index);
                            
                            let px = *self.plus_x_laplacian.get_unchecked(fine_index);
                            if px != 0.0 {
                                ax += px * *self.error.get_unchecked(fine_index + 1);
                            }

                            let py = *self.plus_y_laplacian.get_unchecked(fine_index);
                            if py != 0.0 {
                                ax += py * *self.error.get_unchecked(fine_index + self.width);
                            }
                            
                            if fine_x > 0 {
                                ax += *self.plus_x_laplacian.get_unchecked(fine_index - 1) * *self.error.get_unchecked(fine_index - 1);
                            }
                            
                            if fine_y > 0 {
                                ax += *self.plus_y_laplacian.get_unchecked(fine_index - self.width) * *self.error.get_unchecked(fine_index - self.width);
                            }

                            sum += b - ax;
                        }
                    }
                }
            }

            unsafe {
                *residual_ptr.add(coarse_index as usize) = sum;
            }
        };

        if coarse.fluid_indices.len() > 4096 {
            coarse.fluid_indices.par_iter().with_min_len(4096).for_each(process);
        } else {
            coarse.fluid_indices.iter().for_each(process);
        }
    }

    pub fn prolongate(&mut self, coarse: &Self) {
        let error_ptr = SendPtr(self.error.as_mut_ptr());

        let process = |&fine_index| {
            let coarse_x = (fine_index % self.width as u32) / 2;
            let coarse_y = (fine_index / self.width as u32) / 2;
            let coarse_index = (coarse_y * coarse.width as u32 + coarse_x) as usize;

            unsafe {
                *error_ptr.add(fine_index as usize) += coarse.error[coarse_index];
            }
        };

        if self.fluid_indices.len() > 4096 {
            self.fluid_indices.par_iter().with_min_len(4096).for_each(process);
        } else {
            self.fluid_indices.iter().for_each(process);
        }
    }
}

#[derive(Default, Clone)]
pub struct WorldProperties {
    pub gravity: f32,
    pub border_damping: f32,
    pub cfl: f32, // Recommended value is 5.0
    pub solve_error: f32, // Recommended value is 1e-5
}

#[derive(Default)]
pub struct Apic {
    pub world_properties: WorldProperties,
    pub width: u32,
    pub height: u32,
    pub cells: u32,
    pub cell_size: f32,
    pub num_chunks: u32,
    pub num_particles: u32,

    pub external_force_u: Vec<f32>,
    pub external_force_v: Vec<f32>,
    pub post_force_u: Vec<f32>,
    pub post_force_v: Vec<f32>,

    pub mac_density: Vec<f32>,
    pub smoothed_density: Vec<f32>,
    pub old_density: Vec<f32>,

    pub mac_pressure_grid: Vec<f32>,
    pub mac_grid_u: Vec<f32>,
    pub mac_grid_v: Vec<f32>,
    pub mac_weight_u: Vec<f32>,
    pub mac_weight_v: Vec<f32>,
    pub mac_type_obstacle: Vec<u64>,

    pub mac_valid_u: Vec<bool>,
    pub mac_valid_v: Vec<bool>,

    pub old_grid_u: Vec<f32>,
    pub old_grid_v: Vec<f32>,
    pub next_valid_u: Vec<bool>,
    pub next_valid_v: Vec<bool>,

    pub search_vector: Vec<f32>,
    pub matrix_times_search: Vec<f32>,
    pub w_vector: Vec<f32>,
    //pub part_lookup: Vec<(u32, u32)>,

    pub part_positions: Vec<Vec2x8>,
    pub part_velocities: Vec<Vec2x8>,
    pub part_c_u: Vec<Vec2x8>,
    pub part_c_v: Vec<Vec2x8>,
    pub part_sort: Vec<SortableTuple>,
    pub part_positions_sort: Vec<Vec2x8>,
    pub part_velocities_sort: Vec<Vec2x8>,
    pub part_c_u_sort: Vec<Vec2x8>,
    pub part_c_v_sort: Vec<Vec2x8>,
    pub part_time_residual: Vec<f32x8>,
    pub part_time_residual_sort: Vec<f32x8>,

    pub fluid_cells: u32,

    pub base_grid: MultiGridLevel,
    pub multigrid_levels: Vec<MultiGridLevel>,

    pub timestamp: u32,
    pub prev_deltatime: f32,
}

impl Apic {
    pub fn new(width: u32, height: u32, cell_size: f32, world_properties: WorldProperties) -> Apic {
        let mut flip = Apic {
            base_grid: MultiGridLevel::new(width as usize, height as usize),
            ..Default::default()
        };

        flip.world_properties = world_properties;

        flip.width = width;
        flip.height = height;
        flip.cell_size = cell_size;

        flip.cells = width * height;

        flip.mac_pressure_grid.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);
        flip.mac_grid_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.mac_grid_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.mac_weight_u.resize((flip.cells + flip.height) as usize, 0.0); 
        flip.mac_valid_u.resize((flip.cells + flip.height) as usize, false);
        flip.mac_weight_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.mac_valid_v.resize((flip.cells + flip.width) as usize, false);
        flip.mac_type_obstacle.resize(flip.cells.div_ceil(64) as usize, 0);

        flip.old_grid_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.old_grid_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.next_valid_u.resize((flip.cells + flip.height) as usize, false);
        flip.next_valid_v.resize((flip.cells + flip.width) as usize, false);

        flip.mac_density.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);
        flip.smoothed_density.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);
        flip.old_density.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);

        flip.search_vector.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);
        flip.matrix_times_search.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);
        flip.w_vector.resize((flip.cells.div_ceil(8) * 8) as usize, 0.0);

        flip.part_positions.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_velocities.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_u.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_v.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_positions_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_velocities_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_u_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_v_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        //flip.part_lookup.resize((flip.cells / 2) as usize, (0, 0));
        flip.part_sort.resize((flip.cells / 2) as usize, SortableTuple::new(0, 0));
        flip.num_chunks = flip.cells / 2;

        flip.prev_deltatime = 1.0 / 60.0;

        flip.external_force_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.external_force_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.post_force_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.post_force_v.resize((flip.cells + flip.width) as usize, 0.0);

        let mut initial_positions = Vec::new();
        for y in 0..height {
            for x in 0..width {
                if x == 0 || x == width - 1 || y == 0 || y == height - 1 {
                    let index = y * width + x;
                    flip.mac_type_obstacle[(index / 64) as usize] |= 1 << (index % 64);
                } else {
                    if x > 4 && x < width - 4 && y > 4 && y < height / 2 {
                        for py in 0..3 {
                            for px in 0..3 {
                                let pos_x = (x as f32 + (px as f32 + 0.5) / 3.0) * cell_size;
                                let pos_y = (y as f32 + (py as f32 + 0.5) / 3.0) * cell_size;
                                initial_positions.push(Vec2::new(pos_x, pos_y));
                            }
                        }
                    }
                }
            }
        }

        flip.num_particles = initial_positions.len() as u32;
        if flip.num_particles == 0 {
            flip.num_chunks = 0;
            return flip;
        }

        flip.num_chunks = ((flip.num_particles + 7) / 8) as u32;
        flip.part_positions.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_velocities.resize(flip.num_chunks as usize, Vec2x8::zero());

        flip.part_c_u.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_c_v.resize(flip.num_chunks as usize, Vec2x8::zero());

        flip.part_positions_sort.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_velocities_sort.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_c_u_sort.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_c_v_sort.resize(flip.num_chunks as usize, Vec2x8::zero());
        flip.part_time_residual.resize(flip.num_chunks as usize, f32x8::ZERO);
        flip.part_time_residual_sort.resize(flip.num_chunks as usize, f32x8::ZERO);

        let max_dim = flip.width.max(flip.height).next_power_of_two();
        //let lookup_size = (max_dim * max_dim) as usize;
        //flip.part_lookup.resize(lookup_size, (0, 0));
        flip.part_sort.resize(flip.num_particles as usize, SortableTuple::new(0, 0));

        for (i, pos) in initial_positions.iter().enumerate() {
            let chunk_idx = i / 8;
            let lane = i % 8;
            let mut x_arr = *flip.part_positions[chunk_idx].x.as_array_ref();
            let mut y_arr = *flip.part_positions[chunk_idx].y.as_array_ref();
            x_arr[lane] = pos.x;
            y_arr[lane] = pos.y;
            flip.part_positions[chunk_idx].x = f32x8::from(x_arr);
            flip.part_positions[chunk_idx].y = f32x8::from(y_arr);
        }

        if flip.num_particles % 8 != 0 {
            let last_valid = *initial_positions.last().unwrap();
            let chunk_idx = flip.num_particles / 8;
            let mut x_arr = *flip.part_positions[chunk_idx as usize].x.as_array_ref();
            let mut y_arr = *flip.part_positions[chunk_idx as usize].y.as_array_ref();
            for lane in (flip.num_particles % 8) as usize..8 {
                x_arr[lane] = last_valid.x;
                y_arr[lane] = last_valid.y;
            }
            flip.part_positions[chunk_idx as usize].x = f32x8::from(x_arr);
            flip.part_positions[chunk_idx as usize].y = f32x8::from(y_arr);
        }

        let mut current_width = flip.width as usize;
        let mut current_height = flip.height as usize;

        while current_width % 2 == 0 && current_height % 2 == 0 && current_width > 4 && current_height > 4 {
            current_width = current_width / 2;
            current_height = current_height / 2;

            flip.multigrid_levels.push(MultiGridLevel::new(current_width, current_height));
        }

        return flip;
    }

    #[inline(always)]
    pub fn random_simd(chunk_index: usize, seed: u32) -> f32x8 {
        let base_idx = (chunk_index * 8) as u32;
        let lanes = u32x8::from([
            base_idx,
            base_idx + 1,
            base_idx + 2,
            base_idx + 3,
            base_idx + 4,
            base_idx + 5,
            base_idx + 6,
            base_idx + 7,
        ]);

        let mut mix = lanes ^ u32x8::splat(seed);

        mix ^= mix >> 16;
        mix = mix * u32x8::splat(0x7feb352d);
        mix ^= mix >> 15;
        mix = mix * u32x8::splat(0x846ca68b);
        mix ^= mix >> 16;

        let mantissa_mask = u32x8::splat(0x007FFFFF);
        let exponent_one = u32x8::splat(0x3F800000);
        let float_bits = (mix & mantissa_mask) | exponent_one;

        let random_number: f32x8 = bytemuck::cast(float_bits);

        return random_number - f32x8::splat(1.5);
    }

    #[inline(always)]
    fn u_index(&self, i: i32, j: i32) -> u32 {
        let clamped_i: u32 = i.min(self.width as i32).max(0) as u32;
        let clamped_j: u32 = j.min((self.height - 1) as i32).max(0) as u32;
        return (clamped_j * (self.width as u32 + 1)) + clamped_i;
    }


    #[inline(always)]
    fn v_index(&self, i: i32, j: i32) -> u32 {
        let clamped_i: u32 = i.min((self.width - 1) as i32).max(0) as u32;
        let clamped_j: u32 = j.min(self.height as i32).max(0) as u32;
        
        return (clamped_j * self.width as u32) + clamped_i; 
    }

    #[inline(always)]
    pub fn is_obstacle(&self, i: i32, j: i32) -> u64 {
        if i < 0 || i >= self.width as i32 || j < 0 || j >= self.height as i32 {
            return 1;
        }

        let index = ((j * self.width as i32) + i) as u32;
        let obstacle = self.mac_type_obstacle[(index / 64) as usize];

        return 1 & (obstacle >> (index % 64));
    }

    #[inline(always)]
    pub fn is_fluid(&self, i: i32, j: i32) -> u64 {
        if i < 0 || i >= self.width as i32 || j < 0 || j >= self.height as i32 {
            return 0;
        }

        let index = ((j * self.width as i32) + i) as usize;
        return self.base_grid.is_fluid_index(index);
    }

    #[inline(always)]
    pub fn is_fluid_index(&self, index: usize) -> u64 {
       return self.base_grid.is_fluid_index(index);
    }

    #[inline(always)]
    fn u_index_simd(width: u32, height: u32, i: i32x8, j: i32x8) -> u32x8 {
        let clamped_i: u32x8 = bytemuck::cast(i.min(i32x8::splat(width as i32)).max(i32x8::ZERO));
        let clamped_j: u32x8 = bytemuck::cast(j.min(i32x8::splat((height - 1) as i32)).max(i32x8::ZERO));
        return (clamped_j * u32x8::splat(width as u32 + 1)) + clamped_i;
    }

    #[inline(always)]
    fn v_index_simd(width: u32, height: u32, i: i32x8, j: i32x8) -> u32x8 {
        let clamped_i: u32x8 = bytemuck::cast(i.min(i32x8::splat((width - 1) as i32)).max(i32x8::ZERO));
        let clamped_j: u32x8 = bytemuck::cast(j.min(i32x8::splat(height as i32)).max(i32x8::ZERO));
        return (clamped_j * u32x8::splat(width as u32)) + clamped_i; 
    }

    pub fn update_spatial_lookup(&mut self) {
        let max_x = f32x8::splat(self.width.saturating_sub(1) as f32);
        let max_y = f32x8::splat(self.height.saturating_sub(1) as f32);

        for chunk_index in 0..self.num_chunks as usize {
            let positions = self.part_positions[chunk_index];
            let grid_space_positions = positions / f32x8::splat(self.cell_size);

            let grid_x: u32x8 = bytemuck::cast(grid_space_positions.x.max(f32x8::ZERO).min(max_x).fast_trunc_int());
            let grid_y: u32x8 = bytemuck::cast(grid_space_positions.y.max(f32x8::ZERO).min(max_y).fast_trunc_int());

            let grid_indexes: u32x8 = morton_code(grid_x, grid_y);

            let active_lanes = if chunk_index == self.num_chunks as usize - 1 && self.num_particles % 8 != 0 {
                self.num_particles % 8
            } else {
                8
            };

            for lane in 0..active_lanes as usize {
                let particle_index = (chunk_index * 8) + lane;
                self.part_sort[particle_index].key = grid_indexes.as_array_ref()[lane];
                self.part_sort[particle_index].val = particle_index as u32;
            }
        }

        //self.part_lookup.fill((u32::MAX, u32::MAX));
        self.part_sort[..self.num_particles as usize].radix_sort_unstable();

        let mut position_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut velocity_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut c_u_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut c_v_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut t_res_buffer: [f32; 8] = [0f32; 8];

        //let mut previous_hash: u32 = u32::MAX;
        for (index, tuple) in self.part_sort.iter().enumerate() {

            let hash = tuple.key;
            let payload = tuple.val;

            let buffer_index = index & 7;

            let chunk_index = (payload >> 3) as usize;
            let lane = (payload & 7) as usize; 

            position_buffer[0][buffer_index] = self.part_positions[chunk_index].x.as_array_ref()[lane];
            position_buffer[1][buffer_index] = self.part_positions[chunk_index].y.as_array_ref()[lane];
            velocity_buffer[0][buffer_index] = self.part_velocities[chunk_index].x.as_array_ref()[lane];
            velocity_buffer[1][buffer_index] = self.part_velocities[chunk_index].y.as_array_ref()[lane];
            c_u_buffer[0][buffer_index] = self.part_c_u[chunk_index].x.as_array_ref()[lane];
            c_u_buffer[1][buffer_index] = self.part_c_u[chunk_index].y.as_array_ref()[lane];
            c_v_buffer[0][buffer_index] = self.part_c_v[chunk_index].x.as_array_ref()[lane];
            c_v_buffer[1][buffer_index] = self.part_c_v[chunk_index].y.as_array_ref()[lane];
            t_res_buffer[buffer_index] = self.part_time_residual[chunk_index].as_array_ref()[lane];

            if buffer_index == 7 {
                self.part_positions_sort[index / 8] = Vec2x8 { x: position_buffer[0].into(), y: position_buffer[1].into() };
                self.part_velocities_sort[index / 8] = Vec2x8 { x: velocity_buffer[0].into(), y: velocity_buffer[1].into() };
                self.part_c_u_sort[index / 8] = Vec2x8 { x: c_u_buffer[0].into(), y: c_u_buffer[1].into() };
                self.part_c_v_sort[index / 8] = Vec2x8 { x: c_v_buffer[0].into(), y: c_v_buffer[1].into() };
                self.part_time_residual_sort[index / 8] = f32x8::from(t_res_buffer);
            }

            /*
            if previous_hash != hash {
                if previous_hash != u32::MAX {
                    self.part_lookup[previous_hash as usize].1 = index as u32;
                }

                self.part_lookup[hash as usize].0 = index as u32;
                previous_hash = hash;
            }
            */
        }
        /*
        if previous_hash != u32::MAX {
            self.part_lookup[previous_hash as usize].1 = self.num_particles;
        }
        */

        mem::swap(&mut self.part_positions, &mut self.part_positions_sort);
        mem::swap(&mut self.part_velocities, &mut self.part_velocities_sort);
        mem::swap(&mut self.part_c_u, &mut self.part_c_u_sort);
        mem::swap(&mut self.part_c_v, &mut self.part_c_v_sort);
        mem::swap(&mut self.part_time_residual, &mut self.part_time_residual_sort);
    }

    #[inline(always)]
    fn get_u_grid(width: u32, height: u32, grid_space_positions: Vec2x8) -> (f32x8, f32x8, f32x8, f32x8, u32x8, u32x8, u32x8, u32x8, f32x8, f32x8) {
        let half = f32x8::splat(0.5);
        let one = f32x8::splat(1.0);

        let u_grid_x = grid_space_positions.x;
        let u_grid_y = grid_space_positions.y - half;

        let u_base_x = u_grid_x.floor();
        let u_base_y = u_grid_y.floor();

        let u_base_x_index = u_base_x.fast_trunc_int();
        let u_base_y_index = u_base_y.fast_trunc_int();

        let u_index_bottom_left = Apic::u_index_simd(width, height, u_base_x_index, u_base_y_index);
        let u_index_bottom_right = Apic::u_index_simd(width, height, u_base_x_index + 1, u_base_y_index);
        let u_index_top_left = Apic::u_index_simd(width, height, u_base_x_index, u_base_y_index + 1);
        let u_index_top_right = Apic::u_index_simd(width, height, u_base_x_index + 1, u_base_y_index + 1);

        let u_tx = u_grid_x - u_base_x;
        let u_ty = u_grid_y - u_base_y;
        
        let u_weight_bottom_left = (one - u_tx) * (one - u_ty);
        let u_weight_bottom_right = u_tx * (one - u_ty);
        let u_weight_top_left = (one - u_tx) * u_ty;
        let u_weight_top_right = u_tx * u_ty;

        return (
            u_weight_bottom_left,
            u_weight_bottom_right,
            u_weight_top_left,
            u_weight_top_right,
            u_index_bottom_left,
            u_index_bottom_right,
            u_index_top_left,
            u_index_top_right,
            u_tx,
            u_ty
        );
    }

    #[inline(always)]
    fn get_v_grid(width: u32, height: u32, grid_space_positions: Vec2x8) -> (f32x8, f32x8, f32x8, f32x8, u32x8, u32x8, u32x8, u32x8, f32x8, f32x8) {
        let half = f32x8::splat(0.5);
        let one = f32x8::splat(1.0);

        let v_grid_x = grid_space_positions.x - half;
        let v_grid_y = grid_space_positions.y;

        let v_base_x = v_grid_x.floor();
        let v_base_y = v_grid_y.floor();

        let v_tx = v_grid_x - v_base_x;
        let v_ty = v_grid_y - v_base_y;
        
        let v_weight_bottom_left = (one - v_tx) * (one - v_ty);
        let v_weight_bottom_right = v_tx * (one - v_ty);
        let v_weight_top_left = (one - v_tx) * v_ty;
        let v_weight_top_right = v_tx * v_ty;

        let v_base_x_index = v_base_x.fast_trunc_int();
        let v_base_y_index = v_base_y.fast_trunc_int();

        let v_index_bottom_left = Apic::v_index_simd(width, height, v_base_x_index, v_base_y_index);
        let v_index_bottom_right = Apic::v_index_simd(width, height, v_base_x_index + 1, v_base_y_index);
        let v_index_top_left = Apic::v_index_simd(width, height, v_base_x_index, v_base_y_index + 1);
        let v_index_top_right = Apic::v_index_simd(width, height, v_base_x_index + 1, v_base_y_index + 1);

        return (
            v_weight_bottom_left,
            v_weight_bottom_right,
            v_weight_top_left,
            v_weight_top_right,
            v_index_bottom_left,
            v_index_bottom_right,
            v_index_top_left,
            v_index_top_right,
            v_tx, 
            v_ty
        );
    }

    #[inline(always)]
    fn get_grid_distances(cell_size: f32, tx: f32x8, ty: f32x8) -> (f32x8, f32x8, f32x8, f32x8, f32x8, f32x8, f32x8, f32x8) {
        let dx = f32x8::splat(cell_size);
        let one = f32x8::splat(1.0);

        return (
            -tx * dx,
            -ty * dx,
            (one - tx) * dx,
            -ty * dx,
            -tx * dx,
            (one - ty) * dx,
            (one - tx) * dx,
            (one - ty) * dx
        );
    }

    pub fn transfer_particles_to_grid(&mut self, deltatime: f32, timestamp: u32) {
        self.mac_grid_u.fill(0.0);
        self.mac_grid_v.fill(0.0);
        self.mac_weight_u.fill(0.0);
        self.mac_weight_v.fill(0.0);
        self.base_grid.type_fluid.fill(0);
        self.base_grid.red_indices.clear();
        self.base_grid.black_indices.clear();
        self.base_grid.fluid_indices.clear();

        self.mac_valid_u.fill(false);
        self.mac_valid_v.fill(false);

        self.mac_density.fill(0.0);

        for chunk_index in 0..self.num_chunks as usize {
            let raw_positions = self.part_positions[chunk_index];
            let velocities = self.part_velocities[chunk_index];

            let positions = raw_positions;
            let prev_dt = f32x8::splat(self.prev_deltatime);
            let tau = -self.part_time_residual[chunk_index] / prev_dt;
            let w_t = temporal_kernel_w_t(tau);


            let c_u = self.part_c_u[chunk_index];
            let c_v = self.part_c_v[chunk_index];

            let grid_space_positions = positions / f32x8::splat(self.cell_size);
            let true_grid_space_positions = raw_positions / f32x8::splat(self.cell_size); 

            let half = f32x8::splat(0.5);
            let one = f32x8::splat(1.0);
            
            let c_grid_x = true_grid_space_positions.x - half;
            let c_grid_y = true_grid_space_positions.y - half;
            
            let c_base_x = c_grid_x.floor();
            let c_base_y = c_grid_y.floor();
            let c_tx = c_grid_x - c_base_x;
            let c_ty = c_grid_y - c_base_y;
            
            let c_weight_back_left = ((one - c_tx) * (one - c_ty)) * w_t;
            let c_weight_back_right = (c_tx * (one - c_ty)) * w_t;
            let c_weight_top_left = ((one - c_tx) * c_ty) * w_t;
            let c_weight_top_right = (c_tx * c_ty) * w_t;

            let max_x = i32x8::splat(self.width as i32 - 1);
            let max_y = i32x8::splat(self.height as i32 - 1);
            let cx0 = c_base_x.fast_trunc_int().min(max_x).max(i32x8::ZERO);
            let cx1 = (c_base_x.fast_trunc_int() + i32x8::splat(1)).min(max_x).max(i32x8::ZERO);
            let cy0 = c_base_y.fast_trunc_int().min(max_y).max(i32x8::ZERO);
            let cy1 = (c_base_y.fast_trunc_int() + i32x8::splat(1)).min(max_y).max(i32x8::ZERO);

            let width_splat = i32x8::splat(self.width as i32);
            let c_index_back_left: u32x8 = bytemuck::cast(cy0 * width_splat + cx0);
            let c_index_back_right: u32x8 = bytemuck::cast(cy0 * width_splat + cx1);
            let c_index_top_left: u32x8 = bytemuck::cast(cy1 * width_splat + cx0);
            let c_index_top_right: u32x8 = bytemuck::cast(cy1 * width_splat + cx1);

            // U GRID
            let (
                mut u_weight_bottom_left,
                mut u_weight_bottom_right,
                mut u_weight_top_left,
                mut u_weight_top_right,
                u_index_bottom_left,
                u_index_bottom_right,
                u_index_top_left,
                u_index_top_right,
                u_tx,
                u_ty,
            ) = Apic::get_u_grid(self.width, self.height, grid_space_positions);

            let (
                u_diff_bottom_left_x,
                u_diff_bottom_left_y,
                u_diff_bottom_right_x,
                u_diff_bottom_right_y,
                u_diff_top_left_x,
                u_diff_top_left_y,
                u_diff_top_right_x,
                u_diff_top_right_y
            ) = Apic::get_grid_distances(self.cell_size, u_tx, u_ty);
            
            // V GRID
            let (
                mut v_weight_bottom_left,
                mut v_weight_bottom_right,
                mut v_weight_top_left,
                mut v_weight_top_right,
                v_index_bottom_left,
                v_index_bottom_right,
                v_index_top_left,
                v_index_top_right,
                v_tx,
                v_ty,
            ) = Apic::get_v_grid(self.width, self.height, grid_space_positions);

            let (
                v_diff_bottom_left_x,
                v_diff_bottom_left_y,
                v_diff_bottom_right_x,
                v_diff_bottom_right_y,
                v_diff_top_left_x,
                v_diff_top_left_y,
                v_diff_top_right_x,
                v_diff_top_right_y
            ) = Apic::get_grid_distances(self.cell_size, v_tx, v_ty);

            u_weight_bottom_left *= w_t;
            u_weight_bottom_right *= w_t;
            u_weight_top_left *= w_t;
            u_weight_top_right *= w_t;

            v_weight_bottom_left *= w_t;
            v_weight_bottom_right *= w_t;
            v_weight_top_left *= w_t;
            v_weight_top_right *= w_t;
            

            let u_add_bl = (velocities.x + c_u.x * u_diff_bottom_left_x + c_u.y * u_diff_bottom_left_y) * u_weight_bottom_left;
            let u_add_br = (velocities.x + c_u.x * u_diff_bottom_right_x + c_u.y * u_diff_bottom_right_y) * u_weight_bottom_right;
            let u_add_tl = (velocities.x + c_u.x * u_diff_top_left_x + c_u.y * u_diff_top_left_y) * u_weight_top_left;
            let u_add_tr = (velocities.x + c_u.x * u_diff_top_right_x + c_u.y * u_diff_top_right_y) * u_weight_top_right;

            let v_add_bl = (velocities.y + c_v.x * v_diff_bottom_left_x + c_v.y * v_diff_bottom_left_y) * v_weight_bottom_left;
            let v_add_br = (velocities.y + c_v.x * v_diff_bottom_right_x + c_v.y * v_diff_bottom_right_y) * v_weight_bottom_right;
            let v_add_tl = (velocities.y + c_v.x * v_diff_top_left_x + c_v.y * v_diff_top_left_y) * v_weight_top_left;
            let v_add_tr = (velocities.y + c_v.x * v_diff_top_right_x + c_v.y * v_diff_top_right_y) * v_weight_top_right;

         
            let clamped_x = grid_space_positions.x.min(f32x8::splat(self.width as f32 - 1.0)).max(f32x8::ZERO);
            let clamped_y = grid_space_positions.y.min(f32x8::splat(self.height as f32 - 1.0)).max(f32x8::ZERO);
            let grid_fluid_indexes: u32x8 = bytemuck::cast((clamped_y.fast_trunc_int() * (self.width as i32)) + clamped_x.fast_trunc_int());

            let active_lanes = if chunk_index == self.num_chunks as usize - 1 && self.num_particles % 8 != 0 {
                self.num_particles % 8
            } else {
                8
            };

            let u_add_back_left_arr = u_add_bl.as_array_ref();
            let u_add_back_right_arr = u_add_br.as_array_ref();
            let u_add_top_left_arr = u_add_tl.as_array_ref();
            let u_add_top_right_arr = u_add_tr.as_array_ref();
            let u_wt_back_left_arr = u_weight_bottom_left.as_array_ref();
            let u_wt_back_right_arr = u_weight_bottom_right.as_array_ref();
            let u_wt_top_left_arr = u_weight_top_left.as_array_ref();
            let u_wt_top_right_arr = u_weight_top_right.as_array_ref();

            let v_add_back_left_arr = v_add_bl.as_array_ref();
            let v_add_back_right_arr = v_add_br.as_array_ref();
            let v_add_top_left_arr = v_add_tl.as_array_ref();
            let v_add_top_right_arr = v_add_tr.as_array_ref();
            let v_wt_back_left_arr = v_weight_bottom_left.as_array_ref();
            let v_wt_back_right_arr = v_weight_bottom_right.as_array_ref();
            let v_wt_top_left_arr = v_weight_top_left.as_array_ref();
            let v_wt_top_right_arr = v_weight_top_right.as_array_ref();

            let c_wt_back_left_arr = c_weight_back_left.as_array_ref();
            let c_wt_back_right_arr = c_weight_back_right.as_array_ref();
            let c_wt_top_left_arr = c_weight_top_left.as_array_ref();
            let c_wt_top_right_arr = c_weight_top_right.as_array_ref();

            let ubl_idx = u_index_bottom_left.as_array_ref();
            let ubr_idx = u_index_bottom_right.as_array_ref();
            let utl_idx = u_index_top_left.as_array_ref();
            let utr_idx = u_index_top_right.as_array_ref();

            let vbl_idx = v_index_bottom_left.as_array_ref();
            let vbr_idx = v_index_bottom_right.as_array_ref();
            let vtl_idx = v_index_top_left.as_array_ref();
            let vtr_idx = v_index_top_right.as_array_ref();

            let cbl_idx = c_index_back_left.as_array_ref();
            let cbr_idx = c_index_back_right.as_array_ref();
            let ctl_idx = c_index_top_left.as_array_ref();
            let ctr_idx = c_index_top_right.as_array_ref();

            let grid_fluid_indexes_arr = grid_fluid_indexes.as_array_ref();

            unsafe {
                for lane in 0..active_lanes as usize {
                    let grid_index = *grid_fluid_indexes_arr.get_unchecked(lane) as usize;
                    *self.base_grid.type_fluid.get_unchecked_mut(grid_index / 64) |= 1u64 << (grid_index % 64);

                    let ubl = *ubl_idx.get_unchecked(lane) as usize;
                    let ubr = *ubr_idx.get_unchecked(lane) as usize;
                    let utl = *utl_idx.get_unchecked(lane) as usize;
                    let utr = *utr_idx.get_unchecked(lane) as usize;
                    *self.mac_grid_u.get_unchecked_mut(ubl) += *u_add_back_left_arr.get_unchecked(lane);
                    *self.mac_grid_u.get_unchecked_mut(ubr) += *u_add_back_right_arr.get_unchecked(lane);
                    *self.mac_grid_u.get_unchecked_mut(utl) += *u_add_top_left_arr.get_unchecked(lane);
                    *self.mac_grid_u.get_unchecked_mut(utr) += *u_add_top_right_arr.get_unchecked(lane);
                    *self.mac_weight_u.get_unchecked_mut(ubl) += *u_wt_back_left_arr.get_unchecked(lane);
                    *self.mac_weight_u.get_unchecked_mut(ubr) += *u_wt_back_right_arr.get_unchecked(lane);
                    *self.mac_weight_u.get_unchecked_mut(utl) += *u_wt_top_left_arr.get_unchecked(lane);
                    *self.mac_weight_u.get_unchecked_mut(utr) += *u_wt_top_right_arr.get_unchecked(lane);

                    let vbl = *vbl_idx.get_unchecked(lane) as usize;
                    let vbr = *vbr_idx.get_unchecked(lane) as usize;
                    let vtl = *vtl_idx.get_unchecked(lane) as usize;
                    let vtr = *vtr_idx.get_unchecked(lane) as usize;
                    *self.mac_grid_v.get_unchecked_mut(vbl) += *v_add_back_left_arr.get_unchecked(lane);
                    *self.mac_grid_v.get_unchecked_mut(vbr) += *v_add_back_right_arr.get_unchecked(lane);
                    *self.mac_grid_v.get_unchecked_mut(vtl) += *v_add_top_left_arr.get_unchecked(lane);
                    *self.mac_grid_v.get_unchecked_mut(vtr) += *v_add_top_right_arr.get_unchecked(lane);
                    *self.mac_weight_v.get_unchecked_mut(vbl) += *v_wt_back_left_arr.get_unchecked(lane);
                    *self.mac_weight_v.get_unchecked_mut(vbr) += *v_wt_back_right_arr.get_unchecked(lane);
                    *self.mac_weight_v.get_unchecked_mut(vtl) += *v_wt_top_left_arr.get_unchecked(lane);
                    *self.mac_weight_v.get_unchecked_mut(vtr) += *v_wt_top_right_arr.get_unchecked(lane);

                    let cbl = *cbl_idx.get_unchecked(lane) as usize;
                    let cbr = *cbr_idx.get_unchecked(lane) as usize;
                    let ctl = *ctl_idx.get_unchecked(lane) as usize;
                    let ctr = *ctr_idx.get_unchecked(lane) as usize;
                    *self.mac_density.get_unchecked_mut(cbl) += *c_wt_back_left_arr.get_unchecked(lane);
                    *self.mac_density.get_unchecked_mut(cbr) += *c_wt_back_right_arr.get_unchecked(lane);
                    *self.mac_density.get_unchecked_mut(ctl) += *c_wt_top_left_arr.get_unchecked(lane);
                    *self.mac_density.get_unchecked_mut(ctr) += *c_wt_top_right_arr.get_unchecked(lane);
                }
            }
        }

        self.mac_grid_u.par_iter_mut()
            .zip(self.mac_weight_u.par_iter())
            .zip(self.mac_valid_u.par_iter_mut())
            .with_min_len(4096)
            .for_each(|((u, w), v)| {
                if *w > 0.0 {
                    *v = true;
                    *u /= *w;
                }
            });

        self.mac_grid_v.par_iter_mut()
            .zip(self.mac_weight_v.par_iter())
            .zip(self.mac_valid_v.par_iter_mut())
            .with_min_len(4096)
            .for_each(|((u, w), v)| {
                if *w > 0.0 {
                    *v = true;
                    *u /= *w;
                }
            });

        self.fluid_cells = 0;

        for i in 0..self.base_grid.type_fluid.len() {
            self.base_grid.type_fluid[i] &= !self.mac_type_obstacle[i];

            let mut fluid_mask = self.base_grid.type_fluid[i];

            while fluid_mask != 0 {
                let cell_index = (i * 64) + (fluid_mask.trailing_zeros() as usize);
                self.base_grid.fluid_indices.push(cell_index as u32);
                self.fluid_cells += 1;

                let x = cell_index % self.width as usize;
                let y = cell_index / self.width as usize;
                if (x + y) % 2 == 0 {
                    self.base_grid.red_indices.push(cell_index as u32);
                } else {
                    self.base_grid.black_indices.push(cell_index as u32);
                }

                fluid_mask &= fluid_mask - 1;
            }
        }

        let remainder = self.fluid_cells % 8;
        if remainder != 0 {
            let padding = 8 - remainder;
            let last_val = self.base_grid.fluid_indices[self.fluid_cells as usize - 1];
            for _ in 0..padding {
                self.base_grid.fluid_indices.push(last_val);
            }
        }


    }

    pub fn update_multilevel_grid(&mut self) {
        let mut width = self.width as u32;

        for grid_index in 0..self.multigrid_levels.len() {
            let (fine_grid, coarse_grid) = if grid_index == 0 {
                (&self.base_grid as &MultiGridLevel, &mut self.multigrid_levels[0])
            } else {
                let (left, right) = self.multigrid_levels.split_at_mut(grid_index);
                (&left[grid_index - 1] as &MultiGridLevel, &mut right[0])
            };

            coarse_grid.fluid_indices.clear();
            coarse_grid.red_indices.clear();
            coarse_grid.black_indices.clear();
            coarse_grid.type_fluid.fill(0);
            coarse_grid.diag_laplacian.fill(0.0);
            coarse_grid.inv_diag_laplacian.fill(0.0);
            coarse_grid.plus_x_laplacian.fill(0.0);
            coarse_grid.plus_y_laplacian.fill(0.0);

            let new_width = coarse_grid.width as u32;
            let new_height = coarse_grid.height as u32;

            for cell_index in &fine_grid.fluid_indices {
                let new_x = (cell_index % width) / 2;
                let new_y = (cell_index / width) / 2;
                
                let new_index = (new_y * new_width) + new_x;
                coarse_grid.type_fluid[(new_index / 64) as usize] |= 1u64 << (new_index % 64);
            }

            for index in 0..coarse_grid.type_fluid.len() {
                let mut fluid_mask = coarse_grid.type_fluid[index];
                while fluid_mask != 0 {
                    let cell_index = ((index * 64) + (fluid_mask.trailing_zeros() as usize)) as u32;
                    coarse_grid.fluid_indices.push(cell_index);

                    let x = cell_index % new_width;
                    let y = cell_index / new_width;

                    if (x + y) % 2 == 0 {
                        coarse_grid.red_indices.push(cell_index);
                    } else {
                        coarse_grid.black_indices.push(cell_index);
                    }
                    fluid_mask &= fluid_mask - 1;
                }
            }

            let remainder_red = coarse_grid.red_indices.len() % 8;
            if remainder_red != 0 {
                let padding = 8 - remainder_red;
                let last_val = coarse_grid.red_indices[coarse_grid.red_indices.len() - 1];
                for _ in 0..padding {
                    coarse_grid.red_indices.push(last_val);
                }
            }

            let remainder_black = coarse_grid.black_indices.len() % 8;
            if remainder_black != 0 {
                let padding = 8 - remainder_black;
                let last_val = coarse_grid.black_indices[coarse_grid.black_indices.len() - 1];
                for _ in 0..padding {
                    coarse_grid.black_indices.push(last_val);
                }
            }
            

            for cell_index in &coarse_grid.fluid_indices {
                let coarse_index = *cell_index as usize;
                let cx = coarse_index % coarse_grid.width;
                let cy = coarse_index / coarse_grid.width;
                
                let mut diag = 0.0;
                let mut plus_x = 0.0;
                let mut plus_y = 0.0;
                
                for dy in 0..2 {
                    for dx in 0..2 {
                        let fx = cx * 2 + dx;
                        let fy = cy * 2 + dy;
                        
                        if fx >= fine_grid.width || fy >= fine_grid.height { continue; }
                        
                        let fine_index = fy * fine_grid.width + fx;
                        
                        if fine_grid.is_fluid_index(fine_index) == 1 {
                            diag += fine_grid.diag_laplacian[fine_index];
                            
                            if dx == 0 && fx + 1 < fine_grid.width {
                                diag += 2.0 * fine_grid.plus_x_laplacian[fine_index];
                            }
                            if dy == 0 && fy + 1 < fine_grid.height {
                                diag += 2.0 * fine_grid.plus_y_laplacian[fine_index];
                            }
                            
                            if dx == 1 && fx + 1 < fine_grid.width {
                                plus_x += fine_grid.plus_x_laplacian[fine_index];
                            }
                            if dy == 1 && fy + 1 < fine_grid.height {
                                plus_y += fine_grid.plus_y_laplacian[fine_index];
                            }
                        }
                    }
                }
                
                let final_diag = diag * 0.5;
                coarse_grid.diag_laplacian[coarse_index] = final_diag;
                coarse_grid.inv_diag_laplacian[coarse_index] = if final_diag > 1e-6 { 1.0 / final_diag } else { 0.0 };
                coarse_grid.plus_x_laplacian[coarse_index] = plus_x * 0.5;
                coarse_grid.plus_y_laplacian[coarse_index] = plus_y * 0.5;
            }
            width = new_width;
        }
    }

    pub fn apply_external_forces(&mut self, deltatime: f32) {
        let gravity = self.world_properties.gravity * deltatime;

        for i in 0..self.mac_grid_u.len() {
            self.mac_grid_u[i] += self.external_force_u[i] * deltatime;
        }
        for i in 0..self.mac_grid_v.len() {
            self.mac_grid_v[i] += self.external_force_v[i] * deltatime;
        }
        
        for y in 0..=self.height as i32 {
            for x in 0..self.width as i32 {
                let face_index = self.v_index(x, y) as usize;
                
                let is_bottom_fluid = self.is_fluid(x, y - 1) == 1;
                let is_top_fluid = self.is_fluid(x, y) == 1;
                
                if is_bottom_fluid || is_top_fluid {
                    self.mac_grid_v[face_index] += gravity;
                }
            }
        }
    }

    #[inline(always)]
    pub fn calculate_theta(&self, rho: f32, rho_center: f32, target_density: f32) -> f32 {
        let threshold = target_density * 0.5;
        let diff = rho_center - rho;
        
        if diff.abs() <= 1e-5 { 
            return 1.0; 
        }
        
        return ((rho_center - threshold) / diff).clamp(0.1, 1.0);
    }

    pub fn build_pressure_system(&mut self) {
        self.base_grid.diag_laplacian.fill(0.0);
        self.base_grid.inv_diag_laplacian.fill(0.0);
        self.base_grid.plus_x_laplacian.fill(0.0);
        self.base_grid.plus_y_laplacian.fill(0.0);

        self.smoothed_density.copy_from_slice(&self.mac_density);
        self.old_density.copy_from_slice(&self.mac_density);
        let width = self.width as usize;

        for _ in 0..3 {
            std::mem::swap(&mut self.smoothed_density, &mut self.old_density);    
            let smoothed_ptr = SendPtr(self.smoothed_density.as_mut_ptr());
            let old_density_ptr = SendPtr(self.old_density.as_mut_ptr());

            let smooth_density_process = |cell_index: &u32| {
                let index = *cell_index as usize;
                let grid_x = index % width;
                let grid_y = index / width;
                
                unsafe {
                    let mut sum = *old_density_ptr.add(index) * 4.0;
                    let mut weight = 4.0;
                    
                    if grid_x > 0 && self.is_fluid_index(index - 1) == 1 { sum += *old_density_ptr.add(index - 1); weight += 1.0; }
                    if grid_x < width - 1 && self.is_fluid_index(index + 1) == 1 { sum += *old_density_ptr.add(index + 1); weight += 1.0; }
                    if grid_y > 0 && self.is_fluid_index(index - width) == 1 { sum += *old_density_ptr.add(index - width); weight += 1.0; }
                    if grid_y < self.height as usize - 1 && self.is_fluid_index(index + width) == 1 { sum += *old_density_ptr.add(index + width); weight += 1.0; }
                    
                    *smoothed_ptr.add(index) = sum / weight;
                }
            };

            if self.base_grid.fluid_indices.len() > 4096 {
                self.base_grid.fluid_indices[..self.fluid_cells as usize].par_iter().with_min_len(4096).for_each(smooth_density_process);
            } else {
                self.base_grid.fluid_indices[..self.fluid_cells as usize].iter().for_each(smooth_density_process);
            }
        }

        let target_density = 9.0;
        let noise_threshold = target_density * 1.15; 
        let correction_rate = 0.1;
        let cell_size = self.cell_size;

        let residual_ptr = SendPtr(self.base_grid.residual.as_mut_ptr());
        let diag_ptr = SendPtr(self.base_grid.diag_laplacian.as_mut_ptr());
        let inv_diag_ptr = SendPtr(self.base_grid.inv_diag_laplacian.as_mut_ptr());
        let px_ptr = SendPtr(self.base_grid.plus_x_laplacian.as_mut_ptr());
        let py_ptr = SendPtr(self.base_grid.plus_y_laplacian.as_mut_ptr());

        let process_laplacian = |cell_index: &u32| {
            let fluid_index = *cell_index as usize;

            let grid_x = fluid_index % width;
            let grid_y = fluid_index / width;

            unsafe {
                let mut u_left = *self.mac_grid_u.get_unchecked(self.u_index(grid_x as i32, grid_y as i32) as usize);
                let mut u_right = *self.mac_grid_u.get_unchecked(self.u_index(grid_x as i32 + 1, grid_y as i32) as usize);
                let mut v_bottom = *self.mac_grid_v.get_unchecked(self.v_index(grid_x as i32, grid_y as i32) as usize);
                let mut v_top = *self.mac_grid_v.get_unchecked(self.v_index(grid_x as i32, grid_y as i32 + 1) as usize);

                let non_obstacle_left = 1 - self.is_obstacle(grid_x as i32 - 1, grid_y as i32);
                let non_obstacle_right = 1 - self.is_obstacle(grid_x as i32 + 1, grid_y as i32);
                let non_obstacle_bottom = 1 - self.is_obstacle(grid_x as i32, grid_y as i32 - 1);
                let non_obstacle_top = 1 - self.is_obstacle(grid_x as i32, grid_y as i32 + 1);

                u_left *= non_obstacle_left as f32;
                u_right *= non_obstacle_right as f32;
                v_bottom *= non_obstacle_bottom as f32;
                v_top *= non_obstacle_top as f32;

                let divergence = (u_right - u_left) + (v_top - v_bottom);
            
                let rho_center = self.smoothed_density[fluid_index];

                let density_excess = (rho_center - noise_threshold).max(0.0).min(target_density);

                let fluid_left = self.is_fluid(grid_x as i32 - 1, grid_y as i32);
                let fluid_right = self.is_fluid(grid_x as i32 + 1, grid_y as i32);
                let fluid_bottom = self.is_fluid(grid_x as i32, grid_y as i32 - 1);
                let fluid_top = self.is_fluid(grid_x as i32, grid_y as i32 + 1);

                let mut diag = 0.0;
                let mut px = 0.0;
                let mut py = 0.0;

                if non_obstacle_left == 1 {
                    if fluid_left == 1 { 
                        diag += 1.0;
                    } else {
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index - 1], rho_center, target_density);
                    }
                }
                
                if non_obstacle_right == 1 {
                    if fluid_right == 1 { 
                        diag += 1.0; 
                        px = -1.0; 
                    } else { 
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index + 1], rho_center, target_density); 
                    }
                }

                if non_obstacle_bottom == 1 {
                    if fluid_bottom == 1 { 
                        diag += 1.0;
                    } else {
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index - width], rho_center, target_density);
                    }
                }

                if non_obstacle_top == 1 {
                    if fluid_top == 1 { 
                        diag += 1.0; 
                        py = -1.0; 
                    } else { 
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index + width], rho_center, target_density); 
                    }
                }

                *residual_ptr.add(fluid_index) = -divergence * cell_size 
                    + density_excess * correction_rate * cell_size * cell_size;
                *px_ptr.add(fluid_index) = px;
                *py_ptr.add(fluid_index) = py;
                *diag_ptr.add(fluid_index) = diag;
                *inv_diag_ptr.add(fluid_index) = if diag > 1e-6 { 1.0 / diag } else { 0.0 };
            }
        };

        if self.base_grid.fluid_indices.len() > 4096 {
            self.base_grid.fluid_indices[..self.fluid_cells as usize].par_iter().with_min_len(4096).for_each(process_laplacian);
        } else {
            self.base_grid.fluid_indices[..self.fluid_cells as usize].iter().for_each(process_laplacian);
        }
    }


    pub fn apply_preconditioner(&mut self) {
        self.base_grid.error.fill(0.0);

        self.base_grid.smooth_level(2, false);

        if self.multigrid_levels.len() > 0 {
            self.base_grid.restrict(&mut self.multigrid_levels[0]);

            let last = self.multigrid_levels.len() - 1;

            for level_index in 0..last {
                self.multigrid_levels[level_index].smooth_level(2, false);

                let (left, right) = self.multigrid_levels.split_at_mut(level_index + 1);
                left[level_index].restrict(&mut right[0]);
            }

            self.multigrid_levels[last].smooth_level(5, false);
            self.multigrid_levels[last].smooth_level(5, true);

            for level_index in (0..last).rev() {
                let (left, right) = self.multigrid_levels.split_at_mut(level_index + 1);
                left[level_index].prolongate(&right[0]);

                self.multigrid_levels[level_index].smooth_level(2, true);
            }

            self.base_grid.prolongate(&self.multigrid_levels[0]);
        }

        self.base_grid.smooth_level(2, true);
    }

    #[inline(always)]
    pub fn apply_matrix_and_dot(&self, input: &[f32], output: SendPtr<f32>) -> f32 {
        let width = self.width as usize;
        let fluid_cells = &self.base_grid.fluid_indices[..self.fluid_cells as usize];
        
        let process_cell = |&cell_index| -> f32 {
            let index = cell_index as usize;
            unsafe {
                let mut value = *self.base_grid.diag_laplacian.get_unchecked(index) * *input.get_unchecked(index);
                
                let px = *self.base_grid.plus_x_laplacian.get_unchecked(index);
                if px != 0.0 { value = px.mul_add(*input.get_unchecked(index + 1), value); }

                if index > 0 {
                    let nx = *self.base_grid.plus_x_laplacian.get_unchecked(index - 1);
                    if nx != 0.0 { value = nx.mul_add(*input.get_unchecked(index - 1), value); }
                }

                let py = *self.base_grid.plus_y_laplacian.get_unchecked(index);
                if py != 0.0 { value = py.mul_add(*input.get_unchecked(index + width), value); }

                if index >= width {
                    let ny = *self.base_grid.plus_y_laplacian.get_unchecked(index - width);
                    if ny != 0.0 { value = ny.mul_add(*input.get_unchecked(index - width), value); }
                }
                
                *output.add(index) = value;
                
                return value * *input.get_unchecked(index) 
            }
        };

        return if fluid_cells.len() > 4096 {
            fluid_cells.par_iter().with_min_len(4096).map(process_cell).sum()
        } else {
            fluid_cells.iter().map(process_cell).sum()
        };
    }
    
    pub fn solve_pcg(&mut self) {
        let fluid_cells = self.fluid_cells;

        let pressure_ptr = SendPtr(self.mac_pressure_grid.as_mut_ptr());
        let clear_pressure_process = |cell_index: &u32| {
            unsafe {
                *pressure_ptr.add(*cell_index as usize) = 0.0;
            }
        };

        if fluid_cells > 4096 {
            self.base_grid.fluid_indices[..self.fluid_cells as usize].par_iter().with_min_len(4096).for_each(clear_pressure_process);
        } else {
            self.base_grid.fluid_indices[..self.fluid_cells as usize].iter().for_each(clear_pressure_process);
        }

        self.apply_preconditioner();

        self.search_vector.copy_from_slice(&self.base_grid.error);

        let mut gamma = dot_product(&self.base_grid.residual, &self.base_grid.error);

        if gamma < SAFTY {
            return;
        }

        let max_iterations: usize = 100;
        let residual_ptr = SendPtr(self.base_grid.residual.as_mut_ptr());
        let search_ptr = SendPtr(self.search_vector.as_mut_ptr());
        let matrix_ptr = SendPtr(self.matrix_times_search.as_mut_ptr());

        for _ in 0..max_iterations {

            let delta = self.apply_matrix_and_dot(&self.search_vector, matrix_ptr);

            if delta.abs() < SAFTY {
                break;
            }

            let alpha = gamma / delta;
            let mut max_error: f32 = 0.0;

            let max_error_process = |cell_index: &u32| {
                let index = *cell_index as usize;
                unsafe {
                    *pressure_ptr.add(index) += alpha * *search_ptr.add(index);

                    let new_residual = (-alpha).mul_add(*matrix_ptr.add(index), *residual_ptr.add(index));

                    *residual_ptr.add(index) = new_residual;

                    return new_residual.abs();
                }
            };

            if fluid_cells > 4096 {
                max_error = self.base_grid.fluid_indices[..self.fluid_cells as usize].par_iter().with_min_len(4096).map(max_error_process).reduce(|| 0.0, |a, b| a.max(b));
            } else {
                max_error = self.base_grid.fluid_indices[..self.fluid_cells as usize].iter().map(max_error_process).fold(0.0, |a, b| a.max(b));
            }

            if max_error < self.world_properties.solve_error {
                break;
            }

            self.apply_preconditioner();

            let new_gamma = dot_product(&self.base_grid.residual, &self.base_grid.error);

            if new_gamma < SAFTY {
                break; 
            }

            let beta = new_gamma / gamma;
            gamma = new_gamma;
            
            self.base_grid.fluid_indices[..self.fluid_cells as usize]
                .par_iter()
                .with_min_len(4096)
                .for_each(|&cell_index| {
                let index = cell_index as usize;
                
                unsafe {
                    *search_ptr.add(index) = *self.base_grid.error.get_unchecked(index) + (beta * *self.search_vector.get_unchecked(index));
                }
            });
        }
    }

    pub fn apply_pressure_gradient(&mut self) {
        let cell_size = self.cell_size;
        let target_density = 9.0;
        let inv_cell_size = 1.0 / cell_size;

        let grid_u_ptr = SendPtr(self.mac_grid_u.as_mut_ptr());
        let valid_u_ptr = SendPtr(self.mac_valid_u.as_mut_ptr() as *mut u8);

        (0..self.height as i32).into_par_iter().for_each(|y| {
            for x in 0..=self.width as i32 {
                let face_index = self.u_index(x, y) as usize;

                let left_cell_x = x - 1;

                let left_is_obstacle = self.is_obstacle(left_cell_x, y) == 1;
                let right_is_obstacle = self.is_obstacle(x, y) == 1;

                if left_is_obstacle != right_is_obstacle {
                    unsafe {
                        *grid_u_ptr.add(face_index) = 0.0;
                        *valid_u_ptr.add(face_index) = 1;
                    }
                    continue;
                }

                let left_is_fluid = self.is_fluid(left_cell_x, y) == 1;
                let right_is_fluid = self.is_fluid(x, y) == 1;
                
                let mut pressure_difference = 0.0;
                
                let left_cell_index = ((y * self.width as i32) + left_cell_x) as usize;
                let right_cell_index = ((y * self.width as i32) + x) as usize;

                if left_is_fluid && right_is_fluid {
                    pressure_difference = self.mac_pressure_grid[right_cell_index] - self.mac_pressure_grid[left_cell_index];
                } else if left_is_fluid {
                    let theta = self.calculate_theta(self.smoothed_density[right_cell_index], self.smoothed_density[left_cell_index], target_density);
                    pressure_difference = -self.mac_pressure_grid[left_cell_index] / theta;
                } else if right_is_fluid {
                    let theta = self.calculate_theta(self.smoothed_density[left_cell_index], self.smoothed_density[right_cell_index], target_density);
                    pressure_difference = self.mac_pressure_grid[right_cell_index] / theta;
                }

                unsafe {
                    *grid_u_ptr.add(face_index) -= pressure_difference * inv_cell_size;
                }

            }
        });

        let grid_v_ptr = SendPtr(self.mac_grid_v.as_mut_ptr());
        let valid_v_ptr = SendPtr(self.mac_valid_v.as_mut_ptr() as *mut u8);
        
        (0..=self.height as i32).into_par_iter().for_each(|y| {
            for x in 0..self.width as i32 {

                let face_index = self.v_index(x, y) as usize;

                let bottom_cell_y = y - 1;

                let bottom_is_obstacle = self.is_obstacle(x, bottom_cell_y) == 1;
                let top_is_obstacle = self.is_obstacle(x, y) == 1;

                if bottom_is_obstacle != top_is_obstacle {
                    unsafe {
                        *grid_v_ptr.add(face_index) = 0.0;
                        *valid_v_ptr.add(face_index) = 1;
                    }
                    continue;
                }

                let bottom_is_fluid = self.is_fluid(x, bottom_cell_y) == 1;
                let top_is_fluid = self.is_fluid(x, y) == 1;
                
                let mut pressure_difference = 0.0;
                
                let bottom_cell_index = ((bottom_cell_y * self.width as i32) + x) as usize;
                let top_cell_index = ((y * self.width as i32) + x) as usize;

                if bottom_is_fluid && top_is_fluid {
                    pressure_difference = self.mac_pressure_grid[top_cell_index] - self.mac_pressure_grid[bottom_cell_index];
                } else if bottom_is_fluid {
                    let theta = self.calculate_theta(self.smoothed_density[top_cell_index], self.smoothed_density[bottom_cell_index], target_density);
                    pressure_difference = -self.mac_pressure_grid[bottom_cell_index] / theta;
                } else if top_is_fluid {
                    let theta = self.calculate_theta(self.smoothed_density[bottom_cell_index], self.smoothed_density[top_cell_index], target_density);
                    pressure_difference = self.mac_pressure_grid[top_cell_index] / theta;
                }

                unsafe {
                    *grid_v_ptr.add(face_index) -= pressure_difference * inv_cell_size;
                }
            }
        });
    }

    pub fn transfer_grid_to_particles_and_advect(&mut self, deltatime: f32) {
        let deltatime = f32x8::splat(deltatime);

        let min_x = f32x8::splat(1.001 * self.cell_size);
        let max_x = f32x8::splat((self.width as f32 - 1.001) * self.cell_size);
        let min_y = f32x8::splat(1.001 * self.cell_size);
        let max_y = f32x8::splat((self.height as f32 - 1.001) * self.cell_size);

        let zero = f32x8::ZERO;

        let width = self.width;
        let height = self.height;
        let cell_size = self.cell_size;
        let timestamp = self.timestamp;
        let border_damping = f32x8::splat(-self.world_properties.border_damping);
        
        let mac_grid_u = &self.mac_grid_u;
        let mac_grid_v = &self.mac_grid_v;
        
        self.part_positions.par_iter_mut()
            .zip(self.part_velocities.par_iter_mut())
            .zip(self.part_c_u.par_iter_mut())
            .zip(self.part_c_v.par_iter_mut())
            .zip(self.part_time_residual.par_iter_mut())
            .enumerate()
            .for_each(|(chunk_index, ((((positions, velocity), c_u_org), c_v_org), residual_org))| {

            let grid_space_positions = *positions / f32x8::splat(cell_size);

            // U GRID
            let (
                u_weight_bottom_left,
                u_weight_bottom_right,
                u_weight_top_left,
                u_weight_top_right,
                u_index_bottom_left,
                u_index_bottom_right,
                u_index_top_left,
                u_index_top_right,
                u_tx,
                u_ty,
            ) = Apic::get_u_grid(width, height, grid_space_positions);

            let (
                u_diff_bottom_left_x,
                u_diff_bottom_left_y,
                u_diff_bottom_right_x,
                u_diff_bottom_right_y,
                u_diff_top_left_x,
                u_diff_top_left_y,
                u_diff_top_right_x,
                u_diff_top_right_y
            ) = Apic::get_grid_distances(cell_size, u_tx, u_ty);
            
            // V GRID
            let (
                v_weight_bottom_left,
                v_weight_bottom_right,
                v_weight_top_left,
                v_weight_top_right,
                v_index_bottom_left,
                v_index_bottom_right,
                v_index_top_left,
                v_index_top_right,
                v_tx,
                v_ty,
            ) = Apic::get_v_grid(width, height, grid_space_positions);

            let (
                v_diff_bottom_left_x,
                v_diff_bottom_left_y,
                v_diff_bottom_right_x,
                v_diff_bottom_right_y,
                v_diff_top_left_x,
                v_diff_top_left_y,
                v_diff_top_right_x,
                v_diff_top_right_y
            ) = Apic::get_grid_distances(cell_size, v_tx, v_ty);

            let u_node_bl = gather_f32x8(mac_grid_u, u_index_bottom_left);
            let u_node_br = gather_f32x8(mac_grid_u, u_index_bottom_right);
            let u_node_tl = gather_f32x8(mac_grid_u, u_index_top_left);
            let u_node_tr = gather_f32x8(mac_grid_u, u_index_top_right);

            let v_node_bl = gather_f32x8(mac_grid_v, v_index_bottom_left);
            let v_node_br = gather_f32x8(mac_grid_v, v_index_bottom_right);
            let v_node_tl = gather_f32x8(mac_grid_v, v_index_top_left);
            let v_node_tr = gather_f32x8(mac_grid_v, v_index_top_right);

            let u_velocity = bilinear_interpolate(
                u_weight_bottom_left,
                u_weight_bottom_right,
                u_weight_top_left,
                u_weight_top_right,
                u_node_bl,
                u_node_br,
                u_node_tl,
                u_node_tr,
            );

            let v_velocity = bilinear_interpolate(
                v_weight_bottom_left,
                v_weight_bottom_right,
                v_weight_top_left,
                v_weight_top_right,
                v_node_bl,
                v_node_br,
                v_node_tl,
                v_node_tr,
            );

            let mut new_velocity = Vec2x8 { x: u_velocity, y: v_velocity };

            let residual = *residual_org;
            
            let speed = new_velocity.mag();
            let cfl_local = speed * deltatime / f32x8::splat(cell_size);

            let scale = cfl_local.fast_max(zero).fast_min(f32x8::splat(1.0));
            let gamma = scale * scale * (f32x8::splat(3.0) - f32x8::splat(2.0) * scale);
            
            let jitter = gamma * Apic::random_simd(chunk_index, timestamp) * deltatime;
            
            let deltatime_raw = deltatime + residual + jitter;
            let deltatime_act = deltatime_raw.fast_max(zero).fast_min(deltatime * f32x8::splat(2.0));
            
            *residual_org = deltatime + residual - deltatime_act;
            
            let mut new_positions = *positions + deltatime_act * new_velocity;

            let out_x_min = new_positions.x.cmp_lt(min_x);
            let out_x_max = new_positions.x.cmp_gt(max_x);
            let out_y_min = new_positions.y.cmp_lt(min_y);
            let out_y_max = new_positions.y.cmp_gt(max_y);

            let out_vel_x_min = out_x_min & new_velocity.x.cmp_lt(zero);
            let out_vel_x_max = out_x_max & new_velocity.x.cmp_gt(zero);
            let out_vel_y_min = out_y_min & new_velocity.y.cmp_lt(zero);
            let out_vel_y_max = out_y_max & new_velocity.y.cmp_gt(zero);


            new_positions.x = out_x_min.blend(min_x, new_positions.x);
            new_positions.x = out_x_max.blend(max_x, new_positions.x);
            new_velocity.x = out_vel_x_min.blend(new_velocity.x * border_damping, new_velocity.x);
            new_velocity.x = out_vel_x_max.blend(new_velocity.x * border_damping, new_velocity.x);

            new_positions.y = out_y_min.blend(min_y, new_positions.y);
            new_positions.y = out_y_max.blend(max_y, new_positions.y);
            new_velocity.y = out_vel_y_min.blend(new_velocity.y * border_damping, new_velocity.y);
            new_velocity.y = out_vel_y_max.blend(new_velocity.y * border_damping, new_velocity.y);

            let d_inv = f32x8::splat(4.0 / (cell_size * cell_size));

            let mut c_u_x = f32x8::ZERO;
            let mut c_u_y = f32x8::ZERO;
            c_u_x += u_weight_bottom_left * u_node_bl * u_diff_bottom_left_x;
            c_u_y += u_weight_bottom_left * u_node_bl * u_diff_bottom_left_y;
            c_u_x += u_weight_bottom_right * u_node_br * u_diff_bottom_right_x;
            c_u_y += u_weight_bottom_right * u_node_br * u_diff_bottom_right_y;
            c_u_x += u_weight_top_left * u_node_tl * u_diff_top_left_x;
            c_u_y += u_weight_top_left * u_node_tl * u_diff_top_left_y;
            c_u_x += u_weight_top_right * u_node_tr * u_diff_top_right_x;
            c_u_y += u_weight_top_right * u_node_tr * u_diff_top_right_y;

            let mut c_v_x = f32x8::ZERO;
            let mut c_v_y = f32x8::ZERO;
            c_v_x += v_weight_bottom_left * v_node_bl * v_diff_bottom_left_x;
            c_v_y += v_weight_bottom_left * v_node_bl * v_diff_bottom_left_y;
            c_v_x += v_weight_bottom_right * v_node_br * v_diff_bottom_right_x;
            c_v_y += v_weight_bottom_right * v_node_br * v_diff_bottom_right_y;
            c_v_x += v_weight_top_left * v_node_tl * v_diff_top_left_x;
            c_v_y += v_weight_top_left * v_node_tl * v_diff_top_left_y;
            c_v_x += v_weight_top_right * v_node_tr * v_diff_top_right_x;
            c_v_y += v_weight_top_right * v_node_tr * v_diff_top_right_y;

            let mut c_u = Vec2x8 { x: c_u_x * d_inv, y: c_u_y * d_inv };
            let mut c_v = Vec2x8 { x: c_v_x * d_inv, y: c_v_y * d_inv };

            let reflect_x = out_vel_x_min | out_vel_x_max;
            let reflect_y = out_vel_y_min | out_vel_y_max;
            let negate = f32x8::splat(-1.0);

            c_u.y = reflect_x.blend(c_u.y * negate, c_u.y);
            c_v.x = reflect_x.blend(c_v.x * negate, c_v.x);
            c_u.y = reflect_y.blend(c_u.y * negate, c_u.y);
            c_v.x = reflect_y.blend(c_v.x * negate, c_v.x);

            *positions = new_positions;
            *velocity = new_velocity;

            *c_u_org = c_u;
            *c_v_org = c_v;
        });
    }

    pub fn extrapolate_velocity(&mut self) {
        let width = self.width as usize;
        let height = self.height as usize;

        for _ in 0..4 {
            std::mem::swap(&mut self.mac_grid_u, &mut self.old_grid_u);
            std::mem::swap(&mut self.mac_valid_u, &mut self.next_valid_u);

            let old_grid_u = &self.old_grid_u;
            let next_valid_u = &self.next_valid_u;

            self.mac_grid_u.par_iter_mut()
                .zip(self.mac_valid_u.par_iter_mut())
                .enumerate()
                .with_min_len(512)
                .for_each(|(index, (u, valid))| {
                if next_valid_u[index] {
                    *u = old_grid_u[index];
                    *valid = true;
                } else {
                    let mut sum: f32 = 0.0;
                    let mut count: u32 = 0;
                    let x = index % (width + 1);
                    let y = index / (width + 1);

                    if x > 0 && next_valid_u[index - 1] { sum += old_grid_u[index - 1]; count += 1; }
                    if x < width && next_valid_u[index + 1] { sum += old_grid_u[index + 1]; count += 1; }
                    if y > 0 && next_valid_u[index - (width + 1)] { sum += old_grid_u[index - (width + 1)]; count += 1; }
                    if y < height - 1 && next_valid_u[index + width + 1] { sum += old_grid_u[index + width + 1]; count += 1; }

                    if count > 0 {
                        *u = sum / (count as f32);
                        *valid = true;
                    } else {
                        *u = old_grid_u[index];
                        *valid = false;
                    }
                }
            });
        }
            
            
        for _ in 0..4 {
            std::mem::swap(&mut self.mac_grid_v, &mut self.old_grid_v);
            std::mem::swap(&mut self.mac_valid_v, &mut self.next_valid_v);

            let old_grid_v = &self.old_grid_v;
            let next_valid_v = &self.next_valid_v;

            self.mac_grid_v.par_iter_mut()
                .zip(self.mac_valid_v.par_iter_mut())
                .enumerate()
                .with_min_len(512)
                .for_each(|(index, (v, valid))| {
                if next_valid_v[index] {
                    *v = old_grid_v[index];
                    *valid = true;
                } else {
                    let mut sum: f32 = 0.0;
                    let mut count: u32 = 0;
                    let x = index % width;
                    let y = index / width;

                    if x > 0 && next_valid_v[index - 1] { sum += old_grid_v[index - 1]; count += 1; }
                    if x < width - 1 && next_valid_v[index + 1] { sum += old_grid_v[index + 1]; count += 1; }
                    if y > 0 && next_valid_v[index - width] { sum += old_grid_v[index - width]; count += 1; }
                    if y < height && next_valid_v[index + width] { sum += old_grid_v[index + width]; count += 1; }

                    if count > 0 {
                        *v = sum / (count as f32);
                        *valid = true;
                    } else {
                        *v = old_grid_v[index];
                        *valid = false;
                    }
                }
            });
        }

    }

    pub fn add_radial_force(&mut self, world_x: f32, world_y: f32, radius: f32, strength: f32) {
        let min_x = (((world_x - radius) / self.cell_size).floor() as i32).max(0) as u32;
        let max_x = (((world_x + radius) / self.cell_size).ceil() as i32).min(self.width as i32) as u32;
        let min_y = (((world_y - radius) / self.cell_size).floor() as i32).max(0) as u32;
        let max_y = (((world_y + radius) / self.cell_size).ceil() as i32).min(self.height as i32) as u32;

        let radius_sq = radius * radius;
        let is_repel = strength > 0.0;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                // U Grid faces
                if x <= self.width && y < self.height {
                    let u_world_x = (x as f32) * self.cell_size;
                    let u_world_y = (y as f32 + 0.5) * self.cell_size;
                    let dx = u_world_x - world_x;
                    let dy = u_world_y - world_y;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < radius_sq && dist_sq > 1e-5 {
                        let dist = dist_sq.sqrt();
                        let falloff = 1.0 - (dist / radius);
                        let dir_x = dx / dist;
                        let u_idx = self.u_index(x as i32, y as i32) as usize;
                        let force = dir_x * strength * falloff;
                        
                        if is_repel {
                            self.post_force_u[u_idx] += force;
                        } else {
                            self.external_force_u[u_idx] += force;
                        }
                    }
                }

                // V Grid faces
                if x < self.width && y <= self.height {
                    let v_world_x = (x as f32 + 0.5) * self.cell_size;
                    let v_world_y = (y as f32) * self.cell_size;
                    let dx = v_world_x - world_x;
                    let dy = v_world_y - world_y;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < radius_sq && dist_sq > 1e-5 {
                        let dist = dist_sq.sqrt();
                        let falloff = 1.0 - (dist / radius);
                        let dir_y = dy / dist;
                        let v_idx = self.v_index(x as i32, y as i32) as usize;
                        let force = dir_y * strength * falloff;
                        
                        if is_repel {
                            self.post_force_v[v_idx] += force;
                        } else {
                            self.external_force_v[v_idx] += force;
                        }
                    }
                }
            }
        }
    }

    pub fn get_max_particle_velocity(&self) -> f32 {
        let mut max_velocities = f32x8::ZERO;

        for chunk_index in 0..self.num_chunks as usize {
            let velocities = self.part_velocities[chunk_index];

            max_velocities = max_velocities.fast_max(velocities.mag())
        }

        let mut max_velocity: f32 = 0.0;
        for velocity in max_velocities.as_array_ref() {
            max_velocity = max_velocity.max(*velocity);
        }

        return max_velocity;
    }

    pub fn update(&mut self, frame_deltatime: f32) {
        let mut time_simulated = 0.0;
 
    
        while time_simulated < frame_deltatime {
            let max_velocity = self.get_max_particle_velocity();
            
            let max_safe_dt = if max_velocity > 1e-5 {
                self.world_properties.cfl * self.cell_size / max_velocity
            } else {
                frame_deltatime
            };

            let step_deltatime = max_safe_dt.min(frame_deltatime - time_simulated);
            
            self.update_spatial_lookup();
            self.transfer_particles_to_grid(step_deltatime, self.timestamp);
            self.apply_external_forces(step_deltatime);
            self.build_pressure_system();
            self.update_multilevel_grid();
            self.solve_pcg();
            self.apply_pressure_gradient();

            for i in 0..self.mac_grid_u.len() {
                self.mac_grid_u[i] += self.post_force_u[i] * step_deltatime;
            }
            for i in 0..self.mac_grid_v.len() {
                self.mac_grid_v[i] += self.post_force_v[i] * step_deltatime;
            }

            self.extrapolate_velocity();
            self.transfer_grid_to_particles_and_advect(step_deltatime);
            
            time_simulated += step_deltatime;

            self.prev_deltatime = step_deltatime;

            self.timestamp += 1;
        }

        self.external_force_u.fill(0.0);
        self.external_force_v.fill(0.0);
        self.post_force_u.fill(0.0);
        self.post_force_v.fill(0.0);
    }
}