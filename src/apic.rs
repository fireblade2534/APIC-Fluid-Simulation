use std::mem;
use rdst::{RadixKey, RadixSort};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use ultraviolet::{Vec2, Vec2x8};
use wide::{CmpGt, CmpLt, f32x8, i32x8, u32x8};

const SAFTEY: f32 = 1e-20;

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    
    let mut sum = f32x8::ZERO;
    let mut chunks_a = a.chunks_exact(8);
    let mut chunks_b = b.chunks_exact(8);
    
    for (chunk_a, chunk_b) in chunks_a.by_ref().zip(chunks_b.by_ref()) {
        let va = f32x8::from(*<&[f32; 8]>::try_from(chunk_a).unwrap());
        let vb = f32x8::from(*<&[f32; 8]>::try_from(chunk_b).unwrap());
        sum += va * vb;
    }
    
    let mut result = sum.to_array().iter().sum();
    for (remainder_a, remainder_b) in chunks_a.remainder().iter().zip(chunks_b.remainder().iter()) {
        result += remainder_a * remainder_b;
    }
    
    return result;
}

#[inline(always)]
pub fn gather_f32x8(slice: &[f32], indices: u32x8) -> f32x8 {
    let indexes = indices.to_array();
    unsafe {
        return f32x8::from([
            *slice.get_unchecked(indexes[0] as usize),
            *slice.get_unchecked(indexes[1] as usize),
            *slice.get_unchecked(indexes[2] as usize),
            *slice.get_unchecked(indexes[3] as usize),
            *slice.get_unchecked(indexes[4] as usize),
            *slice.get_unchecked(indexes[5] as usize),
            *slice.get_unchecked(indexes[6] as usize),
            *slice.get_unchecked(indexes[7] as usize),
        ]);
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

struct LocalGrids {
    grid_u: Vec<f32>, 
    grid_v: Vec<f32>, 
    weight_u: Vec<f32>,
    weight_v: Vec<f32>,
    density: Vec<f32>,
    type_fluid: Vec<u64>,
}

impl LocalGrids {
    fn new(apic: &Apic) -> Self {
        Self {
            grid_u: vec![0.0; apic.mac_grid_u.len()],
            weight_u: vec![0.0; apic.mac_weight_u.len()],
            grid_v: vec![0.0; apic.mac_grid_v.len()],
            weight_v: vec![0.0; apic.mac_weight_v.len()],
            density: vec![0.0; apic.mac_density.len()],
            type_fluid: vec![0; apic.mac_type_fluid.len()],
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (a, b) in self.grid_u.iter_mut().zip(&other.grid_u) { *a += b; }
        for (a, b) in self.weight_u.iter_mut().zip(&other.weight_u) { *a += b; }
        for (a, b) in self.grid_v.iter_mut().zip(&other.grid_v) { *a += b; }
        for (a, b) in self.weight_v.iter_mut().zip(&other.weight_v) { *a += b; }
        for (a, b) in self.density.iter_mut().zip(&other.density) { *a += b; }
        for (a, b) in self.type_fluid.iter_mut().zip(&other.type_fluid) { *a |= b; }
        
        return self;
    }
}

#[derive(Default, Clone)]
pub struct WorldProperties {
    pub gravity: f32,
    pub border_damping: f32,
    pub cfl: f32
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
    pub mac_type_fluid: Vec<u64>,
    pub mac_valid_u: Vec<bool>,
    pub mac_valid_v: Vec<bool>,

    pub old_grid_u: Vec<f32>,
    pub old_grid_v: Vec<f32>,
    pub next_valid_u: Vec<bool>,
    pub next_valid_v: Vec<bool>,

    pub smoothed_error: Vec<f32>,
    pub current_error: Vec<f32>,
    pub search_vector: Vec<f32>,
    pub matrix_times_search: Vec<f32>,
    pub diag_laplacian: Vec<f32>,
    pub plus_x_laplacian: Vec<f32>,
    pub plus_y_laplacian: Vec<f32>,
    pub precondition: Vec<f32>,
    pub precondition_temp: Vec<f32>,

    pub part_lookup: Vec<(u32, u32)>,

    pub part_positions: Vec<Vec2x8>,
    pub part_velocities: Vec<Vec2x8>,
    pub part_c_u: Vec<Vec2x8>,
    pub part_c_v: Vec<Vec2x8>,
    pub part_sort: Vec<SortableTuple>,
    pub part_positions_sort: Vec<Vec2x8>,
    pub part_velocities_sort: Vec<Vec2x8>,
    pub part_c_u_sort: Vec<Vec2x8>,
    pub part_c_v_sort: Vec<Vec2x8>,

    pub timestamp: u32,
}

impl Apic {
    pub fn new(width: u32, height: u32, cell_size: f32, world_properties: WorldProperties) -> Apic {
        let mut flip = Apic::default();

        flip.world_properties = world_properties;

        flip.width = width;
        flip.height = height;
        flip.cell_size = cell_size;

        flip.cells = width * height;

        flip.mac_pressure_grid.resize(flip.cells as usize, 0.0);
        flip.mac_grid_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.mac_grid_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.mac_weight_u.resize((flip.cells + flip.height) as usize, 0.0); 
        flip.mac_valid_u.resize((flip.cells + flip.height) as usize, false);
        flip.mac_weight_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.mac_valid_v.resize((flip.cells + flip.width) as usize, false);
        flip.mac_type_obstacle.resize(flip.cells.div_ceil(64) as usize, 0);
        flip.mac_type_fluid.resize(flip.cells.div_ceil(64) as usize, 0);

        flip.old_grid_u.resize((flip.cells + flip.height) as usize, 0.0);
        flip.old_grid_v.resize((flip.cells + flip.width) as usize, 0.0);
        flip.next_valid_u.resize((flip.cells + flip.height) as usize, false);
        flip.next_valid_v.resize((flip.cells + flip.width) as usize, false);

        flip.mac_density.resize(flip.cells as usize, 0.0);
        flip.smoothed_density.resize(flip.cells as usize, 0.0);
        flip.old_density.resize(flip.cells as usize, 0.0);

        flip.smoothed_error.resize(flip.cells as usize, 0.0);
        flip.current_error.resize(flip.cells as usize, 0.0);
        flip.search_vector.resize(flip.cells as usize, 0.0);
        flip.matrix_times_search.resize(flip.cells as usize, 0.0);
        flip.diag_laplacian.resize(flip.cells as usize, 0.0);
        flip.plus_x_laplacian.resize(flip.cells as usize, 0.0);
        flip.plus_y_laplacian.resize(flip.cells as usize, 0.0);
        flip.precondition.resize(flip.cells as usize, 0.0);
        flip.precondition_temp.resize(flip.cells as usize, 0.0);

        flip.part_positions.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_velocities.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_u.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_v.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_positions_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_velocities_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_u_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_c_v_sort.resize((flip.cells / 2) as usize, Vec2x8::zero());
        flip.part_lookup.resize((flip.cells / 2) as usize, (0, 0));
        flip.part_sort.resize((flip.cells / 2) as usize, SortableTuple::new(0, 0));
        flip.num_chunks = flip.cells / 2;

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
        let max_dim = flip.width.max(flip.height).next_power_of_two();
        let lookup_size = (max_dim * max_dim) as usize;
        flip.part_lookup.resize(lookup_size, (0, 0));
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

        return flip;
    }

    #[inline(always)]
    pub fn random_simd(&self, chunk_index: usize, seed: u32) -> f32x8 {
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
        let clamped_j: u32 = j.min(((self.height - 1) as i32)).max(0) as u32;
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

        let index = ((j * self.width as i32) + i) as u32;
        let fluid = self.mac_type_fluid[(index / 64) as usize];

        return 1 & (fluid >> (index % 64));
    }

    #[inline(always)]
    pub fn is_fluid_index(&self, index: usize) -> u64 {
        unsafe {
            let fluid = *self.mac_type_fluid.get_unchecked(index >> 6);
            1 & (fluid >> (index & 63))
        }
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

        self.part_lookup.fill((u32::MAX, u32::MAX));
        self.part_sort[..self.num_particles as usize].radix_sort_unstable();

        let mut position_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut velocity_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut c_u_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];
        let mut c_v_buffer: [[f32; 8]; 2] = [[0f32; 8]; 2];

        let mut previous_hash: u32 = u32::MAX;
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

            if buffer_index == 7 {
                self.part_positions_sort[index / 8] = Vec2x8 { x: position_buffer[0].into(), y: position_buffer[1].into() };
                self.part_velocities_sort[index / 8] = Vec2x8 { x: velocity_buffer[0].into(), y: velocity_buffer[1].into() };
                self.part_c_u_sort[index / 8] = Vec2x8 { x: c_u_buffer[0].into(), y: c_u_buffer[1].into() };
                self.part_c_v_sort[index / 8] = Vec2x8 { x: c_v_buffer[0].into(), y: c_v_buffer[1].into() };
            }

            if previous_hash != hash {
                if previous_hash != u32::MAX {
                    self.part_lookup[previous_hash as usize].1 = index as u32;
                }

                self.part_lookup[hash as usize].0 = index as u32;
                previous_hash = hash;
            }
        }

        if previous_hash != u32::MAX {
            self.part_lookup[previous_hash as usize].1 = self.num_particles;
        }

        mem::swap(&mut self.part_positions, &mut self.part_positions_sort);
        mem::swap(&mut self.part_velocities, &mut self.part_velocities_sort);
        mem::swap(&mut self.part_c_u, &mut self.part_c_u_sort);
        mem::swap(&mut self.part_c_v, &mut self.part_c_v_sort);
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
        self.mac_type_fluid.fill(0);

        self.mac_valid_u.fill(false);
        self.mac_valid_v.fill(false);

        self.mac_density.fill(0.0);

        for chunk_index in 0..self.num_chunks as usize {
            let raw_positions = self.part_positions[chunk_index];
            let velocities = self.part_velocities[chunk_index];

            let temporal_jitter = self.random_simd(chunk_index, timestamp) * f32x8::splat(deltatime);
            let seed_x = timestamp.wrapping_mul(73856093) ^ 0x193a6754;
            let seed_y = timestamp.wrapping_mul(19349663) ^ 0x45678912;

            let max_spatial_jitter = f32x8::splat(self.cell_size);
            let spatial_jitter_x = self.random_simd(chunk_index, seed_x) * max_spatial_jitter;
            let spatial_jitter_y = self.random_simd(chunk_index, seed_y) * max_spatial_jitter;

            let mut positions = raw_positions + (velocities * temporal_jitter);
            positions.x += spatial_jitter_x;
            positions.y += spatial_jitter_y;

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
            
            let c_weight_back_left = (one - c_tx) * (one - c_ty);
            let c_weight_back_right = c_tx * (one - c_ty);
            let c_weight_top_left = (one - c_tx) * c_ty;
            let c_weight_top_right = c_tx * c_ty;

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
                u_weight_bottom_left,
                u_weight_bottom_right,
                u_weight_top_left,
                u_weight_top_right,
                u_index_bottom_left,
                u_index_bottom_right,
                u_index_top_left,
                u_index_top_right,
                _u_tx,
                _u_ty,
            ) = Apic::get_u_grid(self.width, self.height, grid_space_positions);

            let u_base_x = grid_space_positions.x.floor();
            let u_base_y = (grid_space_positions.y - half).floor();
            let true_u_tx = true_grid_space_positions.x - u_base_x;
            let true_u_ty = (true_grid_space_positions.y - half) - u_base_y;

            let (
                u_diff_bottom_left_x,
                u_diff_bottom_left_y,
                u_diff_bottom_right_x,
                u_diff_bottom_right_y,
                u_diff_top_left_x,
                u_diff_top_left_y,
                u_diff_top_right_x,
                u_diff_top_right_y
            ) = Apic::get_grid_distances(self.cell_size, true_u_tx, true_u_ty);
            
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
                _v_tx,
                _v_ty,
            ) = Apic::get_v_grid(self.width, self.height, grid_space_positions);

            let v_base_x = (grid_space_positions.x - half).floor();
            let v_base_y = grid_space_positions.y.floor();
            let true_v_tx = (true_grid_space_positions.x - half) - v_base_x;
            let true_v_ty = true_grid_space_positions.y - v_base_y;

            let (
                v_diff_bottom_left_x,
                v_diff_bottom_left_y,
                v_diff_bottom_right_x,
                v_diff_bottom_right_y,
                v_diff_top_left_x,
                v_diff_top_left_y,
                v_diff_top_right_x,
                v_diff_top_right_y
            ) = Apic::get_grid_distances(self.cell_size, true_v_tx, true_v_ty);


            

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
                    *self.mac_type_fluid.get_unchecked_mut(grid_index / 64) |= 1u64 << (grid_index % 64);

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

        for index in 0..self.mac_grid_u.len() {
            if self.mac_weight_u[index] > 0.0 {
                self.mac_valid_u[index] = true;
                self.mac_grid_u[index] /= self.mac_weight_u[index];
            }
        }

        for index in 0..self.mac_grid_v.len() {
            if self.mac_weight_v[index] > 0.0 {
                self.mac_valid_v[index] = true;
                self.mac_grid_v[index] /= self.mac_weight_v[index];
            }
        }

        for i in 0..self.mac_type_fluid.len() {
            self.mac_type_fluid[i] &= !self.mac_type_obstacle[i];
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
        
        ((rho_center - threshold) / diff).clamp(0.1, 1.0)
    }

    pub fn build_pressure_system(&mut self) {
        self.diag_laplacian.fill(0.0);
        self.plus_x_laplacian.fill(0.0);
        self.plus_y_laplacian.fill(0.0);

        self.smoothed_density.copy_from_slice(&self.mac_density);
        self.old_density.copy_from_slice(&self.mac_density);
        let width = self.width as usize;

        for _ in 0..3 {
            std::mem::swap(&mut self.smoothed_density, &mut self.old_density);    
            for fluid_mask_index in 0..self.mac_type_fluid.len() {
                let mut fluid_mask = self.mac_type_fluid[fluid_mask_index];

                while fluid_mask != 0 {
                    let index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);
                                        
                    let mut sum = self.old_density[index] * 4.0;
                    let mut weight = 4.0;
                    
                    if self.is_fluid_index(index - 1) == 1 { sum += self.old_density[index - 1]; weight += 1.0; }
                    if self.is_fluid_index(index + 1) == 1 { sum += self.old_density[index + 1]; weight += 1.0; }
                    if self.is_fluid_index(index - width) == 1 { sum += self.old_density[index - width]; weight += 1.0; }
                    if self.is_fluid_index(index + width) == 1 { sum += self.old_density[index + width]; weight += 1.0; }
                    
                    self.smoothed_density[index] = sum / weight;

                    fluid_mask &= fluid_mask - 1;
                }
            }
        }

        for fluid_mask_index in 0..self.mac_type_fluid.len() {
            let mut fluid_mask: u64 = self.mac_type_fluid[fluid_mask_index];

            while fluid_mask != 0 {
                let fluid_index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                let grid_x = fluid_index % self.width as usize;
                let grid_y = fluid_index / self.width as usize;

                let mut u_left = self.mac_grid_u[self.u_index(grid_x as i32, grid_y as i32) as usize];
                let mut u_right = self.mac_grid_u[self.u_index(grid_x as i32 + 1, grid_y as i32) as usize];
                let mut v_bottom = self.mac_grid_v[self.v_index(grid_x as i32, grid_y as i32) as usize];
                let mut v_top = self.mac_grid_v[self.v_index(grid_x as i32, grid_y as i32 + 1) as usize];

                let non_obstacle_left = 1 - self.is_obstacle(grid_x as i32 - 1, grid_y as i32);
                let non_obstacle_right = 1 - self.is_obstacle(grid_x as i32 + 1, grid_y as i32);
                let non_obstacle_bottom = 1 - self.is_obstacle(grid_x as i32, grid_y as i32 - 1);
                let non_obstacle_top = 1 - self.is_obstacle(grid_x as i32, grid_y as i32 + 1);

                u_left *= non_obstacle_left as f32;
                u_right *= non_obstacle_right as f32;
                v_bottom *= non_obstacle_bottom as f32;
                v_top *= non_obstacle_top as f32;

                let divergence = (u_right - u_left) + (v_top - v_bottom);
                
                let target_density = 9.0; 
                let rho_center = self.smoothed_density[fluid_index];

                let noise_threshold = target_density * 1.15; 
                let density_excess = (rho_center - noise_threshold).max(0.0).min(target_density);
                let correction_rate = 0.1;

                self.current_error[fluid_index] = -divergence * self.cell_size 
                    + density_excess * correction_rate * self.cell_size * self.cell_size;

                let fluid_left = self.is_fluid(grid_x as i32 - 1, grid_y as i32);
                let fluid_right = self.is_fluid(grid_x as i32 + 1, grid_y as i32);
                let fluid_bottom = self.is_fluid(grid_x as i32, grid_y as i32 - 1);
                let fluid_top = self.is_fluid(grid_x as i32, grid_y as i32 + 1);

                let mut diag = 0.0;
                let width = self.width as usize;

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
                        self.plus_x_laplacian[fluid_index] = -1.0; 
                    } else { 
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index + 1], rho_center, target_density); 
                        self.plus_x_laplacian[fluid_index] = 0.0; 
                    }
                } else {
                    self.plus_x_laplacian[fluid_index] = 0.0;
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
                        self.plus_y_laplacian[fluid_index] = -1.0; 
                    } else { 
                        diag += 1.0 / self.calculate_theta(self.smoothed_density[fluid_index + width], rho_center, target_density); 
                        self.plus_y_laplacian[fluid_index] = 0.0; 
                    }
                } else {
                    self.plus_y_laplacian[fluid_index] = 0.0;
                }

                self.diag_laplacian[fluid_index] = diag;

                fluid_mask &= fluid_mask - 1;
            }
        }
    }


    pub fn apply_preconditioner(&mut self) {
        self.precondition_temp.fill(0.0);
        self.smoothed_error.fill(0.0);

        let width = self.width as usize;

        unsafe {
            for fluid_mask_index in 0..self.mac_type_fluid.len() {
                let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                while fluid_mask != 0 {
                    let fluid_index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                    let mut sum = *self.current_error.get_unchecked(fluid_index);

                    let left_index = fluid_index - 1;
                    let bot_index = fluid_index - width;

                    let plus_x = *self.plus_x_laplacian.get_unchecked(left_index);
                    let p_left = *self.precondition.get_unchecked(left_index);
                    let t_left = *self.precondition_temp.get_unchecked(left_index);
                    sum = (-(plus_x * p_left)).mul_add(t_left, sum);

                    let plus_y = *self.plus_y_laplacian.get_unchecked(bot_index);
                    let p_bot = *self.precondition.get_unchecked(bot_index);
                    let t_bot = *self.precondition_temp.get_unchecked(bot_index);
                    sum = (-(plus_y * p_bot)).mul_add(t_bot, sum);

                    *self.precondition_temp.get_unchecked_mut(fluid_index) = sum * *self.precondition.get_unchecked(fluid_index);

                    fluid_mask &= fluid_mask - 1;
                }
            }
        }

        unsafe {
            for fluid_mask_index in (0..self.mac_type_fluid.len()).rev() {
                let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                while fluid_mask != 0 {
                    let bit_index = 63 - fluid_mask.leading_zeros() as usize;
                    let fluid_index = (fluid_mask_index * 64) + bit_index;

                    let mut sum = *self.precondition_temp.get_unchecked(fluid_index);

                    let right_index = fluid_index + 1;
                    let top_index = fluid_index + width;

                    let precondition = *self.precondition.get_unchecked(fluid_index);

                    let plus_x = *self.plus_x_laplacian.get_unchecked(fluid_index);
                    let s_right = *self.smoothed_error.get_unchecked(right_index);

                    let plus_y = *self.plus_y_laplacian.get_unchecked(fluid_index);
                    let s_top = *self.smoothed_error.get_unchecked(top_index);

                    let inner = plus_x.mul_add(s_right, plus_y * s_top);
                    sum = (-precondition).mul_add(inner, sum);

                    *self.smoothed_error.get_unchecked_mut(fluid_index) = sum * precondition;

                    fluid_mask ^= 1 << bit_index; 
                }
            }
        }
    }

    pub fn solve_pcg(&mut self) {
        self.precondition.fill(0.0);

        unsafe {
            for fluid_mask_index in 0..self.mac_type_fluid.len() {
                let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                while fluid_mask != 0 {
                    let fluid_index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                    *self.mac_pressure_grid.get_unchecked_mut(fluid_index) = 0.0;

                    fluid_mask &= fluid_mask - 1;
                }
            }
        }

        for y in 0..self.height {
            for x in 0..self.width {
                let index = (y * self.width + x) as usize;

                if self.is_fluid_index(index) == 0 {
                    continue;
                }

                let mut ap_value = self.diag_laplacian[index] * self.mac_pressure_grid[index];

                let mut diagonal = self.diag_laplacian[index];

                let mut connection_x: f32 = 0.0;
                let mut connection_y: f32 = 0.0;


                let left_index = index - 1;
                let bottom_index = index - self.width as usize;

                let precon_x = self.precondition[left_index];
                let connection_x = self.plus_x_laplacian[left_index] * precon_x;
                diagonal -= connection_x * connection_x;
                diagonal -= (connection_x * self.plus_y_laplacian[left_index] * precon_x);

                let precon_y = self.precondition[bottom_index];
                let connection_y: f32 = self.plus_y_laplacian[bottom_index] * precon_y;
                diagonal -= connection_y * connection_y;
                diagonal -= (connection_y * self.plus_x_laplacian[bottom_index] * precon_y);

                ap_value += self.plus_x_laplacian[index] * self.mac_pressure_grid[index + 1];
                ap_value += self.plus_x_laplacian[left_index] * self.mac_pressure_grid[left_index];
                ap_value += self.plus_y_laplacian[index] * self.mac_pressure_grid[index + self.width as usize];
                ap_value += self.plus_y_laplacian[bottom_index] * self.mac_pressure_grid[bottom_index];
                

                if diagonal < SAFTEY {
                    diagonal = self.diag_laplacian[index];
                }

                self.precondition[index] = 1.0 / (diagonal.sqrt() + SAFTEY);

                self.current_error[index] -= ap_value;
            }
        }

        self.apply_preconditioner();

        self.search_vector.copy_from_slice(&self.smoothed_error);

        let mut sigma = dot_product(&self.current_error, &self.smoothed_error);

        if sigma < SAFTEY {
            return;
        }

        let max_iterations: usize = 100;

        for _ in 0..max_iterations {
            self.matrix_times_search.fill(0.0);
            unsafe {
                for fluid_mask_index in 0..self.mac_type_fluid.len() {
                    let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                    while fluid_mask != 0 {
                        let index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                        let mut value = *self.diag_laplacian.get_unchecked(index) * *self.search_vector.get_unchecked(index);
                        value = self.plus_x_laplacian.get_unchecked(index).mul_add(*self.search_vector.get_unchecked(index + 1), value);
                        value = self.plus_x_laplacian.get_unchecked(index - 1).mul_add(*self.search_vector.get_unchecked(index - 1), value);
                        value = self.plus_y_laplacian.get_unchecked(index).mul_add(*self.search_vector.get_unchecked(index + self.width as usize), value);
                        value = self.plus_y_laplacian.get_unchecked(index - self.width as usize).mul_add(*self.search_vector.get_unchecked(index - self.width as usize), value);
                        
                        *self.matrix_times_search.get_unchecked_mut(index) = value;
                        fluid_mask &= fluid_mask - 1;
                    }
                }
            }

            let search_dot_matrix = dot_product(&self.search_vector, &self.matrix_times_search);

            if search_dot_matrix.abs() < SAFTEY {
                break;
            }

            let alpha = sigma / search_dot_matrix;

            let mut max_error: f32 = 0.0;
            unsafe {
                for fluid_mask_index in 0..self.mac_type_fluid.len() {
                    let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                    while fluid_mask != 0 {
                        let index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                        *self.mac_pressure_grid.get_unchecked_mut(index) += alpha * *self.search_vector.get_unchecked(index);
                            
                        *self.current_error.get_unchecked_mut(index) -= alpha * *self.matrix_times_search.get_unchecked(index);

                        max_error = max_error.max(self.current_error.get_unchecked(index).abs());

                        fluid_mask &= fluid_mask - 1;
                    }
                }
            }

            if max_error < 1e-8 {
                break;
            }

            self.apply_preconditioner();

            let new_sigma = dot_product(&self.current_error, &self.smoothed_error);

            if new_sigma < SAFTEY {
                break; 
            }

            let beta = new_sigma / sigma;

             unsafe {
                for fluid_mask_index in 0..self.mac_type_fluid.len() {
                    let mut fluid_mask: u64 = *self.mac_type_fluid.get_unchecked(fluid_mask_index);

                    while fluid_mask != 0 {
                        let index = (fluid_mask_index * 64) + (fluid_mask.trailing_zeros() as usize);

                        *self.search_vector.get_unchecked_mut(index) = *self.smoothed_error.get_unchecked(index) + (beta * *self.search_vector.get_unchecked(index));

                        fluid_mask &= fluid_mask - 1;
                    }
                }
            }

            sigma = new_sigma;
        }
    }

    pub fn apply_pressure_gradient(&mut self) {

        for y in 0..self.height as i32 {
            for x in 0..=self.width as i32 {
                let face_index = self.u_index(x, y) as usize;

                let left_cell_x = x - 1;

                let left_is_obstacle = self.is_obstacle(left_cell_x, y) == 1;
                let right_is_obstacle = self.is_obstacle(x, y) == 1;

                if left_is_obstacle != right_is_obstacle {
                    self.mac_grid_u[face_index] = 0.0;
                    self.mac_valid_u[face_index] = true;
                    continue;
                }

                let left_is_fluid = self.is_fluid(left_cell_x, y) == 1;
                let right_is_fluid = self.is_fluid(x, y) == 1;
                
                let target_density = 9.0;
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

                self.mac_grid_u[face_index] -= pressure_difference / self.cell_size;

            }
        }

        for y in 0..=self.height as i32 {
            for x in 0..self.width as i32 {

                let face_index = self.v_index(x, y) as usize;

                let bottom_cell_y = y - 1;

                let bottom_is_obstacle = self.is_obstacle(x, bottom_cell_y) == 1;
                let top_is_obstacle = self.is_obstacle(x, y) == 1;

                if bottom_is_obstacle != top_is_obstacle {
                    self.mac_grid_v[face_index] = 0.0;
                    self.mac_valid_v[face_index] = true;
                    continue;
                }

                let bottom_is_fluid = self.is_fluid(x, bottom_cell_y) == 1;
                let top_is_fluid = self.is_fluid(x, y) == 1;
                
                let target_density = 9.0;
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

                self.mac_grid_v[face_index] -= pressure_difference / self.cell_size;
            }
        }
    }

    pub fn transfer_grid_to_particles_and_advect(&mut self, deltatime: f32) {
        let dt = f32x8::splat(deltatime);

        let min_x = f32x8::splat(1.001 * self.cell_size);
        let max_x = f32x8::splat((self.width as f32 - 1.001) * self.cell_size);
        let min_y = f32x8::splat(1.001 * self.cell_size);
        let max_y = f32x8::splat((self.height as f32 - 1.001) * self.cell_size);

        let zero = f32x8::ZERO;

        let width = self.width;
        let height = self.height;
        let cell_size = self.cell_size;
        
        self.part_positions.par_iter_mut()
            .zip(self.part_velocities.par_iter_mut())
            .zip(self.part_c_u.par_iter_mut())
            .zip(self.part_c_v.par_iter_mut())
            .for_each(|(((positions, velocity), c_u_org), c_v_org)| {

            let grid_space_positions = *positions / f32x8::splat(self.cell_size);

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

            let u_node_bl = gather_f32x8(&self.mac_grid_u, u_index_bottom_left);
            let u_node_br = gather_f32x8(&self.mac_grid_u, u_index_bottom_right);
            let u_node_tl = gather_f32x8(&self.mac_grid_u, u_index_top_left);
            let u_node_tr = gather_f32x8(&self.mac_grid_u, u_index_top_right);

            let v_node_bl = gather_f32x8(&self.mac_grid_v, v_index_bottom_left);
            let v_node_br = gather_f32x8(&self.mac_grid_v, v_index_bottom_right);
            let v_node_tl = gather_f32x8(&self.mac_grid_v, v_index_top_left);
            let v_node_tr = gather_f32x8(&self.mac_grid_v, v_index_top_right);

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
            let mut new_positions =  *positions + dt * new_velocity;

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
            new_velocity.x = out_vel_x_min.blend(new_velocity.x * f32x8::splat(-self.world_properties.border_damping), new_velocity.x);
            new_velocity.x = out_vel_x_max.blend(new_velocity.x * f32x8::splat(-self.world_properties.border_damping), new_velocity.x);

            new_positions.y = out_y_min.blend(min_y, new_positions.y);
            new_positions.y = out_y_max.blend(max_y, new_positions.y);
            new_velocity.y = out_vel_y_min.blend(new_velocity.y * f32x8::splat(-self.world_properties.border_damping), new_velocity.y);
            new_velocity.y = out_vel_y_max.blend(new_velocity.y * f32x8::splat(-self.world_properties.border_damping), new_velocity.y);

            let d_inv = f32x8::splat(4.0 / (self.cell_size * self.cell_size));

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
        std::mem::swap(&mut self.mac_grid_u, &mut self.old_grid_u);
        std::mem::swap(&mut self.mac_valid_u, &mut self.next_valid_u);

        let width = self.width as usize;
        let height = self.height as usize;

        for iter in 0..4 {
            for y in 0..height {
                for x in 0..=width {
                    let index = y * (width + 1) + x;

                    if self.next_valid_u[index] {
                        self.mac_grid_u[index] = self.old_grid_u[index];
                        self.mac_valid_u[index] = true;
                    } else {
                        let mut sum: f32 = 0.0;
                        let mut count: u32 = 0;

                        if x > 0 {
                            let new_index = index - 1;
                            if self.next_valid_u[new_index] {
                                sum += self.old_grid_u[new_index];
                                count += 1;
                            }
                        }

                        if x < width {
                            let new_index = index + 1;
                            if self.next_valid_u[new_index] {
                                sum += self.old_grid_u[new_index];
                                count += 1;
                            }
                        }

                        if y > 0 {
                            let new_index = index - (width + 1);
                            if self.next_valid_u[new_index] {
                                sum += self.old_grid_u[new_index];
                                count += 1;
                            }
                        }

                        if y < height - 1 {
                            let new_index = index + width + 1;
                            if self.next_valid_u[new_index] {
                                sum += self.old_grid_u[new_index];
                                count += 1;
                            }
                        }


                        if count > 0 {
                            self.mac_grid_u[index] = sum / (count as f32);
                            self.mac_valid_u[index] = true;
                        } else {
                            self.mac_grid_u[index] = self.old_grid_u[index];
                            self.mac_valid_u[index] = false;
                        }
                    }
                }
            }
            
            if iter < 3 {
                std::mem::swap(&mut self.mac_grid_u, &mut self.old_grid_u);
                std::mem::swap(&mut self.mac_valid_u, &mut self.next_valid_u);
            }
        }
            
        std::mem::swap(&mut self.mac_grid_v, &mut self.old_grid_v);
        std::mem::swap(&mut self.mac_valid_v, &mut self.next_valid_v);

        for iter in 0..4 {
            for y in 0..=height {
                for x in 0..width {
                    let index = y * width + x;

                    if self.next_valid_v[index] {
                        self.mac_grid_v[index] = self.old_grid_v[index];
                        self.mac_valid_v[index] = true;
                    } else {
                        let mut sum: f32 = 0.0;
                        let mut count: u32 = 0;

                        if x > 0 {
                            let new_index = index - 1;
                            if self.next_valid_v[new_index] {
                                sum += self.old_grid_v[new_index];
                                count += 1;
                            }
                        }

                        if x < width - 1 {
                            let new_index = index + 1;
                            if self.next_valid_v[new_index] {
                                sum += self.old_grid_v[new_index];
                                count += 1;
                            }
                        }

                        if y > 0 {
                            let new_index = index - width;
                            if self.next_valid_v[new_index] {
                                sum += self.old_grid_v[new_index];
                                count += 1;
                            }
                        }

                        if y < height {
                            let new_index = index + width;
                            if self.next_valid_v[new_index] {
                                sum += self.old_grid_v[new_index];
                                count += 1;
                            }
                        }

                        if count > 0 {
                            self.mac_grid_v[index] = sum / (count as f32);
                            self.mac_valid_v[index] = true;
                        } else {
                            self.mac_grid_v[index] = self.old_grid_v[index];
                            self.mac_valid_v[index] = false;
                        }
                    }
                }
            }

            if iter < 3 {
                std::mem::swap(&mut self.mac_grid_v, &mut self.old_grid_v);
                std::mem::swap(&mut self.mac_valid_v, &mut self.next_valid_v);
            }
        }

    }

    pub fn add_radial_force(&mut self, world_x: f32, world_y: f32, radius: f32, strength: f32) {
        let min_x = (((world_x - radius) / self.cell_size).floor() as i32).max(0) as u32;
        let max_x = (((world_x + radius) / self.cell_size).ceil() as i32).min(self.width as i32) as u32;
        let min_y = (((world_y - radius) / self.cell_size).floor() as i32).max(0) as u32;
        let max_y = (((world_y + radius) / self.cell_size).ceil() as i32).min(self.height as i32) as u32;

        let radius_sq = radius * radius;
        let is_repel = strength > 0.0; // Route positive pushing to post_force

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

            self.timestamp += 1;
        }

        self.external_force_u.fill(0.0);
        self.external_force_v.fill(0.0);
        self.post_force_u.fill(0.0);
        self.post_force_v.fill(0.0);
    }
}