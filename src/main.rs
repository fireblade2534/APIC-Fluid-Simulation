/*
This file is entirly made with AI
*/


use std::time::Instant;

mod apic;

use apic::{Apic, DebugOptions, WorldProperties};
use macroquad::{
    camera::{set_camera, set_default_camera, Camera2D},
    color::{Color, BLACK, WHITE, YELLOW},
    input::{
        is_key_pressed, is_mouse_button_down, is_mouse_button_pressed, mouse_position,
        mouse_wheel, KeyCode, MouseButton,
    },
    math::{vec2, Rect, Vec2 as MqVec2},
    shapes::draw_circle_lines,
    text::draw_text,
    texture::{draw_texture_ex, DrawTextureParams, Texture2D},
    time::{get_fps, get_frame_time},
    window::{clear_background, next_frame, Conf},
};

const WIDTH: f32 = 1920.0;
const HEIGHT: f32 = 1080.0;

const CELL_SIZE: f32 = 0.075;
const GRID_WIDTH: u32 = 256;
const GRID_HEIGHT: u32 = 128;

const SIM_SCALE: f32 = WIDTH / (GRID_WIDTH as f32 * CELL_SIZE);
const DRAW_RADIUS: f32 = 3.0;
const PARTICLE_PICK_RADIUS_PX: f32 = 12.0;
const DEBUG_STEP_DT: f32 = 1.0 / 60.0;

#[derive(Default)]
struct RenderStats {
    valid: u32,
    nan: u32,
    out_of_bounds: u32,
}

struct ParticleDebugInfo {
    particle_index: usize,
    chunk_index: usize,
    lane: usize,
    position: [f32; 2],
    velocity: [f32; 2],
    speed: f32,
    c_u: [f32; 2],
    c_v: [f32; 2],
    time_residual: f32,
    grid_cell: [i32; 2],
    grid_index: Option<usize>,
    pressure: Option<f32>,
    density: Option<f32>,
    smoothed_density: Option<f32>,
    is_fluid: bool,
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Fluid Simulation".to_owned(),
        window_width: WIDTH as i32,
        window_height: HEIGHT as i32,
        window_resizable: false,
        ..Default::default()
    }
}

fn make_circle_texture() -> Texture2D {
    let texture_size = 32u16;
    let radius = texture_size as f32 / 2.0;
    let mut bytes = vec![0u8; (texture_size * texture_size * 4) as usize];

    for y in 0..texture_size {
        for x in 0..texture_size {
            let dx = x as f32 - radius + 0.5;
            let dy = y as f32 - radius + 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            let idx = ((y * texture_size + x) * 4) as usize;

            let alpha = if dist < radius - 1.0 {
                255
            } else if dist < radius {
                ((radius - dist) * 255.0) as u8
            } else {
                0
            };

            bytes[idx] = 255;
            bytes[idx + 1] = 255;
            bytes[idx + 2] = 255;
            bytes[idx + 3] = alpha;
        }
    }

    Texture2D::from_rgba8(texture_size, texture_size, &bytes)
}

#[inline]
fn active_lanes(simulation: &Apic, chunk_index: usize) -> usize {
    let is_last_chunk = chunk_index + 1 == simulation.num_chunks as usize;
    if is_last_chunk && simulation.num_particles % 8 != 0 {
        (simulation.num_particles % 8) as usize
    } else {
        8
    }
}

#[inline]
fn particle_slot(particle_index: usize) -> (usize, usize) {
    (particle_index / 8, particle_index % 8)
}

fn particle_render_position(simulation: &Apic, particle_index: usize) -> Option<MqVec2> {
    if particle_index >= simulation.num_particles as usize {
        return None;
    }

    let (chunk, lane) = particle_slot(particle_index);
    let x = simulation.part_positions[chunk].x.as_array_ref()[lane];
    let y = simulation.part_positions[chunk].y.as_array_ref()[lane];

    if !x.is_finite() || !y.is_finite() {
        return None;
    }

    Some(vec2(x * SIM_SCALE, HEIGHT - y * SIM_SCALE))
}

fn particle_debug_info(simulation: &Apic, particle_index: usize) -> Option<ParticleDebugInfo> {
    if particle_index >= simulation.num_particles as usize {
        return None;
    }

    let (chunk, lane) = particle_slot(particle_index);

    let position = [
        simulation.part_positions[chunk].x.as_array_ref()[lane],
        simulation.part_positions[chunk].y.as_array_ref()[lane],
    ];
    let velocity = [
        simulation.part_velocities[chunk].x.as_array_ref()[lane],
        simulation.part_velocities[chunk].y.as_array_ref()[lane],
    ];
    let c_u = [
        simulation.part_c_u[chunk].x.as_array_ref()[lane],
        simulation.part_c_u[chunk].y.as_array_ref()[lane],
    ];
    let c_v = [
        simulation.part_c_v[chunk].x.as_array_ref()[lane],
        simulation.part_c_v[chunk].y.as_array_ref()[lane],
    ];
    let time_residual = simulation.part_time_residual[chunk].as_array_ref()[lane];

    let grid_x = (position[0] / simulation.cell_size).floor() as i32;
    let grid_y = (position[1] / simulation.cell_size).floor() as i32;
    let in_grid = grid_x >= 0
        && grid_x < simulation.width as i32
        && grid_y >= 0
        && grid_y < simulation.height as i32;

    let grid_index = in_grid.then(|| {
        grid_y as usize * simulation.width as usize + grid_x as usize
    });

    Some(ParticleDebugInfo {
        particle_index,
        chunk_index: chunk,
        lane,
        position,
        velocity,
        speed: velocity[0].hypot(velocity[1]),
        c_u,
        c_v,
        time_residual,
        grid_cell: [grid_x, grid_y],
        grid_index,
        pressure: grid_index.map(|i| simulation.mac_pressure_grid[i]),
        density: grid_index.map(|i| simulation.mac_density[i]),
        smoothed_density: grid_index.map(|i| simulation.smoothed_density[i]),
        is_fluid: in_grid && simulation.is_fluid(grid_x, grid_y) != 0,
    })
}

fn debug_step_dt(simulation: &Apic) -> f32 {
    let max_velocity = simulation.get_max_particle_velocity();

    if max_velocity > 1e-5 {
        let cfl_dt =
            simulation.world_properties.cfl * simulation.cell_size / max_velocity;
        DEBUG_STEP_DT.min(cfl_dt)
    } else {
        DEBUG_STEP_DT
    }
}

fn pick_radius_world(camera: &Camera2D, mouse_screen: MqVec2) -> f32 {
    let center = camera.screen_to_world(mouse_screen);
    let edge = camera.screen_to_world(mouse_screen + vec2(PARTICLE_PICK_RADIUS_PX, 0.0));
    (edge - center).length()
}

fn render_particles(
    simulation: &Apic,
    circle_texture: &Texture2D,
    pick_center: Option<MqVec2>,
    pick_radius: f32,
) -> (RenderStats, Option<usize>) {
    let mut stats = RenderStats::default();
    let mut picked = None;
    let mut best_pick_distance_sq = f32::INFINITY;
    let pick_radius_sq = pick_radius * pick_radius;

    for chunk_index in 0..simulation.num_chunks as usize {
        let x = simulation.part_positions[chunk_index].x.as_array_ref();
        let y = simulation.part_positions[chunk_index].y.as_array_ref();
        let vx = simulation.part_velocities[chunk_index].x.as_array_ref();
        let vy = simulation.part_velocities[chunk_index].y.as_array_ref();

        for lane in 0..active_lanes(simulation, chunk_index) {
            let sim_x = x[lane];
            let sim_y = y[lane];

            if !sim_x.is_finite() || !sim_y.is_finite() {
                stats.nan += 1;
                continue;
            }

            let draw_x = sim_x * SIM_SCALE;
            let draw_y = HEIGHT - sim_y * SIM_SCALE;

            if draw_x < -50.0
                || draw_x > WIDTH + 50.0
                || draw_y < -50.0
                || draw_y > HEIGHT + 50.0
            {
                stats.out_of_bounds += 1;
                continue;
            }

            stats.valid += 1;

            if let Some(center) = pick_center {
                let dx = draw_x - center.x;
                let dy = draw_y - center.y;
                let distance_sq = dx * dx + dy * dy;

                if distance_sq <= pick_radius_sq && distance_sq < best_pick_distance_sq {
                    best_pick_distance_sq = distance_sq;
                    picked = Some(chunk_index * 8 + lane);
                }
            }

            let speed_sq = vx[lane] * vx[lane] + vy[lane] * vy[lane];
            let r = (speed_sq * 0.01).min(1.0);
            let g = 0.5;
            let b = 1.0 - r * 0.5;

            draw_texture_ex(
                circle_texture,
                draw_x - DRAW_RADIUS,
                draw_y - DRAW_RADIUS,
                Color::new(r, g, b, 1.0),
                DrawTextureParams {
                    dest_size: Some(vec2(DRAW_RADIUS * 2.0, DRAW_RADIUS * 2.0)),
                    ..Default::default()
                },
            );
        }
    }

    (stats, picked)
}

fn draw_particle_debug_panel(info: &ParticleDebugInfo) {
    let grid_index = info
        .grid_index
        .map(|v| v.to_string())
        .unwrap_or_else(|| "OOB".to_owned());
    let pressure = info
        .pressure
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "N/A".to_owned());
    let density = info
        .density
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "N/A".to_owned());
    let smoothed_density = info
        .smoothed_density
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "N/A".to_owned());

    draw_text(
        &format!(
            "Tracked particle: {} | chunk: {} | lane: {}",
            info.particle_index, info.chunk_index, info.lane
        ),
        10.0,
        155.0,
        22.0,
        YELLOW,
    );

    draw_text(
        &format!(
            "pos: ({:.6}, {:.6}) m | vel: ({:.6}, {:.6}) m/s | speed: {:.6} m/s",
            info.position[0],
            info.position[1],
            info.velocity[0],
            info.velocity[1],
            info.speed,
        ),
        10.0,
        180.0,
        22.0,
        WHITE,
    );

    draw_text(
        &format!(
            "C_u: ({:.6}, {:.6}) | C_v: ({:.6}, {:.6}) | time residual: {:.6}",
            info.c_u[0],
            info.c_u[1],
            info.c_v[0],
            info.c_v[1],
            info.time_residual,
        ),
        10.0,
        205.0,
        22.0,
        WHITE,
    );

    draw_text(
        &format!(
            "grid: ({}, {}) | index: {} | fluid: {} | pressure: {} | density: {} | smoothed: {}",
            info.grid_cell[0],
            info.grid_cell[1],
            grid_index,
            info.is_fluid,
            pressure,
            density,
            smoothed_density,
        ),
        10.0,
        230.0,
        22.0,
        WHITE,
    );
}

#[macroquad::main(window_conf)]
async fn main() {
    let world_properties = WorldProperties {
        gravity: -9.8,
        border_damping: 0.5,
        cfl: 5.0,
        solve_error: 1e-5,
    };

    let mut simulation =
        Apic::new(GRID_WIDTH, GRID_HEIGHT, CELL_SIZE, world_properties);
    let circle_texture = make_circle_texture();

    let mut camera =
        Camera2D::from_display_rect(Rect::new(0.0, HEIGHT, WIDTH, -HEIGHT));

    let mut initial_volume = 0.0;
    let mut time_scale: f32 = 1.0;
    let mut paused = false;

    // This is a stable index while particle sorting is disabled.
    let mut selected_particle: Option<usize> = None;
    let mut last_mouse_pos = MqVec2::from(mouse_position());

    loop {
        let base_dt = get_frame_time().min(0.033);
        let current_mouse_pos = MqVec2::from(mouse_position());
        let left_pressed = is_mouse_button_pressed(MouseButton::Left);

        // Camera
        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            let zoom_factor = if wheel_y > 0.0 { 1.1 } else { 1.0 / 1.1 };
            let before = camera.screen_to_world(current_mouse_pos);
            camera.zoom *= zoom_factor;
            let after = camera.screen_to_world(current_mouse_pos);
            camera.target -= after - before;
        }

        if is_mouse_button_down(MouseButton::Middle) {
            let previous = camera.screen_to_world(last_mouse_pos);
            let current = camera.screen_to_world(current_mouse_pos);
            camera.target -= current - previous;
        }
        last_mouse_pos = current_mouse_pos;

        // Playback controls
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::Up) {
            time_scale = (time_scale * 2.0).min(8.0);
        }
        if is_key_pressed(KeyCode::Down) {
            time_scale = (time_scale * 0.5).max(0.0625);
        }

        let step_once = paused && is_key_pressed(KeyCode::Right);
        let playback_dt = if paused {
            0.0
        } else {
            base_dt * time_scale
        };

        // Mouse force field
        let world_mouse = camera.screen_to_world(current_mouse_pos);
        let sim_mouse_x = world_mouse.x / SIM_SCALE;
        let sim_mouse_y = (HEIGHT - world_mouse.y) / SIM_SCALE;
        let interaction_radius = 1.2;

        if playback_dt > 0.0 {
            if is_mouse_button_down(MouseButton::Left) && !left_pressed {
                simulation.add_radial_force(
                    sim_mouse_x,
                    sim_mouse_y,
                    interaction_radius,
                    150.0,
                );
            } else if is_mouse_button_down(MouseButton::Right) {
                simulation.add_radial_force(
                    sim_mouse_x,
                    sim_mouse_y,
                    interaction_radius,
                    -150.0,
                );
            }
        }

        // Freeze particle ordering while tracking one. This keeps the selected
        // particle at the same chunk/lane without adding a separate ID array.
        let debug_options = DebugOptions {
            disable_particle_sort: selected_particle.is_some(),
        };

        let sim_start = Instant::now();

        if step_once {
            let step_dt = debug_step_dt(&simulation);
            simulation.update(&debug_options, step_dt);
        } else if playback_dt > 0.0 {
            simulation.update(&debug_options, playback_dt);
        }

        let sim_time_ms = sim_start.elapsed().as_secs_f32() * 1000.0;

        let active_fluid_cells: u32 = simulation
            .base_grid
            .type_fluid
            .iter()
            .map(|mask| mask.count_ones())
            .sum();
        let current_volume =
            active_fluid_cells as f32 * (CELL_SIZE * CELL_SIZE);

        if initial_volume == 0.0 && current_volume > 0.0 {
            initial_volume = current_volume;
        }

        // Simulation view + picking
        clear_background(BLACK);
        set_camera(&camera);

        let pick_center = left_pressed.then_some(world_mouse);
        let pick_radius = if left_pressed {
            pick_radius_world(&camera, current_mouse_pos)
        } else {
            0.0
        };

        let (render_stats, picked_particle) =
            render_particles(&simulation, &circle_texture, pick_center, pick_radius);

        if left_pressed {
            selected_particle = picked_particle;
        }

        if let Some(index) = selected_particle {
            if let Some(position) = particle_render_position(&simulation, index) {
                draw_circle_lines(
                    position.x,
                    position.y,
                    DRAW_RADIUS + 4.0,
                    2.0,
                    YELLOW,
                );
            }
        }

        set_default_camera();

        // UI
        let frame_time_ms = base_dt * 1000.0;
        let speed_text = if paused {
            "Paused".to_owned()
        } else {
            format!("{time_scale}x")
        };

        draw_text(
            &format!(
                "FPS: {} | Particles: {} | Frame Time: {:.2}ms | Sim Time: {:.2}ms | Speed: {} | Tick: {}",
                get_fps(),
                simulation.num_particles,
                frame_time_ms,
                sim_time_ms,
                speed_text,
                simulation.timestamp,
            ),
            10.0,
            30.0,
            30.0,
            WHITE,
        );

        let volume_percent = if initial_volume > 0.0 {
            current_volume / initial_volume * 100.0
        } else {
            100.0
        };
        let volume_color = if volume_percent < 95.0 {
            Color::new(1.0, 0.4, 0.4, 1.0)
        } else {
            Color::new(0.4, 1.0, 0.4, 1.0)
        };

        draw_text(
            &format!(
                "Grid Area (Volume): {:.2} m^2 / {:.2} m^2 ({:.1}%)",
                current_volume, initial_volume, volume_percent
            ),
            10.0,
            60.0,
            30.0,
            volume_color,
        );

        let debug_color =
            if render_stats.nan > 0 || render_stats.out_of_bounds > 0 {
                Color::new(1.0, 0.0, 0.0, 1.0)
            } else {
                Color::new(1.0, 1.0, 0.0, 1.0)
            };

        draw_text(
            &format!(
                "Valid: {} | NaN: {} | Out of Bounds: {} | Particle sort: {}",
                render_stats.valid,
                render_stats.nan,
                render_stats.out_of_bounds,
                if selected_particle.is_some() {
                    "OFF (tracking)"
                } else {
                    "ON"
                },
            ),
            10.0,
            90.0,
            30.0,
            debug_color,
        );

        draw_text(
            "Controls: MMB Drag=Pan | Scroll=Zoom | Up/Down=Speed | Space=Pause | Right=Step Tick | L Click=Track | L Hold=Push | R Hold=Pull",
            10.0,
            120.0,
            20.0,
            WHITE,
        );

        if let Some(index) = selected_particle {
            if let Some(info) = particle_debug_info(&simulation, index) {
                draw_particle_debug_panel(&info);
            }
        }

        next_frame().await;
    }
}
