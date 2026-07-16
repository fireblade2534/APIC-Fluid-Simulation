use std::time::Instant;

mod flip;
use flip::{Flip, WorldProperties};
use macroquad::{
    camera::{set_camera, set_default_camera, Camera2D},
    color::{Color, BLACK, WHITE},
    input::{
        is_key_pressed, is_mouse_button_down, mouse_position, mouse_wheel, KeyCode, MouseButton,
    },
    math::{vec2, Rect, Vec2 as MqVec2},
    text::draw_text,
    texture::{draw_texture_ex, DrawTextureParams, Texture2D}, // Added texture functions
    time::{get_fps, get_frame_time},
    window::{clear_background, next_frame, Conf},
};
use wide::{CmpLt, f32x8};

const WIDTH: f32 = 1920.0;
const HEIGHT: f32 = 1080.0;
const CELL_SIZE: f32 = 0.075;
const GRID_WIDTH: u32 = 160;
const GRID_HEIGHT: u32 = 90;
const SIM_SCALE: f32 = WIDTH / (GRID_WIDTH as f32 * CELL_SIZE);
const DRAW_RADIUS: f32 = 3.0;

fn window_conf() -> Conf {
    Conf {
        window_title: "Fluid Simulation".to_owned(),
        window_width: WIDTH as i32,
        window_height: HEIGHT as i32,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let world_properties = WorldProperties {
        gravity: -9.8,
        border_damping: 0.5,
        density: 1000.0
    };

    let mut simulation = Flip::new(
        GRID_WIDTH,
        GRID_HEIGHT,
        CELL_SIZE,
        world_properties
    );

    // --- Generate an Anti-Aliased Circle Texture on Startup ---
    // This creates a smooth 32x32 white circle in memory.
    let texture_size = 32u16;
    let radius = texture_size as f32 / 2.0;
    let mut bytes = vec![0u8; (texture_size * texture_size * 4) as usize];
    
    for y in 0..texture_size {
        for x in 0..texture_size {
            let dx = x as f32 - radius + 0.5;
            let dy = y as f32 - radius + 0.5;
            let dist = (dx * dx + dy * dy).sqrt();
            
            let idx = ((y * texture_size + x) * 4) as usize;
            
            // Anything inside (radius - 1px) is completely solid
            // Anything outside (radius) is completely transparent
            // Anything in between is smoothly blended to avoid jagged pixels
            let alpha = if dist < radius - 1.0 {
                255
            } else if dist < radius {
                ((radius - dist) * 255.0) as u8
            } else {
                0
            };
            
            bytes[idx] = 255;     // R
            bytes[idx + 1] = 255; // G
            bytes[idx + 2] = 255; // B
            bytes[idx + 3] = alpha; // A
        }
    }
    let circle_texture = Texture2D::from_rgba8(texture_size, texture_size, &bytes); //

    let mut camera = Camera2D::from_display_rect(Rect::new(0.0, HEIGHT, WIDTH, -HEIGHT));

    let mut initial_volume: f32 = 0.0;

    // Control States
    let mut time_scale: f32 = 1.0;
    let mut paused: bool = false;
    let mut last_mouse_pos = MqVec2::from(mouse_position());

    loop {
        let base_dt = get_frame_time().min(0.033);

        let (mouse_x, mouse_y) = mouse_position();
        let current_mouse_pos = MqVec2::new(mouse_x, mouse_y);

        // --- Camera Pan & Zoom Controls ---
        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            // Zoom in or out by 10%
            let zoom_factor = if wheel_y > 0.0 { 1.1 } else { 1.0 / 1.1 };
            
            // Adjust the target so we zoom directly towards the mouse cursor
            let mouse_world_before = camera.screen_to_world(current_mouse_pos);
            camera.zoom *= zoom_factor;
            let mouse_world_after = camera.screen_to_world(current_mouse_pos);
            camera.target -= mouse_world_after - mouse_world_before;
        }

        if is_mouse_button_down(MouseButton::Middle) {
            let p1 = camera.screen_to_world(last_mouse_pos);
            let p2 = camera.screen_to_world(current_mouse_pos);
            // Move camera target opposite to mouse movement to drag the world
            camera.target -= p2 - p1;
        }
        last_mouse_pos = current_mouse_pos;

        // --- Sim Slowdown & Speed Controls ---
        if is_key_pressed(KeyCode::Space) {
            paused = !paused;
        }
        if is_key_pressed(KeyCode::Up) {
            time_scale = (time_scale * 2.0).min(8.0); // max 8x speed
        }
        if is_key_pressed(KeyCode::Down) {
            time_scale = (time_scale * 0.5).max(0.0625); // min 1/16x speed
        }

        let dt = if paused { 0.0 } else { base_dt * time_scale };

        // --- Interaction ---
        let world_mouse = camera.screen_to_world(current_mouse_pos);
        let sim_mouse_x = world_mouse.x / SIM_SCALE;
        let sim_mouse_y = (HEIGHT - world_mouse.y) / SIM_SCALE;
        
        let interaction_radius = 1.0; 
        let mut force_strength = 0.0;
        
        if is_mouse_button_down(MouseButton::Left) {
            force_strength = -50.0;
        } else if is_mouse_button_down(MouseButton::Right) {
            force_strength = 50.0;
        }

        if force_strength != 0.0 && dt > 0.0 {
            let radius_sq = f32x8::splat(interaction_radius * interaction_radius);
            let force = f32x8::splat(force_strength * dt);
            
            for chunk_index in 0..simulation.num_chunks as usize {
                let pos = simulation.part_positions[chunk_index];
                let dx = pos.x - f32x8::splat(sim_mouse_x);
                let dy = pos.y - f32x8::splat(sim_mouse_y);
                let dist_sq = dx * dx + dy * dy;

                let mask = dist_sq.cmp_lt(radius_sq);
                
                let dist = dist_sq.sqrt() + f32x8::splat(0.0001);
                let dir_x = dx / dist;
                let dir_y = dy / dist;

                let dv_x = dir_x * force;
                let dv_y = dir_y * force;

                let mut vel = simulation.part_velocities[chunk_index];
                vel.x = mask.blend(vel.x + dv_x, vel.x);
                vel.y = mask.blend(vel.y + dv_y, vel.y);
                simulation.part_velocities[chunk_index] = vel;
            }
        }

        // --- Simulation Step ---
        let sim_start_time = Instant::now();
        if dt > 0.0 {
            simulation.update(dt);
        }
        let sim_time_ms = sim_start_time.elapsed().as_secs_f32() * 1000.0;
        
        let active_fluid_cells: u32 = simulation.mac_type_fluid.iter().map(|&mask| mask.count_ones()).sum();
        let current_volume = active_fluid_cells as f32 * (CELL_SIZE * CELL_SIZE);
        
        if initial_volume == 0.0 && current_volume > 0.0 {
            initial_volume = current_volume;
        }

        clear_background(BLACK);
        set_camera(&camera);
        
        let mut valid_count = 0;
        let mut nan_count = 0;
        let mut oob_count = 0;

        for chunk_index in 0..simulation.num_chunks as usize {
            let x_arr = *simulation.part_positions[chunk_index].x.as_array_ref();
            let y_arr = *simulation.part_positions[chunk_index].y.as_array_ref();
            let vx_arr = *simulation.part_velocities[chunk_index].x.as_array_ref();
            let vy_arr = *simulation.part_velocities[chunk_index].y.as_array_ref();
            
            // Mask out unused trailing lanes in the last chunk so they don't corrupt counts
            let is_last_chunk = chunk_index == (simulation.num_chunks as usize - 1);
            let active_lanes = if is_last_chunk && simulation.num_particles % 8 != 0 {
                simulation.num_particles % 8
            } else {
                8
            };

            for lane in 0..active_lanes as usize {
                let px_world = x_arr[lane];
                let py_world = y_arr[lane];

                // Check for NaN
                if px_world.is_nan() || py_world.is_nan() {
                    nan_count += 1;
                    continue;
                }

                let px = px_world * SIM_SCALE;
                let py = HEIGHT - (py_world * SIM_SCALE);
                
                // Check if vastly out of bounds (which means it glitched past the clamp)
                if px < -50.0 || px > WIDTH + 50.0 || py < -50.0 || py > HEIGHT + 50.0 {
                    oob_count += 1;
                    continue;
                }

                valid_count += 1;
                
                // Optimization: Avoid calling expensive `.sqrt()` in the scalar render loop.
                let speed_sq = vx_arr[lane] * vx_arr[lane] + vy_arr[lane] * vy_arr[lane];
                let r = (speed_sq * 0.01).min(1.0); 
                let g = 0.5;
                let b = 1.0 - r * 0.5;
                
                // Draw a colored particle circle.
                // This utilizes GPU instancing/batching under the hood.
                draw_texture_ex(
                    &circle_texture,
                    px - DRAW_RADIUS,
                    py - DRAW_RADIUS,
                    Color::new(r, g, b, 1.0),
                    DrawTextureParams {
                        dest_size: Some(vec2(DRAW_RADIUS * 2.0, DRAW_RADIUS * 2.0)), //
                        ..Default::default()
                    }
                );
            }
        }

        set_default_camera();

        // --- Render UI ---
        let particle_count = simulation.num_particles;
        let frame_time_ms = base_dt * 1000.0;
        
        let speed_str = if paused { "Paused".to_string() } else { format!("{}x", time_scale) };
        let stats_text = format!(
            "FPS: {} | Particles: {} | Frame Time: {:.2}ms | Sim Time: {:.2}ms | Speed: {}", 
            get_fps(), 
            particle_count, 
            frame_time_ms, 
            sim_time_ms,
            speed_str
        );

        let volume_percent = if initial_volume > 0.0 { (current_volume / initial_volume) * 100.0 } else { 100.0 };
        let volume_text = format!(
            "Grid Area (Volume): {:.2} m^2 / {:.2} m^2 ({:.1}%)", 
            current_volume, 
            initial_volume, 
            volume_percent
        );

        let debug_text = format!(
            "Valid: {} | NaN: {} | Out of Bounds: {}",
            valid_count, nan_count, oob_count
        );

        draw_text(&stats_text, 10.0, 30.0, 30.0, WHITE);
        
        let vol_color = if volume_percent < 95.0 { Color::new(1.0, 0.4, 0.4, 1.0) } else { Color::new(0.4, 1.0, 0.4, 1.0) };
        draw_text(&volume_text, 10.0, 60.0, 30.0, vol_color);

        let debug_color = if nan_count > 0 || oob_count > 0 { Color::new(1.0, 0.0, 0.0, 1.0) } else { Color::new(1.0, 1.0, 0.0, 1.0) };
        draw_text(&debug_text, 10.0, 90.0, 30.0, debug_color);
        
        let controls_text = "Controls: MMB Drag=Pan | Scroll=Zoom | Up/Down=Speed | Space=Pause | L/R Click=Force";
        draw_text(controls_text, 10.0, 120.0, 20.0, WHITE);
        
        next_frame().await;
    }
}