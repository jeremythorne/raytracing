use glam::Vec3;
use rand::rngs::ThreadRng;
use super::*;

pub struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    _w: Vec3,
    lens_radius: f32
}

impl Camera {
    pub fn new(look_from:Vec3,
            look_at:Vec3,
            vup:Vec3,
            fov:f32, 
            aspect_ratio:f32,
            aperture:f32,
            focus_dist:f32
            ) -> Camera {
        let theta = fov * std::f32::consts::PI / 180.;
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let lens_radius = aperture / 2.;

        let origin = look_from;
        let horizontal = focus_dist * viewport_width * u;
        let vertical = focus_dist * viewport_height * v;
        Camera {
            origin,
            lower_left_corner: origin - 0.5 * (horizontal + vertical) - focus_dist * w,
            horizontal,
            vertical,
            u,
            v,
            _w: w,
            lens_radius
        }
    }

    pub fn get_ray(&self, s:f32, t:f32, rng: &mut ThreadRng) -> Ray {
        let rd = self.lens_radius * random_in_unit_disk(rng);
        let offset = self.u * rd.x + self.v * rd.y;
        Ray::new(self.origin + offset,
                    self.lower_left_corner +
                            s * self.horizontal +
                            t * self.vertical -
                            self.origin -
                            offset)
    }
}