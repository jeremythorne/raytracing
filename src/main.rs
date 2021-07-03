use image;
use glam::{Vec3};

#[derive(Clone, Copy)]
struct Ray {
    a: Vec3,
    b: Vec3
}

impl Ray {
    fn new(a: Vec3, b:Vec3) -> Ray {
        Ray { a, b }
    }

    fn origin(self) -> Vec3 {
        self.a
    }

    fn direction(self) -> Vec3 {
        self.b
    }

    fn point_at_parameter(self, t:f32) -> Vec3 {
        self.a + t * self.b
    }
}

fn hit_sphere(centre: Vec3, radius: f32, r: &Ray) -> bool {
    let oc = r.origin() - centre;
    let a = r.direction().dot(r.direction());
    let b = 2.0 * oc.dot(r.direction());
    let c = oc.dot(oc) - radius * radius;
    let discriminant = b * b - 4. * a * c;
    discriminant > 0.0
}

fn colour(r: &Ray) -> Vec3 {
    if hit_sphere(Vec3::new(0.0, 0.0, -1.0), 0.5, &r) {
        return Vec3::new(1.0, 0.0, 0.0);
    }
    let unit_direction = r.direction().normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}

fn main() {
    let nx = 200;
    let ny = 100;
    let mut img = image::RgbImage::new(nx, ny);

    let lower_left_corner = Vec3::new(-2.0, -1.0, -1.0);
    let horizontal = Vec3::new(4.0, 0.0, 0.0);
    let vertical = Vec3::new(0.0, 2.0, 0.0);
    let origin = Vec3::new(0.0, 0.0, 0.0);

    for (i, j, pixel) in img.enumerate_pixels_mut() {
        let u = (i as f32) / (nx as f32);
        let v = 1.0 - (j as f32) / (ny as f32);
        let r = Ray::new(origin, lower_left_corner + u * horizontal + v * vertical);
        let col = colour(&r);

        let ir = (255.99 * col.x) as u8;
        let ig = (255.99 * col.y) as u8;
        let ib = (255.99 * col.z) as u8;
        *pixel = image::Rgb([ir, ig, ib]);
    }
    img.save("out.png").unwrap();
}
