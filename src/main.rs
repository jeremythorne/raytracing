use image;
use glam::{Vec3};

struct Ray {
    a: Vec3,
    b: Vec3
}

impl Ray {
    fn new(a: Vec3, b:Vec3) -> Ray {
        Ray { a, b }
    }

    fn origin(&self) -> Vec3 {
        self.a
    }

    fn direction(&self) -> Vec3 {
        self.b
    }

    fn point_at_parameter(&self, t:f32) -> Vec3 {
        self.a + t * self.b
    }
}

struct HitRecord {
    t:f32,
    p:Vec3,
    normal:Vec3
}

trait Hitable {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord>;
}

struct Sphere {
    centre: Vec3,
    radius: f32
}

impl Hitable for Sphere {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord> {
        let oc = r.origin() - self.centre;
        let a = r.direction().dot(r.direction());
        let b = 2.0 * oc.dot(r.direction());
        let c = oc.dot(oc) - self.radius * self.radius;
        let discriminant = b * b - 4. * a * c;
        if discriminant > 0. {
            for temp in [
                (-b - discriminant.sqrt()) / (2. * a),
                (-b + discriminant.sqrt()) / (2. * a)].iter() {
                if *temp < t_max && *temp > t_min {
                    let p = r.point_at_parameter(*temp);
                    return Some(HitRecord {
                        t: *temp,
                        p: p,
                        normal: (p - self.centre) / self.radius
                    });
                }
            }
        }
        None
    }
}

struct HitableList<'a> {
    list:Vec<&'a dyn Hitable>
}

impl <'a> Hitable for HitableList<'a> {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord> {
        let mut rec:Option<HitRecord> = None;
        let mut closest_so_far = t_max;
        for hitable in self.list.iter() {
            if let Some(temp_rec) = hitable.hit(r, t_min, closest_so_far) {
                closest_so_far = temp_rec.t;
                rec = Some(temp_rec);
            }     
        }
        rec
    }
}

fn colour(hitable: &dyn Hitable, r: &Ray) -> Vec3 {
    let ot = hitable.hit(&r, 0.0, f32::INFINITY);
    if let Some(t) = ot {
        let n = t.normal;
        return 0.5 * Vec3::new(n.x + 1., n.y + 1., n.z + 1.)
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

    let sphere = Sphere {
        centre: Vec3::new(0., 0., -1.),
        radius: 0.5
    };
    let sphere2 = Sphere {
        centre: Vec3::new(0., -100.5, -1.),
        radius: 100.
    };

    let hit_list = HitableList {
        list:vec![&sphere, &sphere2]
    };

    for (i, j, pixel) in img.enumerate_pixels_mut() {
        let u = (i as f32) / (nx as f32);
        let v = 1.0 - (j as f32) / (ny as f32);
        let r = Ray::new(origin, lower_left_corner + u * horizontal + v * vertical);
        let col = colour(&hit_list, &r);

        let ir = (255.99 * col.x) as u8;
        let ig = (255.99 * col.y) as u8;
        let ib = (255.99 * col.z) as u8;
        *pixel = image::Rgb([ir, ig, ib]);
    }
    img.save("out.png").unwrap();
}
