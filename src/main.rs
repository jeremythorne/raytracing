use image;
use glam::{Vec3};
use rand::{Rng};
use rand::rngs::ThreadRng;
use std::rc::Rc;

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

struct AttenuatedRay {
    attenuation: Vec3,
    ray: Ray
}

trait Material {
    fn scatter(&self, r: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<AttenuatedRay>;
}

struct Lambertian {
    albedo: Vec3
}

impl Material for Lambertian {
    fn scatter(&self, _r: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<AttenuatedRay> {
        let mut scatter_direction = rec.normal + random_unit_vector(rng);

        if near_zero(scatter_direction) {
            scatter_direction = rec.normal;
        }

        Some(AttenuatedRay {
            attenuation: self.albedo,
            ray: Ray::new(rec.p, scatter_direction)
        })
    }

}

struct HitRecord {
    t:f32,
    p:Vec3,
    front_face: bool,
    normal:Vec3,
    material: Rc<dyn Material>
}

impl HitRecord {
    fn new(t: f32, p:Vec3, r: &Ray, outward_normal:Vec3,
            material: Rc<dyn Material>) -> HitRecord {
        let (front_face, normal) = HitRecord::face_normal(r, outward_normal);   
        HitRecord {
            t,
            p,
            front_face,
            normal,
            material
        }
    }

    fn face_normal(r: &Ray, outward_normal: Vec3) -> (bool, Vec3) {
        let front_face = r.direction().dot(outward_normal) < 0.;
        let normal = if front_face { outward_normal } else { -outward_normal };
        (front_face, normal)
    }
}

trait Hitable {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord>;
}

struct Sphere {
    centre: Vec3,
    radius: f32,
    material: Rc<dyn Material>
}

impl Sphere {
    fn new(centre:&[f32;3], radius:f32, material: Rc<dyn Material>) -> Sphere {
        Sphere {
            centre: Vec3::from_slice(centre),
            radius,
            material
        }
    }
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
                    return Some(HitRecord::new(
                        *temp, p, r,
                        (p - self.centre) / self.radius,
                        Rc::clone(&self.material)));
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

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3
}

impl Camera {
    fn new() -> Camera {
        Camera {
            origin: Vec3::ZERO,
            lower_left_corner: Vec3::new(-2., -1., -1.),
            horizontal: Vec3::new(4., 0., 0.),
            vertical: Vec3::new(0., 2., 0.)
        }
    }

    fn get_ray(&self, u:f32, v:f32) -> Ray {
        Ray::new(self.origin, self.lower_left_corner +
                            u * self.horizontal +
                            v * self.vertical -
                            self.origin)
    }
}

fn vec3_random(rng: &mut impl Rng, min:f32, max:f32) -> Vec3 {
    Vec3::new(rng.gen_range(min..max),
                rng.gen_range(min..max),
                rng.gen_range(min..max))
}

fn random_in_unit_sphere(rng: &mut impl Rng) -> Vec3 {
    loop {
        let p = vec3_random(rng, -1., 1.);
        if p.length_squared() <= 1. {
            return p;
        }
    }
}

fn random_unit_vector(rng: &mut impl Rng) -> Vec3 {
    random_in_unit_sphere(rng).normalize()
}

fn near_zero(v: Vec3) -> bool {
    let e = 1e-8;
    v.x.abs() < e && v.y.abs() < e && v.z.abs() < e
}

fn ray_colour(world: &dyn Hitable, r: &Ray, depth: i32, rng: &mut ThreadRng) -> Vec3 {
    if depth < 0 {
        return Vec3::ZERO;
    }

    if let Some(rec) = world.hit(&r, 0.001, f32::INFINITY) {
        if let Some(scattered) = rec.material.scatter(r, &rec, rng) {
            return scattered.attenuation * ray_colour(world, &scattered.ray, depth - 1, rng);
        } else {
            return Vec3::ZERO;
        }
    }
    let unit_direction = r.direction().normalize();
    let t = 0.5 * (unit_direction.y + 1.0);
    (1.0 - t) * Vec3::ONE + t * Vec3::new(0.5, 0.7, 1.0)
}

fn write_colour(colour:Vec3, samples_per_pixel:u32) -> [u8; 3] {
    let scale = 1.0 / (samples_per_pixel as f32);
    let scaled = colour * scale;
    let gamma_corrected = Vec3::new(scaled.x.sqrt(),
                                    scaled.y.sqrt(),
                                    scaled.z.sqrt());
    let two55 = 255. * gamma_corrected.clamp(Vec3::ZERO, Vec3::ONE);
    [
        two55.x as u8,
        two55.y as u8,
        two55.z as u8,
    ]
}

fn main() {
    let nx = 200;
    let ny = 100;
    let ns = 100;
    let max_depth = 50;
    let mut img = image::RgbImage::new(nx, ny);
    let mut rng = rand::thread_rng();

    let camera = Camera::new();

    let material:Rc<dyn Material> = Rc::new(Lambertian {
        albedo: Vec3::new(0.8, 0.8, 0.)
    });

    let sphere = Sphere::new(&[0., 0., -1.], 0.5, Rc::clone(&material));
    let sphere2 = Sphere::new(&[0., -100.5, -1.], 100., Rc::clone(&material));
    let sphere3 = Sphere::new(&[-0.5, -0.5, -0.5], 0.25, Rc::clone(&material));

    let hit_list = HitableList {
        list:vec![&sphere, &sphere2, &sphere3]
    };

    for (i, j, pixel) in img.enumerate_pixels_mut() {
        let mut col = Vec3::ZERO;
        for _s in 0..ns {
            let a:f32 = rng.gen();
            let b:f32 = rng.gen();
            let u = ((i as f32) + a) / (nx as f32);
            let v = 1.0 - ((j as f32) + b) / (ny as f32);
            let r = camera.get_ray(u, v);
            col += ray_colour(&hit_list, &r, max_depth, &mut rng);

        }
        *pixel = image::Rgb(write_colour(col, ns));
    }
    img.save("out.png").unwrap();
}
