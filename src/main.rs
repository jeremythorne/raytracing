use image;
use glam::{Vec3};
use rand::{Rng};
use rand::rngs::ThreadRng;

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

struct Metal {
    albedo: Vec3,
    fuzz: f32
}

impl Material for Metal {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<AttenuatedRay> {
        let reflected = reflect(r_in.direction().normalize(), rec.normal) +
                self.fuzz * random_in_unit_sphere(rng);

        if reflected.dot(rec.normal) < 0. {
            None
        } else {
            Some(AttenuatedRay {
                attenuation: self.albedo,
                ray: Ray::new(rec.p, reflected)
            })
        }
    }
}

struct Dialectric {
    ir: f32
}

impl Dialectric {
    fn reflectance(&self, cosine:f32, ref_idx:f32) -> f32 {
        // Schlick's approximation
        let r0 = (1. - ref_idx) / (1. + ref_idx);
        let r02 = r0 * r0;
        r02 + (1. - r0) * (1. - cosine).powi(5)
    }
}

impl Material for Dialectric {
    fn scatter(&self, r_in: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<AttenuatedRay> {

        let refraction_ratio = if rec.front_face { 1.0/self.ir } else { self.ir };
        let unit_direction = r_in.direction().normalize();

        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = sin_theta * refraction_ratio > 1.;
        let direction = if cannot_refract || self.reflectance(cos_theta, refraction_ratio) > rng.gen() {
            reflect(unit_direction, rec.normal)
        } else {
            refract(unit_direction, rec.normal, refraction_ratio)
        };
        Some(AttenuatedRay {
            attenuation: Vec3::ONE,
            ray: Ray::new(rec.p, direction)
        })
    }
}
struct HitRecord<'a> {
    t:f32,
    p:Vec3,
    front_face: bool,
    normal:Vec3,
    material: &'a dyn Material
}

impl <'a> HitRecord<'a> {
    fn new(t: f32, p:Vec3, r: &Ray, outward_normal:Vec3,
            material: &'a dyn Material) -> HitRecord<'a> {
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

struct Sphere<'a> {
    centre: Vec3,
    radius: f32,
    material: &'a dyn Material
}

impl <'a> Sphere<'a> {
    fn new(centre:&[f32;3], radius:f32, material: &'a dyn Material) -> Sphere<'a> {
        Sphere {
            centre: Vec3::from_slice(centre),
            radius,
            material
        }
    }
}

impl <'a> Hitable for Sphere<'a> {
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
                        self.material));
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

fn reflect(a: Vec3, b:Vec3) -> Vec3 {
    a - 2. * a.dot(b) * b
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-uv).dot(n).min(1.);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
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

    let yellow = Lambertian {
        albedo: Vec3::new(0.8, 0.8, 0.)
    };

    let glass = Dialectric {
        ir: 1.5
    };

    let blue_metal = Metal {
        albedo: Vec3::new(0.0, 0.8, 0.4),
        fuzz: 0.2
    };

    let sphere = Sphere::new(&[0., 0., -1.], 0.5, &glass);
    let ground_sphere = Sphere::new(&[0., -100.5, -1.], 100., &yellow);
    let sphere3 = Sphere::new(&[-0.5, -0.25, -1.0], 0.25, &blue_metal);

    let hit_list = HitableList {
        list:vec![&sphere, &ground_sphere, &sphere3]
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
