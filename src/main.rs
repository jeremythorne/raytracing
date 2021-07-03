use image;
use glam::{Vec3};
use rand::Rng;

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

impl Sphere {
    fn new(centre:&[f32;3], radius:f32) -> Sphere {
        Sphere {
            centre: Vec3::from_slice(centre),
            radius
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

fn vec3_random<R: Rng>(rng: &mut R, min:f32, max:f32) -> Vec3 {
    Vec3::new(rng.gen_range(min..max),
                rng.gen_range(min..max),
                rng.gen_range(min..max))
}

fn random_in_unit_sphere<R: Rng>(rng: &mut R) -> Vec3 {
    loop {
        let p = vec3_random(rng, -1., 1.);
        if p.length_squared() <= 1. {
            return p;
        }
    }
}

fn random_unit_vector<R: Rng>(rng: &mut R) -> Vec3 {
    random_in_unit_sphere(rng).normalize()
}

fn ray_colour<R: Rng>(world: &dyn Hitable, r: &Ray, depth: i32, rng: &mut R) -> Vec3 {
    if depth < 0 {
        return Vec3::ZERO;
    }

    if let Some(rec) = world.hit(&r, 0.001, f32::INFINITY) {
        let target = rec.p + rec.normal + random_unit_vector(rng);
        return 0.5 * ray_colour(world, &Ray::new(rec.p, target - rec.p), depth - 1, rng);
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

    let sphere = Sphere::new(&[0., 0., -1.], 0.5);
    let sphere2 = Sphere::new(&[0., -100.5, -1.], 100.);
    let sphere3 = Sphere::new(&[-0.5, -0.5, -0.5], 0.25);

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
