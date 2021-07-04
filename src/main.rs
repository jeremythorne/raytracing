use image;
use glam::{Vec3, vec3};
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

struct HitableList {
    list:Vec<Rc<dyn Hitable>>
}

impl Hitable for HitableList {
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
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    _w: Vec3,
    lens_radius: f32
}

impl Camera {
    fn new(look_from:Vec3,
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

    fn get_ray(&self, s:f32, t:f32, rng: &mut ThreadRng) -> Ray {
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

fn vec3_random(rng: &mut impl Rng, min:f32, max:f32) -> Vec3 {
    vec3(rng.gen_range(min..max),
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

fn random_in_unit_disk(rng: &mut impl Rng) -> Vec3 {
    loop {
        let p = vec3(
            rng.gen_range((-1.)..1.),
            rng.gen_range((-1.)..1.),
            0.);
        if p.length_squared() <= 1. {
            return p;
        }
    }
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

fn random_scene(rng: &mut ThreadRng) -> HitableList {
    let mut list = Vec::<Rc<dyn Hitable>>::new();
    let ground_material = Rc::new(Lambertian { albedo:vec3(0.5, 0.5, 0.5) });
    list.push(Rc::new(Sphere::new(&[0., -1000., 0.], 1000.0, ground_material)));
    
    list.push(Rc::new(Sphere::new(&[0., 1., 0.], 1.,
                        Rc::new(Dialectric { ir: 1.5 } ))));
    list.push(Rc::new(Sphere::new(&[-4., 1., 0.], 1.,
                        Rc::new(Lambertian { albedo: vec3(0.4, 0.2, 0.1) } ))));
    list.push(Rc::new(Sphere::new(&[4., 1., 0.], 1.,
                        Rc::new(Metal { albedo: vec3(0.7, 0.6, 0.5), fuzz: 0. } ))));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat:f32 = rng.gen();
            let centre = 0.9 * vec3(rng.gen(), 0., rng.gen()) +
                vec3(a as f32, 0.2, b as f32);
            
            if (centre - vec3(4., 0.2, 0.)).length() > 0.9 {
                let material:Rc<dyn Material> =
                if choose_mat < 0.8 {
                    let albedo = vec3(rng.gen(), rng.gen(), rng.gen()) *
                                 vec3(rng.gen(), rng.gen(), rng.gen());
                    Rc::new(Lambertian{ albedo })
                } else if choose_mat < 0.95 {
                    let albedo = vec3_random(rng, 0.5, 1.);
                    let fuzz = rng.gen_range(0.0..0.5);
                    Rc::new(Metal { albedo, fuzz })
                } else {
                    Rc::new(Dialectric { ir: 1.5})
                };
                list.push(Rc::new(Sphere::new(&centre.to_array(), 0.2, material)));
             }
        }
    }

    HitableList {
        list
    }
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
    (1.0 - t) * Vec3::ONE + t * vec3(0.5, 0.7, 1.0)
}

fn write_colour(colour:Vec3, samples_per_pixel:u32) -> [u8; 3] {
    let scale = 1.0 / (samples_per_pixel as f32);
    let scaled = colour * scale;
    let gamma_corrected = vec3(scaled.x.sqrt(),
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
    let samples_per_pixel = 100;
    let max_depth = 50;
    let mut rng = rand::thread_rng();

    let aspect_ratio = 16. / 9.;
    let image_width = 400;
    let image_height = (image_width as f32 / aspect_ratio) as u32;

    let mut img = image::RgbImage::new(image_width, image_height);

    let look_from = vec3(13., 2., 3.);
    let look_at = vec3(0., 0., 0.);
    let camera = Camera::new(
        look_from,
        look_at,
        vec3(0., 1., 0.),
        20., aspect_ratio,
        0.1,
        10.);


    let world = random_scene(&mut rng);
    let hit_list = &world;

    let total_pixels = image_width * image_height;

    for (i, j, pixel) in img.enumerate_pixels_mut() {
        let mut pixel_colour = Vec3::ZERO;
        for _s in 0..samples_per_pixel {
            let a:f32 = rng.gen();
            let b:f32 = rng.gen();
            let u = ((i as f32) + a) / (image_width as f32 - 1.);
            let v = 1.0 - ((j as f32) + b) / (image_height as f32 - 1.);
            let r = camera.get_ray(u, v, &mut rng);
            pixel_colour += ray_colour(hit_list, &r, max_depth, &mut rng);

        }
        *pixel = image::Rgb(write_colour(pixel_colour, samples_per_pixel));
        let percentage_complete = (j * image_width + i) * 100 / total_pixels;
        print!("\r{}% complete\r", percentage_complete);
    }
    println!("");
    img.save("out.png").unwrap();
}
