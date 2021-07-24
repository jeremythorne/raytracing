use image;
use glam::{Vec3, vec3};
use rand::{Rng};
use rand::rngs::ThreadRng;
use raytracing::*;

fn _random_scene(rng: &mut ThreadRng) -> HitableList {
    let mut list = HitableList::new();
    let ground_material = Lambertian { albedo:vec3(0.5, 0.5, 0.5) };
    list.push(Sphere::new(&[0., -1000., 0.], 1000.0, ground_material));
    
    list.push(Sphere::new(&[0., 1., 0.], 1., Dialectric { ir: 1.5 } ));
    list.push(Sphere::new(&[-4., 1., 0.], 1.,
                        Lambertian { albedo: vec3(0.4, 0.2, 0.1) } ));
    list.push(Sphere::new(&[4., 1., 0.], 1.,
                        Metal { albedo: vec3(0.7, 0.6, 0.5), fuzz: 0. } ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat:f32 = rng.gen();
            let centre = 0.9 * vec3(rng.gen(), 0., rng.gen()) +
                vec3(a as f32, 0.2, b as f32);
            
            if (centre - vec3(4., 0.2, 0.)).length() > 0.9 {
                if choose_mat < 0.8 {
                    let albedo = vec3_random(rng, 0., 1.) * vec3_random(rng, 0., 1.);
                    list.push(Sphere::new(&centre.to_array(), 0.2, Lambertian{ albedo }));
                } else if choose_mat < 0.95 {
                    let albedo = vec3_random(rng, 0.5, 1.);
                    let fuzz = rng.gen_range(0.0..0.5);
                    list.push(Sphere::new(&centre.to_array(), 0.2, Metal { albedo, fuzz }));
                } else {
                    list.push(Sphere::new(&centre.to_array(), 0.2, Dialectric { ir: 1.5}));
                };
             }
        }
    }

    list
}

fn ray_colour(world: &dyn Hitable, r: &Ray, depth: i32, rng: &mut ThreadRng) -> Vec3 {
    if depth < 0 {
        return Vec3::ZERO;
    }

    if let Some(rec) = world.hit(&r, 0.001, 1000.0) {
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

    let look_from = vec3(6., 2., 6.);
    let look_at = vec3(0., 0., 0.);
    let camera = Camera::new(
        look_from,
        look_at,
        vec3(0., 1., 0.),
        20., aspect_ratio,
        0.1,
        10.);


    let mug = MaterialMesh{
            mesh: Mesh::load("mug.stl").unwrap(), 
            // mesh: Mesh::cube_of_cubes(5), 
            material: Metal{ albedo:vec3(1., 1., 0.2), fuzz: 0.2}
    };

    let cube = MaterialMesh{
            mesh: Mesh::cube(1., vec3(2., 0.5, 0.)), 
            material: Lambertian{ albedo:vec3(0.2, 1., 0.2)}
    };

    let cube2 = MaterialMesh{
            mesh: Mesh::cube(0.5, vec3(0., 0.25, 2.)), 
            material: Lambertian { albedo:vec3(1., 0.2, 0.2) }
    };
    //let world = random_scene(&mut rng);
    let mut world = HitableList::new();
    world.push(mug);
    world.push(cube);
    world.push(cube2);
    world.push(Sphere::new(&[0., -100., 0.], 100., Lambertian{ albedo:vec3( 0.2, 0.3, 0.5 )}));

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
