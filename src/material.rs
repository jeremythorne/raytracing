use glam::Vec3;
use rand::rngs::ThreadRng;
use crate::ray::{AttenuatedRay, Ray};
use super::*;

pub trait Material {
    fn scatter(&self, r: &Ray, rec: &HitRecord, rng: &mut ThreadRng) -> Option<AttenuatedRay>;
}

pub struct Lambertian {
    pub albedo: Vec3
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

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f32
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

pub struct Dialectric {
    pub ir: f32
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