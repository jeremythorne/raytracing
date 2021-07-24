use glam::{Vec3, vec3};
use ordered_float::OrderedFloat;
use rand::{Rng};
use std::cmp::Eq;
use std::hash::{Hash, Hasher};

mod aabb;
mod camera;
mod kdtree;
mod geometry;
mod material;
mod ray;

pub use aabb::AABB;
pub use camera::Camera;
pub use geometry::*;
pub use kdtree::{KdTree, IndexedTriangle};
pub use material::*;
pub use ray::Ray;

pub struct HashableVec3(pub Vec3);

impl Hash for HashableVec3 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        OrderedFloat(self.0.x).hash(state);
        OrderedFloat(self.0.y).hash(state);
        OrderedFloat(self.0.z).hash(state);
    }
}

impl PartialEq for HashableVec3 {
    fn eq(&self, other: &Self) -> bool {
        OrderedFloat(self.0.x) == OrderedFloat(other.0.x) &&
        OrderedFloat(self.0.y) == OrderedFloat(other.0.y) &&
        OrderedFloat(self.0.z) == OrderedFloat(other.0.z)
    }
}

impl Eq for HashableVec3 {}

pub struct HitRecord <'a> {
    pub t:f32,
    pub p:Vec3,
    pub front_face: bool,
    pub normal:Vec3,
    pub material: &'a dyn Material
}

impl <'a> HitRecord<'a> {
    pub fn new(t: f32, p:Vec3, r: &Ray, outward_normal:Vec3,
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

pub trait Hitable {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord>;
    fn bounding_box(&self) -> Option<AABB>;
}

pub fn vec3_random(rng: &mut impl Rng, min:f32, max:f32) -> Vec3 {
    vec3(rng.gen_range(min..max),
                rng.gen_range(min..max),
                rng.gen_range(min..max))
}

pub fn random_in_unit_sphere(rng: &mut impl Rng) -> Vec3 {
    loop {
        let p = vec3_random(rng, -1., 1.);
        if p.length_squared() <= 1. {
            return p;
        }
    }
}

pub fn random_unit_vector(rng: &mut impl Rng) -> Vec3 {
    random_in_unit_sphere(rng).normalize()
}

pub fn random_in_unit_disk(rng: &mut impl Rng) -> Vec3 {
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

pub fn near_zero(v: Vec3) -> bool {
    let e = 1e-8;
    v.x.abs() < e && v.y.abs() < e && v.z.abs() < e
}

pub fn reflect(a: Vec3, b:Vec3) -> Vec3 {
    a - 2. * a.dot(b) * b
}

pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = (-uv).dot(n).min(1.);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}