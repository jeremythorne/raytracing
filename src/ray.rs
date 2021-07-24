use glam::Vec3;

pub struct Ray {
    a: Vec3,
    b: Vec3
}

impl Ray {
    pub fn new(a: Vec3, b:Vec3) -> Ray {
        Ray { a, b }
    }

    pub fn origin(&self) -> Vec3 {
        self.a
    }

    pub fn direction(&self) -> Vec3 {
        self.b
    }

    pub fn point_at_parameter(&self, t:f32) -> Vec3 {
        self.a + t * self.b
    }
}

pub struct AttenuatedRay {
    pub attenuation: Vec3,
    pub ray: Ray
}