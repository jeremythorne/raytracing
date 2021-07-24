use glam::Vec3;
use crate::ray::Ray;

#[derive(Clone, Copy)]
pub struct AABB {
    pub minimum: Vec3,
    pub maximum: Vec3
}

impl AABB {
    pub fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> bool {
        let mut tm = t_min;
        let mut tx = t_max;
        for i in 0..3 {
            let invd = 1. / r.direction()[i];
            let mut t0 = (self.minimum[i] - r.origin()[i]) * invd;
            let mut t1 = (self.maximum[i] - r.origin()[i]) * invd;
            if invd < 0. {
                let t = t0;
                t0 = t1;
                t1 = t;
            }
            tm = tm.max(t0);
            tx = tx.min(t1);
            if tx < tm {
                return false;
            }
        }
        true
    }

    // return tmin, tmax for all three axes
    pub fn hit_t(&self, r:&Ray) -> (Vec3, Vec3) {
        let mut tm = Vec3::ZERO;
        let mut tx = Vec3::ZERO;
        for i in 0..3 {
            let invd = 1. / r.direction()[i];
            let mut t0 = (self.minimum[i] - r.origin()[i]) * invd;
            let mut t1 = (self.maximum[i] - r.origin()[i]) * invd;
            if invd < 0. {
                let t = t0;
                t0 = t1;
                t1 = t;
            }
            tm[i] = t0;
            tx[i] = t1;
        }
        (tm, tx)
    }

    pub fn overlap(t_min: Vec3, t_max: Vec3) -> bool {
        if t_min.is_nan() || t_max.is_nan() {
            panic!("t_min {} t_max {}", t_min, t_max);
        }
        let tmax_of_min = t_min.max_element();
        let tmin_of_max = t_max.min_element();
        tmin_of_max > tmax_of_min
    }

    pub fn surround(&self, other: &AABB) -> AABB {
        AABB {
            minimum: self.minimum.min(other.minimum),
            maximum: self.maximum.max(other.maximum)
        }
    }

    pub fn volume(&self) -> f32 {
        let diff = self.maximum - self.minimum;
        diff.x * diff.y * diff.z
    }
}

#[cfg(test)]
mod tests {
    use glam::{vec3};
    use super::*;

    #[test]
    fn test_hit() {
        let r = Ray::new(vec3(0., 0., -2.), vec3(0., 0., 1.));
        let aabb = AABB{ minimum: vec3(-1., -1., -1.), maximum: vec3(1., 1., 1.)};
        assert_eq!(aabb.hit(&r, 0., 3.), true);
    }

    #[test]
    fn test_overlap_same() {
        let a = vec3(0., 0., 0.);
        let b = vec3(1., 1., 1.);
        assert_eq!(AABB::overlap(a, b), true);
    }

    #[test]
    fn test_overlap() {
        let a = vec3(0., 0.8, 0.9);
        let b = vec3(1., 1.5, 1.);
        assert_eq!(AABB::overlap(a, b), true);
    }

    #[test]
    fn test_no_overlap() {
        let a = vec3(0., 0.8, 0.9);
        let b = vec3(0.7, 1.5, 1.);
        assert_eq!(AABB::overlap(a, b), false);
    }

    #[test]
    fn test_hit_t() {
        let r = Ray::new(vec3(0., 0., -2.), vec3(0., 0., 1.));
        let aabb = AABB{ minimum: vec3(-1., -1., -1.), maximum: vec3(1., 1., 1.)};
        let (a, b) = aabb.hit_t(&r);
        assert_eq!(AABB::overlap(a, b), true);
    } 

    #[test]
    fn test_no_hit_t() {
        let r = Ray::new(vec3(2., 0., -2.), vec3(0., 0., 1.));
        let aabb = AABB{ minimum: vec3(-1., -1., -1.), maximum: vec3(1., 1., 1.)};
        let (a, b) = aabb.hit_t(&r);
        assert_eq!(AABB::overlap(a, b), false);
    }

    #[test]
    fn test_half_hit_t() {
        // vector parallel to x-axis
        let r = Ray::new(vec3(-2., 0.5, 0.5), vec3(1., 0., 0.));
        // unit cube
        let aabb = AABB{ minimum: vec3(0., 0., 0.), maximum: vec3(1., 1., 1.)};
        let (a, b) = aabb.hit_t(&r);
        assert_eq!(AABB::overlap(a, b), true);

        // now divide the cube in two down the x-axis
        let mut right2 = aabb.minimum.clone();
        right2[0] = (aabb.minimum[0] + aabb.maximum[0]) / 2.;
        let right_aabb = AABB{ minimum: right2, maximum: aabb.maximum};
        let (a2, b2) = right_aabb.hit_t(&r);
        // minimum t value on x-axis should be mid point between a[0] and b[0]
        let mid = (a[0] + b[0]) / 2.;
        assert_eq!(a2[0], mid);
        assert_eq!(b2, b);
        assert_eq!(AABB::overlap(a2, b2), true);
    } 


}
