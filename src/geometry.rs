use byteorder::{LittleEndian, ReadBytesExt};
use glam::{Vec3, vec3};
use std::collections::HashMap;
use std::fs::File;
use std::error::Error;
use std::io::{BufReader, Read};

use super::*;

pub struct Sphere<M:Material> {
    centre: Vec3,
    radius: f32,
    material: M
}

impl <M:Material> Sphere<M> {
    pub fn new(centre:&[f32;3], radius:f32, material: M) -> Sphere<M> {
        Sphere {
            centre: Vec3::from_slice(centre),
            radius,
            material
        }
    }
}

impl <M:Material> Hitable for Sphere<M> {
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
                        &self.material));
                }
            }
        }
        None
    }

    fn bounding_box(&self) -> Option<AABB> {
        let r = vec3(self.radius, self.radius, self.radius);
        return Some(AABB {
            minimum: self.centre - r,
            maximum: self.centre + r
        })
    }
}

struct Triangle {
    vertices: [Vec3; 3],
    normal: Vec3
}

struct TriangleHit {
    t: f32,
    p: Vec3,
    n: Vec3
}

impl Triangle {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<TriangleHit> {
        //Moller Trumbore algorithm from scratchapixel.com
        let v0v1 = self.vertices[1] - self.vertices[0];
        let v0v2 = self.vertices[2] - self.vertices[0];
        let pvec = r.direction().cross(v0v2);
        let det = v0v1.dot(pvec);
        if det.abs() < 0.0001 {
            // ray and triangle are parallel
            return None
        }
        let invdet = 1. / det;
        let tvec = r.origin() - self.vertices[0];
        let u = tvec.dot(pvec) * invdet;
        if u < 0. || u > 1. {
            return None
        }
        let qvec = tvec.cross(v0v1);
        let v = r.direction().dot(qvec) * invdet;
        if v < 0. || v > 1. {
            return None
        }

        let t = v0v2.dot(qvec) * invdet;
        if t > t_min && t < t_max {
            let p = r.point_at_parameter(t);
            Some(TriangleHit { t, p, n: self.normal })
        } else {
            None
        }
    }
}

pub struct Mesh {
    kd_tree: KdTree,
    vertices: Vec<Vec3>,
}

impl Mesh {
    pub fn cube() -> Mesh {
        let vertices = vec!(
            vec3(-0.5, -0.5, -0.5),
            vec3(-0.5, 0.5, -0.5),
            vec3(0.5, 0.5, -0.5),
            vec3(0.5, -0.5, -0.5),
            vec3(-0.5, -0.5, 0.5),
            vec3(-0.5, 0.5, 0.5),
            vec3(0.5, 0.5, 0.5),
            vec3(0.5, -0.5, 0.5),
        );

        let indexed_triangles = vec!(
            //// front
            IndexedTriangle { indices: [4, 6, 5], normal: vec3(0., 0., -1.) },
            //IndexedTriangle { indices: [4, 7, 6], normal: vec3(0., 0., -1.) },
            //// back
            //IndexedTriangle { indices: [3, 1, 2], normal: vec3(0., 0., 1.) },
            //IndexedTriangle { indices: [3, 0, 1], normal: vec3(0., 0., 1.) },
            //// left
            //IndexedTriangle { indices: [0, 5, 1], normal: vec3(-1., 0., 0.) },
            //IndexedTriangle { indices: [0, 4, 5], normal: vec3(-10., 0., 0.) },
            //// right
            //IndexedTriangle { indices: [7, 2, 6], normal: vec3(1., 0., 0.) },
            //IndexedTriangle { indices: [7, 3, 2], normal: vec3(1., 0., 0.) },
            //// top
            //IndexedTriangle { indices: [5, 2, 1], normal: vec3(0., 1., 0.) },
            //IndexedTriangle { indices: [5, 6, 2], normal: vec3(0., 1., 0.) },
            //// bottom
            //IndexedTriangle { indices: [7, 0, 3], normal: vec3(0., -1., 0.) },
            //IndexedTriangle { indices: [7, 4, 0], normal: vec3(0., -1., 0.) },
         );

         Mesh {
             kd_tree: KdTree::new(&indexed_triangles, &vertices),
             vertices
         }
    }

    pub fn load(fname: &str) -> Result<Mesh, Box<dyn Error>> {
        if !fname.ends_with(".stl") {
            Err("can only read stl files")?
        }
        let file = File::open(fname)?;
        let mut reader = BufReader::new(file);
        let mut header = [0u8; 80];
        reader.read(&mut header)?;
        if header.starts_with(b"solid") {
            Err("only binary stl files supported")?
        }
        let count = reader.read_u32::<LittleEndian>()?;
        println!("{} triangles", count);
        let mut triangles = Vec::<Triangle>::new();
        let mut vmin = Vec3::ZERO;
        let mut vmax = Vec3::ZERO;
        for c in 0..count {
            let mut coords = [0.; 3];
            for j in 0..3 {
                coords[j] = reader.read_f32::<LittleEndian>()?;
            }
            let normal = Vec3::from(coords);
            let normal = normal.normalize();
            let mut vertices = [Vec3::ZERO; 3];
            for i in 0..3 {
                for j in 0..3 {
                    coords[j] = reader.read_f32::<LittleEndian>()?;
                    if coords[j].abs() > 100. {
                        println!("{} {} {} coord {}", c, i, j, coords[j]);
                    }
                }
                // swap z, y
                vertices[i] = Vec3::new(coords[0], coords[2], coords[1]);
                if c == 0 && i == 0 {
                    vmin = vertices[i].clone();
                    vmax = vertices[i].clone();
                } else {
                    vmin = vmin.min(vertices[i]);
                    vmax = vmax.max(vertices[i]);
                }
            }
            let mut _attributes = [0u8; 2];
            reader.read(&mut _attributes)?;
            triangles.push(Triangle {
                vertices,
                normal
            });
        }
        println!("min {} max {}", vmin, vmax);
        let (indexed_triangles, vertices) = Mesh::to_indexed(triangles);
        let kd_tree = KdTree::new(&indexed_triangles, &vertices);
        Ok(Mesh {
            kd_tree,
            vertices
        })
    }

    fn to_indexed(triangles: Vec<Triangle>) -> (Vec<IndexedTriangle>, Vec<Vec3>) {
        let mut v2i = HashMap::<HashableVec3, usize>::new();
        let mut vertices = Vec::<Vec3>::new();
        let indexed_triangles = triangles.iter()
            .map(|t| {
                let mut indices = [0usize; 3];
                for j in 0..3 {
                    let v = t.vertices[j];
                    let hv = HashableVec3(v);
                    indices[j] = match v2i.get(&hv) {
                        Some(i) => {
                            *i
                        },
                        None => {
                            let i = vertices.len();
                            vertices.push(v);
                            v2i.insert(hv, i);
                            i
                        }
                    }
                }
                IndexedTriangle {
                    indices,
                    normal: t.normal
                }
            }).collect();
        (indexed_triangles, vertices) 
    }

    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<TriangleHit> {
        let indexed_triangles = self.kd_tree.hit(r, t_min, t_max);
        if indexed_triangles.is_none() {
            return None;
        }
        let indexed_triangles = indexed_triangles.unwrap();
        let mut rec:Option<TriangleHit> = None;
        let mut closest_so_far = t_max;
        for it in indexed_triangles.iter() {
            let mut vertices = [Vec3::ZERO; 3];
            for i in 0..3 {
                vertices[i] = self.vertices[it.indices[i]];
            }
            let triangle = Triangle {
                vertices,
                normal: it.normal.clone()
            };
            if let Some(temp_rec) = triangle.hit(r, t_min, closest_so_far) {
                closest_so_far = temp_rec.t;
                rec = Some(temp_rec);
            }     
        }
        rec
    }

    fn bounding_box(&self) -> Option<AABB> {
        Some(self.kd_tree.aabb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_indexed() {
        let vertices = [Vec3::ZERO, Vec3::ONE, vec3(0., 0., 1.)];
        let triangles = vec!(
            Triangle { vertices: vertices.clone(), normal: Vec3::ONE },
            Triangle { vertices: vertices.clone(), normal: Vec3::ONE }
        );
        let (indexed, out_vertices) = Mesh::to_indexed(triangles);
        assert_eq!(indexed.len(), 2);
        assert_eq!(out_vertices.len(), 3);
        assert_eq!(indexed[0].normal, Vec3::ONE);
        for t in indexed {
            for i in 0..3 {
                assert_eq!(out_vertices[t.indices[i]], vertices[i]);
            }
        }
    }
}

pub struct MaterialMesh<M:Material> {
    pub mesh: Mesh,
    pub material: M
}

impl <M:Material> Hitable for MaterialMesh<M> {
    fn hit(&self, r: &Ray, t_min:f32, t_max:f32) -> Option<HitRecord> {
        let rec = self.mesh.hit(r, t_min, t_max);
        if let Some(t) = rec {
            Some(HitRecord::new(t.t, t.p, r, t.n, &self.material))
        } else {
            None
        }
    }

    fn bounding_box(&self) -> Option<AABB> {
        self.mesh.bounding_box()
    }
}

pub struct HitableList {
    list:Vec<Box<dyn Hitable>>
}

impl HitableList {
    pub fn new() -> HitableList {
        HitableList {
            list: vec![]
        }
    }

    pub fn push(&mut self, hitable: impl Hitable + 'static) {
        self.list.push(Box::new(hitable));
    }
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

    fn bounding_box(&self) -> Option<AABB> {
        let mut a:Option<AABB> = None;
        for h in self.list.iter() {
            if let Some(b) = h.bounding_box() {
                if let Some(aa) = a {
                    a = Some(aa.surround(&b));
                } else {
                    a = Some(b);
                }
            }
        }
        a
    }
}