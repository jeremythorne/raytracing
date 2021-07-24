use glam::Vec3;
use crate::{AABB, Ray};

#[derive(Copy, Clone)]
pub struct IndexedTriangle {
    pub indices: [usize; 3],
    pub normal: Vec3
}

enum Content {
    Leaf(Vec<IndexedTriangle>),
    Branch(Box<Node>, Box<Node>)
}

struct Node {
    child: Content,
}

impl Node {
    fn new(triangles: &Vec<IndexedTriangle>, vertices: &Vec<Vec3>,
            vmin: Vec3, vmax: Vec3, level: u32) -> Node {
        let axis = (level % 3) as usize;
        let plane = (vmax[axis] + vmin[axis]) / 2.;

        println!("level {}, axis {}, plane {}, num triangles {}",
            level, axis, plane, triangles.len());

        if triangles.len() < 100 || level > 10 {
            return Node {
                child: Content::Leaf(triangles.clone().to_vec())
            }
        } else {
            let mut left_vmax = vmax.clone();
            left_vmax[axis] = plane;
            let left_triangles = triangles.iter()
                .filter(|a|
                    vertices[a.indices[0]][axis] < plane ||
                    vertices[a.indices[1]][axis] < plane ||
                    vertices[a.indices[2]][axis] < plane
                )
                .cloned().collect();
            let mut right_vmin = vmin.clone();
            right_vmin[axis] = plane;
            let right_triangles = triangles.iter()
                .filter(|a|
                    vertices[a.indices[0]][axis] >= plane ||
                    vertices[a.indices[1]][axis] >= plane ||
                    vertices[a.indices[2]][axis] >= plane
                )
                .cloned().collect();
            return Node {
                child: Content::Branch(
                    Box::<Node>::new(Node::new(&left_triangles, vertices,
                        vmin, left_vmax, level + 1)),
                    Box::<Node>::new(Node::new(&right_triangles, vertices,
                        right_vmin, vmax, level + 1))
                )
            }
        }
    }

    fn hit(&self, vt_min:Vec3, vt_max:Vec3, level:u32) ->Option<Vec<IndexedTriangle>> {
        if !AABB::overlap(vt_min, vt_max) {
            return None
        }
        match &self.child {
            Content::Leaf(triangles) => { 
                return Some(triangles.clone());
            },
            Content::Branch(left, right) => {
                let axis = (level % 3) as usize;
                let t_mid = (vt_min[axis] + vt_max[axis]) / 2.;
                let mut left_vt_max = vt_max.clone();
                left_vt_max[axis] = t_mid;
                let left_triangles = left.hit(vt_min, left_vt_max, level + 1);
                let mut right_vt_min = vt_min.clone();
                right_vt_min[axis] = t_mid;
                let right_triangles = right.hit(right_vt_min, vt_max, level + 1);
                
                match left_triangles {
                    Some(mut lt) => {
                        match right_triangles {
                            Some(mut rt) => {
                                lt.append(&mut rt);
                                Some(lt)
                            },
                            None => {
                                Some(lt)
                            }
                        }
                    },
                    None => {
                        right_triangles.clone()
                    }
                }
            }
        }
    }
}

pub struct KdTree {
    root: Node,
    pub aabb: AABB
}

impl KdTree {
    pub fn new(triangles: &Vec<IndexedTriangle>, vertices: &Vec<Vec3>) -> KdTree {
        let mut vmin = vertices[0].clone();
        let mut vmax = vertices[0].clone();
        for vertex in vertices {
            vmin = vmin.min(*vertex);
            vmax = vmax.max(*vertex);
        }
        KdTree {
            root: Node::new(triangles, vertices, vmin, vmax, 0),
            aabb: AABB { minimum: vmin, maximum: vmax}
        }
    }

    pub fn hit(&self, r:&Ray, t_min: f32, t_max:f32) -> Option<Vec<IndexedTriangle>> {
        let (vt_min, vt_max) = self.aabb.hit_t(r);
        let vt_min = vt_min.max(Vec3::splat(t_min));
        let vt_max = vt_max.min(Vec3::splat(t_max));
        self.root.hit(vt_min, vt_max, 0)
    }
}