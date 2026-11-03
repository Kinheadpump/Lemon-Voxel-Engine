use glam::{IVec3, UVec3};
use vx_core::ChunkPos;

pub struct ChunkWindow<T> {
    extent: IVec3,
    slots: Vec<Option<(ChunkPos, T)>>,
}

impl<T> ChunkWindow<T> {
    pub fn new(extent: UVec3) -> Self {
        assert!(extent.min_element() > 0, "window extent must be non-zero");
        let slot_count = (extent.x * extent.y * extent.z) as usize;
        Self {
            extent: extent.as_ivec3(),
            slots: std::iter::repeat_with(|| None).take(slot_count).collect(),
        }
    }

    pub fn for_view_radius(horizontal_chunks: u32, vertical_chunks: u32) -> Self {
        let horizontal = 2 * horizontal_chunks + 1;
        Self::new(UVec3::new(horizontal, 2 * vertical_chunks + 1, horizontal))
    }

    #[inline]
    fn slot_index(&self, pos: ChunkPos) -> usize {
        let wrapped = pos.0.rem_euclid(self.extent);
        ((wrapped.y * self.extent.z + wrapped.z) * self.extent.x + wrapped.x) as usize
    }

    #[inline]
    pub fn get(&self, pos: ChunkPos) -> Option<&T> {
        match &self.slots[self.slot_index(pos)] {
            Some((stored, value)) if *stored == pos => Some(value),
            _ => None,
        }
    }

    #[inline]
    pub fn get_mut(&mut self, pos: ChunkPos) -> Option<&mut T> {
        let index = self.slot_index(pos);
        match &mut self.slots[index] {
            Some((stored, value)) if *stored == pos => Some(value),
            _ => None,
        }
    }

    pub fn insert(&mut self, pos: ChunkPos, value: T) -> Option<(ChunkPos, T)> {
        let index = self.slot_index(pos);
        self.slots[index].replace((pos, value))
    }

    pub fn remove(&mut self, pos: ChunkPos) -> Option<T> {
        let index = self.slot_index(pos);
        match &self.slots[index] {
            Some((stored, _)) if *stored == pos => self.slots[index].take().map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn extent(&self) -> UVec3 {
        self.extent.as_uvec3()
    }

    pub fn iter(&self) -> impl Iterator<Item = (ChunkPos, &T)> {
        self.slots
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(pos, value)| (*pos, value)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inserting_a_wrapped_position_evicts_the_old_occupant() {
        let mut window = ChunkWindow::for_view_radius(2, 1);
        assert!(window.insert(ChunkPos::new(-2, 0, 1), 7).is_none());
        assert_eq!(window.get(ChunkPos::new(-2, 0, 1)), Some(&7));
        assert_eq!(window.get(ChunkPos::new(3, 0, 1)), None);
        let evicted = window.insert(ChunkPos::new(3, 0, 1), 9);
        assert_eq!(evicted, Some((ChunkPos::new(-2, 0, 1), 7)));
        assert_eq!(window.remove(ChunkPos::new(-2, 0, 1)), None);
        assert_eq!(window.remove(ChunkPos::new(3, 0, 1)), Some(9));
    }
}
